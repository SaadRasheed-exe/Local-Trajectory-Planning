from core.types import *
from anytree import NodeMixin
from anytree.search import findall
from planner.motion_primitives import MotionPrimitive
from planner.motion_primitives import get_motion_primitives
from planner.state_node import StateNode
from shapely.geometry import Polygon
from core.dynamics import kinematic_bicycle
import heapq
import math
from omegaconf import DictConfig, OmegaConf
from utils.helper import convert_cfg_to_native
import time

from core.geometry import get_magnitude, get_signed_magnitude
from utils.helper import lane_blocked_ahead
from planner.cost.cost import calculate_node_cost, calculate_heuristic_cost, calculate_total_cost

class IDProvider:
    def __init__(self, start_id=0):
        self.current_id = start_id

    def get_id(self):
        result = self.current_id
        self.current_id += 1
        return result


# Minimum speed [m/s] used for motion-primitive expansion. Below this the
# acceleration range degenerates to deadbeat braking (v -> 0), so a stopped
# vehicle could never generate forward motion and the search would deadlock.
_MIN_PLANNING_SPEED: float = 1.5


import math
import time
import heapq
from typing import List, Optional

def plan(request: PlanningRequest, cfg: DictConfig, debug: bool = False) -> PlanResult:
    """
    Computes a trajectory using a Hybrid A* search algorithm.

    This function expands a search tree from the vehicle's starting state towards a goal region. 
    It evaluates dynamically feasible motion primitives and prioritizes nodes based on 
    accumulated path costs and a heuristic estimate to the goal. The search is constrained 
    by physical limits, a maximum horizon depth, and a strict time budget.

    Args:
        request (PlanningRequest): The planning request containing the initial state, 
                                   the environment, the goal region, and target speed.
        cfg (DictConfig): The Hydra configuration object containing parameters for the 
                          planner, cost weights, motion primitives, and vehicle dynamics.
        debug (bool): If True, prints detailed performance KPIs and statistics to the console.

    Returns:
        PlanResult: An object containing the success flag, the final trajectory (if successful), 
                    the total cost, and a diagnostic status message.
    """

    cfg = convert_cfg_to_native(cfg)

    open_nodes_pq: list[StateNode] = []
    end_nodes_pq: list[StateNode] = []
    id_provider = IDProvider()

    max_depth = math.ceil(cfg.planner.horizon / cfg.planner.dt_sim)
    call_time = time.perf_counter()
    exit_on_first = bool(getattr(cfg.planner, "exit_on_first_solution", False))
    # Accepting the very first full-depth terminal lets whichever primitive
    # chain reaches max depth fastest win, even when its cost is far worse
    # than alternatives (e.g. road-edge-hugging arcs). Collecting at least
    # min_terminals terminals before stopping keeps cycles short while
    # restoring basic solution quality.
    min_terminals = max(1, int(getattr(cfg.planner, "min_terminals", 1)))
    terminals_found = 0
    solution_found = False

    # Greedy dive bias for exit-on-first mode: with terminals only reachable at
    # full depth, plain f-ordering floods the exponentially wide shallow
    # frontier before any terminal matures. Subtracting a per-level discount
    # from the sort key makes deeper nodes pop first while same-depth nodes
    # still compete on true cost.
    progress_bonus = float(getattr(cfg.cost.search, "progress_bonus", 0.0)) if exit_on_first else 0.0

    # Return-mode gate: with no blocker ahead the overtake is complete, so a
    # strong opposite-lane penalty pushes the ego back into its original lane.
    # While a blocker remains ahead, penalties stay untouched so passing is
    # never discouraged. Evaluated once per planning cycle (not per node).
    ret_cfg = getattr(cfg.cost, "cost_opposite_lane_return_mode", None)
    if ret_cfg is not None and not lane_blocked_ahead(
        start_state=request.start_state,
        predicted_env=request.environment,
        veh_cfg=cfg.vehicle,
        lookahead_m=float(ret_cfg.lookahead_m),
        corridor_half_width_m=float(ret_cfg.corridor_half_width_m),
        speed_margin_mps=float(ret_cfg.speed_margin_mps),
        rear_hold_m=float(ret_cfg.rear_hold_m),
    ):
        opposite_lane_weight_scale = float(ret_cfg.multiplier)
    else:
        opposite_lane_weight_scale = 1.0

    # --- Debug Tracking Variables ---
    total_generated_nodes = 1  
    max_depth_reached = 0
    ttfs_sec: Optional[float] = None

    # Anytime fallback: deepest collision-free node seen so far. If the budget
    # expires before any terminal node exists, the search returns this node's
    # path extended at constant velocity instead of failing hard.
    best_partial_node: Optional[StateNode] = None

    if is_in_goal_region(request.start_state, request.goal_region, cfg.vehicle):
        return PlanResult(
            success=False,
            goal_region_reached=None,
            trajectory=None,
            cost=None,
            status_message='Vehicle is already in the goal region at function call.'
        )

    root = StateNode(
        id=id_provider.get_id(),
        state_stamped=request.start_state,
        node_cost=0.0,
        heuristic_cost=0.0,
        path_cost=0.0,
        total_cost=0.0,
        detailed_costs=None,
        goal_region_reached=False,
        node_depth=0,
        parent=None
    )

    # Discretized state deduplication (standard Hybrid A* practice): states
    # falling into the same (x, y, yaw, speed) cell are transpositions - only
    # the cheapest representative is kept in the open set. Without this the
    # exponentially wide primitive fan is stored and re-expanded verbatim.
    _XY_BIN = 1.0
    _YAW_BIN = 0.09
    _V_BIN = 1.0
    visited_cells: dict[tuple, StateNode] = {}
    def _cell_key(stamped_state) -> tuple:
        st = stamped_state.state
        return (
            int(round(st.pos.x / _XY_BIN)),
            int(round(st.pos.y / _XY_BIN)),
            int(round(st.yaw / _YAW_BIN)),
            int(round(get_signed_magnitude(st.velocity, st.yaw) / _V_BIN)),
        )
    visited_cells[_cell_key(request.start_state)] = root

    heapq.heappush(open_nodes_pq, (root.total_cost, root.id, root))

    while True:
        loop_start_time = time.perf_counter()

        if not open_nodes_pq:
            break

        prev_node = heapq.heappop(open_nodes_pq)[2]
        prev_stamped_state = prev_node.state_stamped

        raw_velocity = get_signed_magnitude(
            prev_stamped_state.state.velocity, prev_stamped_state.state.yaw
        )
        # Allow the search to hypothesize forward motion from (near-)standstill;
        # reverse driving keeps its signed speed untouched.
        expansion_speed = (
            max(raw_velocity, _MIN_PLANNING_SPEED) if raw_velocity >= 0.0 else raw_velocity
        )

        motion_primitives = get_motion_primitives(
            velocity=expansion_speed,
            steering_angle=prev_stamped_state.state.steering_angle,
            veh_cfg=cfg.vehicle,
            velocity_limit=request.target_speed,
            acceleration=get_signed_magnitude(prev_stamped_state.state.acceleration, prev_stamped_state.state.yaw),
            mp_cfg=cfg.motion_primitives,
            internal_dt=cfg.planner.dt_sim
        )
        
        for mp in motion_primitives:
            total_generated_nodes += 1
            
            curr_state = kinematic_bicycle(
                stamped_state=prev_stamped_state, 
                control=EgoInput(mp.steering_angle, mp.acceleration), 
                dt=mp.dt, 
                vehicle_params=cfg.vehicle
            )
            
            path_cost = prev_node.path_cost + prev_node.node_cost

            node_cost, detailed_costs = calculate_node_cost(
                prev_state=prev_stamped_state, curr_state=curr_state,
                request=request, cost_cfg=cfg.cost, veh_cfg=cfg.vehicle,
                opposite_lane_weight_scale=opposite_lane_weight_scale
            )
            
            heuristic_cost = calculate_heuristic_cost(
                state=curr_state, request=request, veh_cfg=cfg.vehicle, mp_cfg=cfg.motion_primitives
            )
            
            total_cost, detailed_costs = calculate_total_cost(
                path_cost=path_cost, node_cost=node_cost, heuristic_cost=heuristic_cost, 
                detailed_costs=detailed_costs, cost_cfg=cfg.cost
            )

            goal_region_reached = is_in_goal_region(curr_state, request.goal_region, cfg.vehicle)
            
           

            node_depth = prev_node.node_depth + 1
            max_depth_reached = max(max_depth_reached, node_depth)

            new_node = StateNode(
                id=id_provider.get_id(), state_stamped=curr_state,
                node_cost=node_cost, heuristic_cost=heuristic_cost, path_cost=path_cost,
                total_cost=total_cost, detailed_costs=detailed_costs,
                goal_region_reached=goal_region_reached, node_depth=node_depth,
                parent=prev_node, motion_primitive=mp
            )
            
            _cell = _cell_key(curr_state)
            _seen = visited_cells.get(_cell)
            if _seen is not None and _seen.total_cost <= new_node.total_cost:
                continue
            visited_cells[_cell] = new_node

            if not math.isinf(node_cost) and (
                best_partial_node is None
                or new_node.node_depth > best_partial_node.node_depth
                or (new_node.node_depth == best_partial_node.node_depth
                    and new_node.total_cost < best_partial_node.total_cost)
            ):
                best_partial_node = new_node

            if not new_node.goal_region_reached and new_node.node_depth < max_depth:
                sort_cost = new_node.total_cost - progress_bonus * new_node.node_depth
                heapq.heappush(open_nodes_pq, (sort_cost, new_node.id, new_node))
            else:
                heapq.heappush(end_nodes_pq, new_node)
                 # Record Time to First Solution (TTFS)
                if ttfs_sec is None:
                    ttfs_sec = time.perf_counter() - call_time
                    # Any terminal node (goal reached or full horizon) yields a
                    # usable trajectory; stopping here keeps plan cycles short
                    # and deterministic instead of optimizing until the budget
                    # runs out.
                    terminals_found += 1
                    if exit_on_first and terminals_found >= min_terminals:
                        solution_found = True

        loop_stop_time = time.perf_counter()
        loop_duration = loop_stop_time - loop_start_time

        if solution_found and exit_on_first:
            break

        time_elapsed = loop_stop_time - call_time
        time_budget_sec = cfg.planner.max_compute_time / 1000.0
        extract_time_buffer_sec = cfg.planner.extract_path_time / 1000.0
        
        if (time_elapsed + loop_duration + extract_time_buffer_sec > time_budget_sec):
            break

    # --- Result Construction ---
    best_end_node = None
    if end_nodes_pq:
        best_end_node = heapq.heappop(end_nodes_pq)

    use_fallback = False
    if best_end_node is None or math.isinf(best_end_node.total_cost):
        if best_partial_node is not None:
            best_end_node = best_partial_node
            use_fallback = True

    success = best_end_node is not None and not math.isinf(best_end_node.total_cost)

    path_length_raw = 0
    if success:
        path = extract_path_stamped_states(best_end_node)
        if use_fallback:
            # Extend the partial path at constant velocity along the current
            # steering angle so the trajectory always spans the full horizon.
            ext_state = path[-1]
            while len(path) < max_depth + 1:
                ext_state = kinematic_bicycle(
                    stamped_state=ext_state,
                    control=EgoInput(ext_state.state.steering_angle, 0.0),
                    dt=cfg.planner.dt_sim,
                    vehicle_params=cfg.vehicle
                )
                path.append(ext_state)
        path_length_raw = len(path)

        rescaling_factor = int(cfg.planner.dt_output / cfg.planner.dt_sim)
        path = path[::rescaling_factor]
        trajectory = Trajectory(states=path)

        result = PlanResult(
            success=True,
            goal_region_reached=best_end_node.goal_region_reached,
            trajectory=trajectory,
            cost=best_end_node.total_cost,
            status_message=(
                "Budget exhausted - returned best partial trajectory "
                "extended to horizon" if use_fallback else None
            ),
            debug_root_node=root
        )
    else:
        status_message = f"End Node reached: {best_end_node is not None}"
        if best_end_node is not None:
            status_message += f" | Node details: {best_end_node.__repr__()}"

        result = PlanResult(
            success=False,
            goal_region_reached=None,
            trajectory=None,
            cost=None,
            status_message=status_message,
            debug_root_node=root
        )

    # --- Debug Console Output ---
    if debug:
        total_time_ms = (time.perf_counter() - call_time) * 1000.0
        ms_per_node = total_time_ms / total_generated_nodes if total_generated_nodes > 0 else 0.0
        efficiency_pct = (path_length_raw / total_generated_nodes) * 100 if total_generated_nodes > 0 else 0.0
        ttfs_str = f"{(ttfs_sec * 1000.0):.2f} ms" if ttfs_sec is not None else "N/A (Goal not reached)"

        print("\n" + "="*50)
        print("PLANNER DEBUG STATISTICS")
        print("="*50)
        print(f"Time Budget       : {cfg.planner.max_compute_time:.2f} ms")
        print(f"Compute Time      : {total_time_ms:.2f} ms")
        print(f"TTFS              : {ttfs_str}")
        print(f"Speed             : {ms_per_node:.4f} ms / node")
        print(f"Total Nodes Gen.  : {total_generated_nodes}")
        print(f"Search Efficiency : {efficiency_pct:.2f} %")
        print(f"Max Depth Reached : {max_depth_reached} / {max_depth}")
        print(f"Success           : {success}")
        print("="*50 + "\n")

    return result          

def extract_path_stamped_states(end_node: StateNode) -> List[EgoStateStamped]:
    node = end_node
    path:List[EgoStateStamped] = []

    while node is not None:
        path.append(node.state_stamped)
        node = node.parent

    path.reverse()
    return path




def is_in_goal_region(
    state_stamped: EgoStateStamped, 
    goal_region: GoalRegion, 
    veh_cfg: DictConfig
) -> bool:
    """
    Checks if the ego vehicle's bounding box intersects with the goal region's bounding box.

    Args:
        state_stamped (EgoStateStamped): The current state of the vehicle (position at rear axle).
        goal_region (GoalRegion): The target area defined by center, dimensions, and yaw.
        veh_cfg (DictConfig): The vehicle configuration containing dimensions (length, width, rear_to_wheel).

    Returns:
        bool: True if the vehicle's bounding box overlaps with the goal region, False otherwise.
    """
    
    # --- 1. Construct Vehicle Polygon ---
    ego_x = state_stamped.state.pos.x
    ego_y = state_stamped.state.pos.y
    ego_yaw = state_stamped.state.yaw
    
    # Calculate distance from rear axle to the geometric center of the vehicle
    axle_to_center = (veh_cfg.length / 2.0) - veh_cfg.rear_to_wheel
    
    # Shift to geometric center
    ego_center_x = ego_x + axle_to_center * math.cos(ego_yaw)
    ego_center_y = ego_y + axle_to_center * math.sin(ego_yaw)
    
    # Create the vehicle bounding box polygon
    ego_polygon = _create_oriented_bounding_box(
        center_x=ego_center_x, 
        center_y=ego_center_y, 
        width=veh_cfg.width, 
        length=veh_cfg.length, 
        yaw=ego_yaw
    )
    
    # --- 2. Construct Goal Region Polygon ---
    
    goal_polygon = _create_oriented_bounding_box(
        center_x=goal_region.center.x, 
        center_y=goal_region.center.y, 
        width=goal_region.width, 
        length=goal_region.length, 
        yaw=goal_region.yaw
    )
    
    # --- 3. Check for Intersection ---
    return ego_polygon.intersects(goal_polygon)


def _create_oriented_bounding_box(
    center_x: float, 
    center_y: float, 
    width: float, 
    length: float, 
    yaw: float
) -> Polygon:
    """
    Helper function to create a shapely Polygon for an oriented bounding box.
    """
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    
    # Half dimensions
    hw = width / 2.0
    hl = length / 2.0
    
    # The 4 corners relative to the center (unrotated)
    corners_local = [
        (hl, hw),   # Front Left
        (hl, -hw),  # Front Right
        (-hl, -hw), # Rear Right
        (-hl, hw)   # Rear Left
    ]
    
    # Rotate and translate corners to global coordinates
    corners_global = []
    for lx, ly in corners_local:
        gx = center_x + (lx * cos_y - ly * sin_y)
        gy = center_y + (lx * sin_y + ly * cos_y)
        corners_global.append((gx, gy))
        
    return Polygon(corners_global)