"""Planning-request preparation and shared run-loop policies.

Everything here is plain application logic over core types plus the selected
components; it contains no runtime-mode specifics.
"""
import time
from omegaconf import DictConfig, OmegaConf

from math import atan2, cos, hypot, sin
from typing import List, Optional, Tuple

from simulation.simulate import Simulation

from components.collision.collision_queries import (
    get_colliding_object_ids,
    get_distance_to_objects,
    has_exited_lanes,
)
from components.predictors.constant_velocity.predictor import predict_environment
from core.geometry import get_signed_magnitude
from core.road_queries import get_ego_lane_info, lane_representative_yaw
from core.types import GoalRegion, Lane, Vector2D
from core.types.perception import PredictedEnvironment
from core.types.planning import PlanningRequest, Trajectory
from core.types.road import Environment
from core.types.vehicle import EgoStateStamped

FALLBACK_TARGET_SPEED = 10.0   # [m/s], used when ego is not matched to any lane
FALLBACK_GOAL_LENGTH = 5.0     # [m]
FALLBACK_GOAL_WIDTH = 5.0      # [m]

# The run stops once the ego is this close to its lane's last centerline point.
LANE_END_STOP_MARGIN_M = 5.0


# ---------------------------------------------------------------------------
# Rolling-goal placement
# ---------------------------------------------------------------------------

def get_goal_region(
    curr_ego_state: EgoStateStamped,
    lanes: List[Lane],
    horizon: int,
    length: float,
    width: float,
    target_speed: float = 5.0,
) -> GoalRegion:
    """
    Get the goal region which is horizon seconds ahead in the same lane.

    Lanes whose direction opposes the ego's current heading are ignored when
    placing the rolling goal, otherwise the anchor flips behind the vehicle
    during lateral maneuvers on two-way roads.
    """
    ego_vel = curr_ego_state.state.velocity

    # Direction filter: compare each lane's representative yaw against ego heading.
    aligned = [
        lane for lane in lanes
        if lane.centerline
        and cos(lane_representative_yaw(lane) - atan2(ego_vel.y, ego_vel.x)) > 0.25
    ]
    anchor_lanes = aligned if aligned else lanes

    min_dist = float('inf')
    nearest_lane_yaw, nearest_lane_center = 0.0, Vector2D(x=0.0, y=0.0)
    found = False
    for lane in anchor_lanes:
        for point, yaw in lane.centerline:
            dist = hypot(curr_ego_state.state.pos.x - point.x, curr_ego_state.state.pos.y - point.y)
            if dist < min_dist:
                min_dist = dist
                nearest_lane_yaw = yaw
                nearest_lane_center = point
                found = True
    if not found:
        nearest_lane_yaw, nearest_lane_center = get_nearest_lane_center(curr_ego_state, lanes)

    vel = curr_ego_state.state.velocity
    vel_magnitude = hypot(vel.x, vel.y)

    # Ensure a minimum lookahead distance even when the vehicle is stopped
    planning_speed = max(vel_magnitude, target_speed)
    distance_ahead = planning_speed * (horizon / 1000.0)

    goal_center_x = nearest_lane_center.x + distance_ahead * cos(nearest_lane_yaw)
    goal_center_y = nearest_lane_center.y + distance_ahead * sin(nearest_lane_yaw)

    return GoalRegion(
        center=Vector2D(x=goal_center_x, y=goal_center_y),
        length=length,
        width=width,
        yaw=nearest_lane_yaw
    )


def get_nearest_lane_end_distance(ego_state: EgoStateStamped, lanes: List[Lane]) -> float:
    """
    Distance [m] from the ego position to the end of its nearest lane.

    The nearest lane is the one containing the globally closest centerline
    point; 'end' is that lane's final centerline point.
    """
    nearest_lane = None
    min_dist = float('inf')
    for lane in lanes:
        for point, _yaw in lane.centerline:
            dist = hypot(ego_state.state.pos.x - point.x, ego_state.state.pos.y - point.y)
            if dist < min_dist:
                min_dist = dist
                nearest_lane = lane

    if nearest_lane is None or not nearest_lane.centerline:
        return float('inf')

    end_point = nearest_lane.centerline[-1][0]
    return hypot(
        ego_state.state.pos.x - end_point.x,
        ego_state.state.pos.y - end_point.y,
    )


def get_nearest_lane_center(ego_state: EgoStateStamped, lanes: List[Lane]) -> Tuple[float, Vector2D]:
    """
    Find the nearest lane center point to the ego vehicle.
    
    Args:
        ego_state (EgoStateStamped): The current state of the ego vehicle.
        lanes (List[Lane]): A list of available lanes.

    Returns:
        float: The yaw of the nearest lane center point.
        Vector2D: The position of the nearest lane center point.
    """
    
    min_dist = float('inf')
    nearest_point = Vector2D(x=0.0, y=0.0)
    nearest_lane_yaw = 0.0

    for lane in lanes:
        for point, yaw in lane.centerline:
            dist = hypot(ego_state.state.pos.x - point.x, ego_state.state.pos.y - point.y)
            if dist < min_dist:
                min_dist = dist
                nearest_point = point
                nearest_lane_yaw = yaw

    return nearest_lane_yaw, nearest_point


# ---------------------------------------------------------------------------
# Planning-request assembly
# ---------------------------------------------------------------------------

def _build_predicted_environment(curr_env, cfg: DictConfig) -> PredictedEnvironment:
    try:
        return predict_environment(
            environment=curr_env,
            prediction_horizon=cfg.planner.horizon,
            dt=cfg.planner.dt_sim,
        )
    except ValueError:
        # Scenarios without dynamic objects: nothing to predict.
        return PredictedEnvironment(
            objects={},
            lanes=curr_env.lanes,
            dt=cfg.planner.dt_sim,
            horizon=cfg.planner.horizon,
        )


def _get_target_speed(ego_state: EgoStateStamped, curr_env, cfg: DictConfig) -> float:
    _, _, _, _, _, lane_speed_limit = get_ego_lane_info(
        ego_state=ego_state.state,
        ego_length=cfg.vehicle.length,
        ego_width=cfg.vehicle.width,
        ego_rear_to_wheel=cfg.vehicle.rear_to_wheel,
        lanes=curr_env.lanes,
    )
    return lane_speed_limit if lane_speed_limit else FALLBACK_TARGET_SPEED


def _build_goal_region(
    lanes,
    ego_state: EgoStateStamped,
    cfg: DictConfig,
    target_speed: float,
):
    """
    Anchor the rolling goal region relative to the ego pose.

    This is a local planner: the goal is always 'horizon seconds ahead on the
    current lane' (see get_goal_region), never a fixed map
    coordinate. Called once per planning loop for the actual planning target
    and once per animation frame for the live display.
    """
    goal_cfg = OmegaConf.select(cfg, "scenario.goal")
    return get_goal_region(
        curr_ego_state=ego_state,
        lanes=lanes,
        horizon=int(goal_cfg.horizon) if goal_cfg is not None else cfg.planner.horizon,
        length=float(goal_cfg.length) if goal_cfg is not None else FALLBACK_GOAL_LENGTH,
        width=float(goal_cfg.width) if goal_cfg is not None else FALLBACK_GOAL_WIDTH,
        target_speed=target_speed,
    )


def _build_planning_pipeline(curr_env, ego_state: EgoStateStamped, cfg: DictConfig):
    """Build predictions and assemble a PlanningRequest from a world snapshot."""
    pred_env = _build_predicted_environment(curr_env, cfg)

    target_speed = _get_target_speed(ego_state, curr_env, cfg)
    goal_region = _build_goal_region(curr_env.lanes, ego_state, cfg, target_speed)

    request = PlanningRequest(
        start_state=ego_state,
        goal_region=goal_region,
        target_speed=target_speed,
        environment=pred_env,
    )
    return pred_env, goal_region, request


def _prepare_planning(sim: Simulation, cfg: DictConfig):
    """Fetch world state, build predictions and assemble a PlanningRequest."""
    curr_env = sim.get_environment()
    ego_state = sim.get_ego_state()
    pred_env, goal_region, request = _build_planning_pipeline(curr_env, ego_state, cfg)
    return curr_env, ego_state, pred_env, goal_region, request


# ---------------------------------------------------------------------------
# Control fallback + stop conditions
# ---------------------------------------------------------------------------

def _brake_command(cfg: DictConfig) -> Tuple[float, float]:
    # vehicle.max_deceleration is negative by convention.
    return float(cfg.vehicle.max_deceleration), 0.0


def _compute_control_command(
    controller,
    ego_state: EgoStateStamped,
    trajectory: Optional[Trajectory],
    cfg: DictConfig,
) -> Tuple[float, float]:
    """Returns (acceleration, steer_rate); falls back to emergency braking."""
    if trajectory is None or ego_state.timestamp >= trajectory.states[-1].timestamp:
        print("No usable trajectory (none available or plan exhausted). Emergency braking.")
        return _brake_command(cfg)
    try:
        command = controller.compute_control(ego_state, trajectory)
        return command.acceleration, command.steer_rate
    except Exception as exc:
        print(f"Controller failed ({exc}). Emergency braking.")
        return _brake_command(cfg)


def _pace_real_time(loop_start: float, step_ms: int) -> None:
    remaining_s = step_ms / 1000.0 - (time.perf_counter() - loop_start)
    if remaining_s > 0.0:
        time.sleep(remaining_s)
    else:
        print(f"Warning: loop overran its {step_ms} ms budget by {-remaining_s * 1000.0:.1f} ms")


def _stop_reason_for_state(
    ego_state: EgoStateStamped,
    environment: Environment,
    lanes,
    lane_polygons,
    cfg: DictConfig,
) -> Optional[str]:
    """
    Returns a shutdown reason if the run must stop now, else None.

    Stop conditions: the ego bounding box intersects another vehicle, the ego
    bounding box is completely off the drivable area (all lanes), or the ego
    has reached the end of its lane. Evaluated once per simulation step.
    """
    colliding_ids = get_colliding_object_ids(
        ego_state=ego_state,
        environment=environment,
        ego_length=float(cfg.vehicle.length),
        ego_width=float(cfg.vehicle.width),
        ego_rear_to_wheel=float(cfg.vehicle.rear_to_wheel),
    )
    if colliding_ids:
        return f"Collision with object(s) {colliding_ids}."

    if has_exited_lanes(
        ego_state=ego_state,
        lane_polygons=lane_polygons,
        ego_length=float(cfg.vehicle.length),
        ego_width=float(cfg.vehicle.width),
        ego_rear_to_wheel=float(cfg.vehicle.rear_to_wheel),
    ):
        return "Ego fully exited the lane boundaries."

    if get_nearest_lane_end_distance(ego_state, lanes) <= LANE_END_STOP_MARGIN_M:
        return "Lane end reached."

    return None


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _log_object_distances(
    previous_ego: EgoStateStamped,
    current_ego: EgoStateStamped,
    pred_env: PredictedEnvironment,
    cfg: DictConfig,
) -> None:
    ff_cfg = cfg.cost.cost_objects_force_field
    distances, is_collision = get_distance_to_objects(
        current_ego=current_ego,
        previous_ego=previous_ego,
        predicted_env=pred_env,
        ego_length=cfg.vehicle.length,
        ego_width=cfg.vehicle.width,
        ego_rear_to_wheel=cfg.vehicle.rear_to_wheel,
        resolution_ms=int(ff_cfg.resolution_ms),
        calc_exact_distance=float(ff_cfg.d_ffl),
    )
    print(f"[debug] Object distances: {distances} | collision: {is_collision}")


def _export_search_tree(plan_result, pred_env, goal_region, cfg: DictConfig) -> None:
    # Lazy import: bokeh is only needed when debugging.
    from visualization.tree_visualizer import visualize_search_tree

    visualize_search_tree(
        plan_result=plan_result,
        pred_env=pred_env,
        goal_region=goal_region,
        cfg=cfg,
        vehicle_cfg=cfg.vehicle,
        output_filename="debug_tree.html",
    )
