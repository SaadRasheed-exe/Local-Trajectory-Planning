import heapq
from math import pi, cos, sin, hypot, atan2
from core.geometry import (
    MovingObject,
    get_bbox_corners,
    get_signed_magnitude,
    get_vector,
    get_x_y_yaw_from_state,
)
from typing import Tuple, List, Optional, overload, Sequence
from core.types.agents import DynamicObject, DynamicObjectStamped
from core.types.geometry import Vector2D
from core.types.perception import PredictedEnvironment
from core.types.road import Environment, Lane
from core.types.vehicle import EgoState, EgoStateStamped
from shapely.geometry import LineString, Polygon


def get_colliding_object_ids(
    ego_state: EgoStateStamped,
    environment: Environment,
    ego_length: float,
    ego_width: float,
    ego_rear_to_wheel: float,
) -> List[int]:
    """
    Return the ids of all objects whose bounding box currently intersects the
    ego vehicle's bounding box.

    Reference-point convention matches _ego_object_distance: the ego state
    position is at the rear axle (shifted to the geometric center here), while
    object positions are their geometric centers.
    """
    ego_x, ego_y, ego_yaw = get_x_y_yaw_from_state(ego_state)
    ego_x_center = ego_x + (ego_length / 2.0 - ego_rear_to_wheel) * cos(ego_yaw)
    ego_y_center = ego_y + (ego_length / 2.0 - ego_rear_to_wheel) * sin(ego_yaw)

    ego_corners = get_bbox_corners(ego_x_center, ego_y_center, ego_yaw, ego_length, ego_width)
    ego_polygon = Polygon([(corner.x, corner.y) for corner in ego_corners])

    colliding_ids: List[int] = []
    for obj in environment.objects:
        obj_x, obj_y, obj_yaw = get_x_y_yaw_from_state(obj)
        obj_corners = get_bbox_corners(obj_x, obj_y, obj_yaw, obj.state.length, obj.state.width)
        obj_polygon = Polygon([(corner.x, corner.y) for corner in obj_corners])
        if ego_polygon.intersects(obj_polygon):
            colliding_ids.append(obj.state.id)
    return colliding_ids


def build_lane_polygons(lanes: List[Lane]) -> List[Polygon]:
    """
    Build one drivable-area polygon per lane by buffering its centerline by half
    the lane width.

    Lanes are static for the duration of a run, so callers should build these
    once and reuse the result on every simulation step.
    """
    polygons: List[Polygon] = []
    for lane in lanes:
        if len(lane.centerline) < 2:
            continue
        centerline = LineString([(p.x, p.y) for p, _yaw in lane.centerline])
        polygons.append(centerline.buffer(lane.width / 2.0))
    return polygons


def has_exited_lanes(
    ego_state: EgoStateStamped,
    lane_polygons: List[Polygon],
    ego_length: float,
    ego_width: float,
    ego_rear_to_wheel: float,
) -> bool:
    """
    True once the ego bounding box has zero overlap with every lane polygon,
    i.e. the vehicle is completely off the drivable area (union of all lanes).

    Reference-point convention matches get_colliding_object_ids: the ego state
    position is at the rear axle (shifted to the geometric center here).
    Boundary contact still counts as being on the road.
    """
    if not lane_polygons:
        return False

    ego_x, ego_y, ego_yaw = get_x_y_yaw_from_state(ego_state)
    ego_x_center = ego_x + (ego_length / 2.0 - ego_rear_to_wheel) * cos(ego_yaw)
    ego_y_center = ego_y + (ego_length / 2.0 - ego_rear_to_wheel) * sin(ego_yaw)

    ego_corners = get_bbox_corners(ego_x_center, ego_y_center, ego_yaw, ego_length, ego_width)
    ego_polygon = Polygon([(corner.x, corner.y) for corner in ego_corners])

    return not any(ego_polygon.intersects(lane_polygon) for lane_polygon in lane_polygons)


def _ego_object_distance(
    ego_state: EgoStateStamped,
    obj: DynamicObjectStamped,
    ego_length: float,
    ego_width: float,
    ego_rear_to_wheel: float,
    exact_calc_threshold: float
) -> float:
    """
    Calculate the precise distance from the ego vehicle to a dynamic object.
    
    Uses a highly optimized two-step approach: 
    1. A fast circular bounding box approximation to quickly reject objects far away.
    2. Precise Shapely polygon math only for objects closer than `exact_calc_threshold`.
    """
    ego_x, ego_y, ego_yaw = get_x_y_yaw_from_state(ego_state)
    
    # Shift the reference point from the rear axle to the geometric center of the vehicle
    ego_x_center = ego_x + (ego_length / 2.0 - ego_rear_to_wheel) * cos(ego_yaw)
    ego_y_center = ego_y + (ego_length / 2.0 - ego_rear_to_wheel) * sin(ego_yaw)
    
    obj_x_center, obj_y_center, obj_yaw = get_x_y_yaw_from_state(obj)

    # 1. Fast Path: Circular Bounding Box Approximation
    # Calculate the distance between the geometric centers
    center_dist = hypot(ego_x_center - obj_x_center, ego_y_center - obj_y_center)
    
    # Calculate the circumscribed circle radii for both vehicles
    ego_radius = hypot(ego_length, ego_width) / 2.0
    obj_radius = hypot(obj.state.length, obj.state.width) / 2.0
    
    # Calculate the theoretical minimum distance assuming worst-case rotation
    min_possible_dist = center_dist - ego_radius - obj_radius

    # If the objects are safely outside the critical zone, skip Shapely completely to save CPU cycles
    if min_possible_dist > exact_calc_threshold:
        return min_possible_dist

    # 2. Slow Path: Exact Polygon Distance for close-range accuracy
    ego_corners = get_bbox_corners(ego_x_center, ego_y_center, ego_yaw, ego_length, ego_width)
    obj_corners = get_bbox_corners(obj_x_center, obj_y_center, obj_yaw, obj.state.length, obj.state.width)

    # Convert to Shapely polygons to calculate the exact shortest distance between the shapes
    ego_polygon = Polygon([(corner.x, corner.y) for corner in ego_corners])
    obj_polygon = Polygon([(corner.x, corner.y) for corner in obj_corners])

    return ego_polygon.distance(obj_polygon)


@overload
def _create_interpolated_object(
    base: EgoStateStamped,
    new_timestamp: int,
    new_pos: Vector2D,
    new_yaw: float,
    new_velocity: float
) -> EgoStateStamped:
    ...


@overload
def _create_interpolated_object(
    base: DynamicObjectStamped,
    new_timestamp: int,
    new_pos: Vector2D,
    new_yaw: float,
    new_velocity: float
) -> DynamicObjectStamped:
    ...


def _create_interpolated_object(
    base: MovingObject,
    new_timestamp: int,
    new_pos: Vector2D,
    new_yaw: float,
    new_velocity: float
) -> MovingObject:
    """Helper factory to instantiate the correct object type after interpolation."""
    vel_vector = get_vector(new_velocity, new_yaw)

    if isinstance(base, EgoStateStamped):
        return EgoStateStamped(
            timestamp=new_timestamp,
            state=EgoState(
                pos=new_pos,
                yaw=new_yaw,
                velocity=vel_vector,
                acceleration=Vector2D(0.0, 0.0),
                steering_angle=0.0,
            )
        )

    elif isinstance(base, DynamicObjectStamped):
        return DynamicObjectStamped(
            timestamp=new_timestamp,
            state=DynamicObject(
                id=base.state.id,
                obj_class=base.state.obj_class,
                pos=new_pos,
                yaw=new_yaw,
                velocity=vel_vector,
                acceleration=Vector2D(0.0, 0.0),
                width=base.state.width,
                length=base.state.length
            )
        )

    raise TypeError("Unsupported object type")


@overload
def _interpolate_states(
    start_state: EgoStateStamped,
    end_state: EgoStateStamped,
    new_resolution_ms: int
) -> Sequence[EgoStateStamped]:
    ...


@overload
def _interpolate_states(
    start_state: DynamicObjectStamped,
    end_state: DynamicObjectStamped,
    new_resolution_ms: int
) -> Sequence[DynamicObjectStamped]:
    ...


def _interpolate_states(
    start_state: MovingObject,
    end_state: MovingObject,
    new_resolution_ms: int
) -> Sequence[MovingObject]:
    """
    Interpolate the vehicle state linearly from the start state to the end state.
    This increases the temporal resolution to prevent vehicles from "teleporting" through each other.
    """
    orig_resolution_ms = end_state.timestamp - start_state.timestamp
    if orig_resolution_ms <= new_resolution_ms:
        return [start_state, end_state]
    
    # Calculate step differences for spatial and kinematic properties
    dx = (end_state.state.pos.x - start_state.state.pos.x)
    dy = (end_state.state.pos.y - start_state.state.pos.y)
    
    # Use signed magnitudes to properly handle reversing vehicles during interpolation
    dvel = (get_signed_magnitude(end_state.state.velocity, end_state.state.yaw) - 
            get_signed_magnitude(start_state.state.velocity, start_state.state.yaw))
    
    # Calculate the shortest angular difference (handles the pi to -pi wraparound)
    dyaw = (end_state.state.yaw - start_state.state.yaw + pi) % (2 * pi) - pi
    
    start_vel_magnitude = get_signed_magnitude(start_state.state.velocity, start_state.state.yaw)

    interpolated_states = [start_state]
    t = 0

    while t < orig_resolution_ms:
        t += new_resolution_ms
        
        # Linearly scale parameters based on the passed time fraction
        interp_x = start_state.state.pos.x + (t * dx / orig_resolution_ms)
        interp_y = start_state.state.pos.y + (t * dy / orig_resolution_ms)
        interp_yaw = start_state.state.yaw + (t * dyaw / orig_resolution_ms)
        interp_vel = start_vel_magnitude + (t * dvel / orig_resolution_ms)

        interpolated_state = _create_interpolated_object(
            base=start_state,
            new_timestamp=start_state.timestamp + t,
            new_pos=Vector2D(x=interp_x, y=interp_y),
            new_yaw=interp_yaw,
            new_velocity=interp_vel
        )
        interpolated_states.append(interpolated_state)

    interpolated_states.append(end_state)
    return interpolated_states


def get_distance_to_objects(
    current_ego: EgoStateStamped,
    previous_ego: EgoStateStamped,
    predicted_env: PredictedEnvironment,
    ego_length: float,
    ego_width: float,
    ego_rear_to_wheel: float,
    resolution_ms: int,
    calc_exact_distance: float
) -> Tuple[Optional[Sequence[Tuple[int, float]]], bool]:
    """
    Checks the continuous interval between two planning steps for collisions.
    Calculates and returns the aggregated minimum distances to all objects.
    """

    # Coarse rejection: if every predicted object state is provably beyond the
    # influence range of both interval endpoints, no distance or collision is
    # possible and the expensive sweep can be skipped entirely.
    ego_radius = (ego_length**2 + ego_width**2) ** 0.5 / 2.0
    influence_range = calc_exact_distance + 2.0 * ego_radius
    endpoints = (
        (previous_ego.state.pos.x, previous_ego.state.pos.y),
        (current_ego.state.pos.x, current_ego.state.pos.y),
    )
    any_object_in_range = False
    for obj_states_all in predicted_env.objects.values():
        for obj_state in obj_states_all:
            for ex, ey in endpoints:
                if (
                    abs(obj_state.state.pos.x - ex) <= influence_range
                    and abs(obj_state.state.pos.y - ey) <= influence_range
                ):
                    any_object_in_range = True
                    break
            if any_object_in_range:
                break
        if any_object_in_range:
            break
    if not any_object_in_range:
        return [], False

    # Sub-divide the planning step into smaller intervals to detect fast-moving intersections
    interpolated_ego_states = _interpolate_states(previous_ego, current_ego, resolution_ms)

    # Filter predicted objects to only include those relevant for this specific time interval
    relevant_objects = []
    for obj_id, obj_states in predicted_env.objects.items():
        relevant_objects.append([])
        for obj_state in obj_states:
            if previous_ego.timestamp <= obj_state.timestamp <= current_ego.timestamp:
                # Use a priority queue (heapq) to easily sync object timestamps with the ego timeline
                heapq.heappush(relevant_objects[-1], (obj_state.timestamp, obj_state))

    distance_results = []
    for obj_states in relevant_objects:
        if not obj_states:
            continue
        
        min_distance = float('inf')
        for ego_state in interpolated_ego_states:
            # Sync timelines: discard object states that are chronologically behind the current ego state
            while obj_states and obj_states[0][0] < ego_state.timestamp:
                heapq.heappop(obj_states)
            if not obj_states:
                break
            
            # Retrieve the most synchronous object state
            obj_state = obj_states[0][1]
            distance = _ego_object_distance(ego_state, obj_state, ego_length, ego_width, ego_rear_to_wheel, calc_exact_distance)
            min_distance = min(min_distance, distance)

            # Hard collision detected, terminate evaluation early
            if distance < 0.1:  
                return None, True
        
        distance_results.append((obj_state.state.id, min_distance))
    
    return distance_results, False


def _get_lane_offset(
    ego_center_x: float, 
    ego_center_y: float, 
    ego_yaw: float, 
    lane: Lane, 
    closest_idx: int
) -> Tuple[float, float]:
    """Calculate the perpendicular offset of the ego vehicle's geometric center from the lane centerline."""
    min_distance = float('inf')
    yaw_offset = 0.0

    # Restrict the segment search to the immediate neighborhood of the closest waypoint
    start_idx = max(0, closest_idx - 1)
    end_idx = min(len(lane.centerline) - 1, closest_idx + 1)

    for i in range(start_idx, end_idx):
        p1, yaw1 = lane.centerline[i]
        p2, yaw2 = lane.centerline[i + 1]

        # Vector defining the lane segment
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        
        sq_len = dx * dx + dy * dy
        if sq_len == 0.0:
            continue
        
        # Calculate the normalized perpendicular vector (normal vector)
        inv_len = 1.0 / (sq_len ** 0.5)
        nx = -dy * inv_len
        ny = dx * inv_len

        # Project the ego point onto the normal vector to get the lateral distance
        distance_to_line = abs((ego_center_x - p1.x) * nx + (ego_center_y - p1.y) * ny)
        
        if distance_to_line < min_distance:
            min_distance = distance_to_line
            # Calculate the angular deviation relative to the average segment direction
            yaw_offset = (ego_yaw - (yaw1 + yaw2) / 2.0 + pi) % (2 * pi) - pi

    return min_distance, yaw_offset


def get_ego_lane_info(
    ego_state: EgoState,
    ego_length: float,
    ego_width: float,
    ego_rear_to_wheel: float,
    lanes: List[Lane],
    reference_yaw: float = None
) -> Tuple[int, float, float, float, bool, float]:
    """
    Evaluates the ego vehicle's position relative to the road topology.

    When reference_yaw is given, the primary lane is selected only among
    travel-direction lanes whose representative direction agrees with it
    (falling back to the aligned lane with the highest footprint overlap).
    This supplies the lateral cost gradient that pulls the ego back into its
    original lane once an overtake is complete; without it the nearest-
    overlap rule keeps referencing the lane the ego is currently inside and
    no return pressure exists.

    Returns:
    lane_id: int (ID of the best matching lane)
    distance_to_lane_center: float [m]
    yaw_offset_to_lane_direction: float [rad]
    lane_occlusion: float Ratio [0.0 - 1.0] (1.0 = fully inside the drivable lane boundaries)
    opposite_lane: bool (True if driving against the lane's intended direction)
    speed_limit: float [m/s]
    """
    occlusion_sum = 0.0
    best_lane_id = -1
    best_dist = float('inf')
    best_yaw_offset = 0.0
    is_opposite = False
    best_speed_limit = 0.0
    # Highest-overlap aligned lane, used as fallback when none exceeds the
    # 0.5 primary threshold (e.g. ego still mostly inside the other lane).
    fb_ratio = -1.0
    fb_dist = 0.0
    fb_yaw_offset = 0.0
    fb_opposite = False
    fb_speed_limit = 0.0
    fb_lane_id = -1

    # 1. Forcibly shift coordinates to the geometric center for accurate boundary checks!
    # Relying on the rear axle would falsely flag front-bumper lane departures.
    ego_yaw = ego_state.yaw
    ego_center_offset = ego_length / 2.0 - ego_rear_to_wheel
    ego_center_x = ego_state.pos.x + ego_center_offset * cos(ego_yaw)
    ego_center_y = ego_state.pos.y + ego_center_offset * sin(ego_yaw)

    # Maximum possible geometric radius of the vehicle (distance from center to a corner)
    max_ego_radius = (ego_length**2 + ego_width**2)**0.5 / 2.0

    for lane in lanes:
        if not lane.centerline:
            continue

        # Travel-direction filter for return mode: skip lanes whose
        # representative direction opposes the requested reference yaw.
        if reference_yaw is not None:
            p0 = lane.centerline[0][0]
            p1 = lane.centerline[min(1, len(lane.centerline) - 1)][0]
            if cos(atan2(p1.y - p0.y, p1.x - p0.x) - reference_yaw) <= 0.25:
                continue
        # Quick closest-point search for spatial rejection.
        # Coarse-to-fine: stride scan first, then refine around the best hit
        # (centerlines are densely sampled; a full scan per node is wasteful).
        centerline_len = len(lane.centerline)
        stride = max(1, centerline_len // 64)
        closest_sq_dist = float('inf')
        closest_idx = -1

        for i in range(0, centerline_len, stride):
            p = lane.centerline[i][0]
            dx = p.x - ego_center_x
            dy = p.y - ego_center_y
            sq_dist = dx * dx + dy * dy
            if sq_dist < closest_sq_dist:
                closest_sq_dist = sq_dist
                closest_idx = i

        refine_start = max(0, closest_idx - stride)
        refine_end = min(centerline_len, closest_idx + stride + 1)
        for i in range(refine_start, refine_end):
            p = lane.centerline[i][0]
            dx = p.x - ego_center_x
            dy = p.y - ego_center_y
            sq_dist = dx * dx + dy * dy
            if sq_dist < closest_sq_dist:
                closest_sq_dist = sq_dist
                closest_idx = i

        # Rejection radius based on the geometric center
        # If the vehicle's bounding circle cannot possibly touch the lane boundary, skip the lane.
        # Skipped when a reference_yaw filter is active: the ego may be far outside the
        # reference lane mid-return, and its offset must be measured anyway to supply
        # the pull-back gradient (inf would poison every downstream cost).
        rejection_threshold = (lane.width / 2.0) + max_ego_radius + 2.0
        if reference_yaw is None and closest_sq_dist > rejection_threshold**2:
            continue

        # Calculate the exact lateral offset using the CENTER coordinates
        dist_to_center, yaw_offset = _get_lane_offset(ego_center_x, ego_center_y, ego_yaw, lane, closest_idx)

        # Second early exit: If the center is too far from the boundaries
        if reference_yaw is None and dist_to_center > (lane.width / 2.0) + max_ego_radius:
            continue

        # Mathematical approximation of the bounding box overlap.
        # Instead of heavy polygon intersection math, we project the vehicle's width and length
        # laterally to estimate how much of its footprint spans across the lane.
        lateral_projection = abs(ego_width * cos(yaw_offset)) + abs(ego_length * sin(yaw_offset))
        half_proj = lateral_projection / 2.0
        
        left_edge_ego = dist_to_center - half_proj
        right_edge_ego = dist_to_center + half_proj
        half_lane = lane.width / 2.0
        
        # Clamp the vehicle's projected edges to the lane boundaries to find the overlapping segment
        overlap_left = max(-half_lane, min(half_lane, left_edge_ego))
        overlap_right = max(-half_lane, min(half_lane, right_edge_ego))
        overlap_width = overlap_right - overlap_left
        
        # IMPORTANT: This is now a pure ratio between 0.0 and 1.0!
        # It represents the percentage of the vehicle footprint that is currently legally on the road.
        if lateral_projection > 0:
            lane_occlusion_ratio = overlap_width / lateral_projection
        else:
            lane_occlusion_ratio = 0.0

        occlusion_sum += lane_occlusion_ratio

        # Determine the primary matching lane based on the highest occlusion ratio
        if lane_occlusion_ratio > 0.5 and best_lane_id == -1:
            best_lane_id = lane.id
            best_dist = dist_to_center
            best_yaw_offset = yaw_offset
            # If the yaw deviation is greater than 90 degrees, the vehicle is facing the wrong way
            is_opposite = (yaw_offset > pi / 2 or yaw_offset < -pi / 2)
            best_speed_limit = lane.speed_limit

        if reference_yaw is not None and lane_occlusion_ratio > fb_ratio:
            fb_ratio = lane_occlusion_ratio
            fb_lane_id = lane.id
            fb_dist = dist_to_center
            fb_yaw_offset = yaw_offset
            fb_opposite = (yaw_offset > pi / 2 or yaw_offset < -pi / 2)
            fb_speed_limit = lane.speed_limit

    if best_lane_id == -1 and fb_lane_id != -1:
        best_lane_id, best_dist, best_yaw_offset = fb_lane_id, fb_dist, fb_yaw_offset
        is_opposite, best_speed_limit = fb_opposite, fb_speed_limit

    return best_lane_id, best_dist, best_yaw_offset, occlusion_sum, is_opposite, best_speed_limit
