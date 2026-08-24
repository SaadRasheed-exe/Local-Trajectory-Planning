"""Pure lane-geometry queries over the static road model."""
from math import atan2, cos, hypot, pi, sin, sqrt
from typing import List, Tuple

from core.geometry import get_signed_magnitude
from core.types.geometry import Vector2D
from core.types.road import Lane
from core.types.vehicle import EgoState, EgoStateStamped


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

# --------------------------------------------------------------
# Return-mode gate: blocking-query semantics over road + predictions
# --------------------------------------------------------------

def lane_blocked_ahead(
    start_state: EgoStateStamped,
    predicted_env,
    veh_cfg,
    lookahead_m: float = 150.0,
    corridor_half_width_m: float = 4.0,
    speed_margin_mps: float = 0.5,
    rear_hold_m: float = 12.0,
) -> bool:
    """
    True if any predicted traffic object blocks the ego's forward corridor.

    A blocker must simultaneously:
      - lie within the longitudinal window (-rear_hold_m, lookahead_m] along
        the travel-direction lane axis. The extended rear window provides
        hysteresis: a vehicle still alongside counts as a blocker even when
        its center drifts just behind the perpendicular foot, so the
        return-mode gate cannot flicker mid-overtake,
      - sit within the lateral corridor around that axis
        (|l| <= corridor_half_width_m),
      - move slower than the ego along that axis by more than the margin
        (oncoming traffic has negative forward speed and thus qualifies).

    The reference frame is the travel-direction lane axis rather than the
    ego's instantaneous heading, so a mid-overtake steering tilt cannot
    rotate a truly-ahead vehicle out of the corridor. Evaluated once per
    planning cycle; used to gate the opposite-lane return penalty so it
    only engages once no blocker remains alongside or ahead.
    """
    st = start_state.state
    ref_yaw = _travel_direction_yaw(st, predicted_env)
    ref_x, ref_y = cos(ref_yaw), sin(ref_yaw)
    ego_speed = get_signed_magnitude(st.velocity, ref_yaw)
    threshold_speed = ego_speed - speed_margin_mps

    for obj_states in predicted_env.objects.values():
        # Judge blocking on the object's current position only. Iterating the
        # full prediction trail instead makes a passed vehicle's projected
        # tail (extending tens of meters down-road over the horizon) count as
        # "ahead in corridor" indefinitely, so the gate could never settle.
        stamped = min(
            obj_states,
            key=lambda o: abs(o.timestamp - start_state.timestamp),
        )
        obj = stamped.state
        dx = obj.pos.x - st.pos.x
        dy = obj.pos.y - st.pos.y
        s = dx * ref_x + dy * ref_y
        if not (-rear_hold_m < s <= lookahead_m):
            continue
        l = -dx * ref_y + dy * ref_x
        if abs(l) > corridor_half_width_m:
            continue
        obj_forward_speed = obj.velocity.x * ref_x + obj.velocity.y * ref_y
        if obj_forward_speed < threshold_speed:
            return True
    return False


def _travel_direction_yaw(ego_state: EgoState, predicted_env) -> float:
    """
    Reference yaw for longitudinal/lateral gating queries: the representative
    yaw of the first lane whose direction agrees with the ego's velocity,
    falling back to the ego pose yaw when no lane matches.
    """
    vx, vy = ego_state.velocity.x, ego_state.velocity.y
    if sqrt(vx * vx + vy * vy) < 1e-3:
        return ego_state.yaw
    vel_yaw = atan2(vy, vx)
    for lane in getattr(predicted_env, "lanes", []) or []:
        if lane.centerline and cos(lane_representative_yaw(lane) - vel_yaw) > 0.25:
            return lane_representative_yaw(lane)
    return ego_state.yaw


def lane_representative_yaw(lane: Lane) -> float:
    """Representative direction of a lane: yaw of its first centerline segment."""
    p0, _ = lane.centerline[0]
    p1, _ = lane.centerline[min(1, len(lane.centerline) - 1)]
    return atan2(p1.y - p0.y, p1.x - p0.x)


