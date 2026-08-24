"""Request-preparation policy: rolling-goal placement and lane anchoring."""
from math import atan2, cos, hypot, sin
from typing import List, Tuple

from core.geometry import get_signed_magnitude
from core.road_queries import lane_representative_yaw
from core.types import GoalRegion, Lane, Vector2D
from core.types.vehicle import EgoStateStamped


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


