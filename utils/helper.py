from math import cos, pi, sin, hypot
from typing import List, Optional, Tuple
from configparser import ConfigParser
from models.models import Vector2D, EgoStateStamped, Lane, GoalRegion
import copy
from types import SimpleNamespace
from omegaconf import OmegaConf


def global_to_ego_axis(poi_x: float, poi_y: float, ego_x: float, ego_y: float, ego_yaw: float, poi_yaw: Optional[float] = None) -> Tuple[float, float, Optional[float]]:
    """Convert a point of interest (POI) from global coordinates to the ego vehicle's local coordinate frame."""

    dx = poi_x - ego_x
    dy = poi_y - ego_y

    x_ego =  dx * cos(ego_yaw) + dy * sin(ego_yaw)
    y_ego = -dx * sin(ego_yaw) + dy * cos(ego_yaw)

    if poi_yaw is not None:
        yaw_ego = poi_yaw - ego_yaw
        yaw_ego = (yaw_ego + pi) % (2 * pi) - pi
        return x_ego, y_ego, yaw_ego

    return x_ego, y_ego, None


def get_bbox_corners(x: float, y: float, yaw: float, length: float, width: float) -> List[Vector2D]:
    """Compute the corners of a bounding box given its center position, orientation, length, and width."""

    return [
        Vector2D(
            x = x + (length / 2) * cos(yaw) - (width / 2) * sin(yaw),
            y = y + (length / 2) * sin(yaw) + (width / 2) * cos(yaw)
        ),
        Vector2D(
            x = x + (length / 2) * cos(yaw) + (width / 2) * sin(yaw),
            y = y + (length / 2) * sin(yaw) - (width / 2) * cos(yaw)
        ),
        Vector2D(
            x = x - (length / 2) * cos(yaw) + (width / 2) * sin(yaw),
            y = y - (length / 2) * sin(yaw) - (width / 2) * cos(yaw)
        ),
        Vector2D(
            x = x - (length / 2) * cos(yaw) - (width / 2) * sin(yaw),
            y = y - (length / 2) * sin(yaw) + (width / 2) * cos(yaw)
        )
    ]


def check_line_intersection(p1: Vector2D, p2: Vector2D, p3: Vector2D, p4: Vector2D) -> bool:
    """Check if the line segments p1p2 and p3p4 intersect."""

    d1 = Vector2D(x = p2.x - p1.x, y = p2.y - p1.y)
    d2 = Vector2D(x = p4.x - p3.x, y = p4.y - p3.y)

    det = d1.x * d2.y - d1.y * d2.x

    if det == 0:
        return False

    t = ((p3.x - p1.x) * d2.y - (p3.y - p1.y) * d2.x) / det
    u = ((p3.x - p1.x) * d1.y - (p3.y - p1.y) * d1.x) / det

    return (0 <= t <= 1) and (0 <= u <= 1)


def get_magnitude(vector: Vector2D) -> float:
    return hypot(vector.x, vector.y)

def get_signed_magnitude(vector: Vector2D, yaw: float) -> float:
    """
    Calculates the longitudinal length of a 2D vector relative to the vehicle's heading.
    
    Args:
        vector (Vector2D): The vector to project (e.g., velocity or acceleration).
        yaw (float): The current heading of the vehicle in radians.
        
    Returns:
        float: 
            Positive value: Vector points in the direction of travel (moving forward / accelerating).
            Negative value: Vector points opposite to the direction of travel (reversing / braking).
    """
    return vector.x * cos(yaw) + vector.y * sin(yaw)


def get_vector(magnitude: float, direction: float) -> Vector2D:
    return Vector2D(x = magnitude * cos(direction), y = magnitude * sin(direction))


def shift_rear_axle_to_cg(state_stamped: EgoStateStamped, lr: float) -> EgoStateStamped:
    """
    Shifts the vehicle's position from the center of the rear axle to the center of gravity (CG).
    
    Args:
        state_stamped (EgoStateStamped): The original state referenced at the rear axle.
        lr (float): The distance from the rear axle to the center of gravity [m].
        
    Returns:
        EgoStateStamped: A new state object with the position shifted to the CG.
    """
    # Create a deep copy to avoid modifying the original object in memory
    new_stamped = copy.deepcopy(state_stamped)
    
    yaw = new_stamped.state.yaw
    
    # Shift forward along the vehicle's longitudinal axis
    new_stamped.state.pos.x += lr * cos(yaw)
    new_stamped.state.pos.y += lr * sin(yaw)
    
    return new_stamped


def shift_cg_to_rear_axle(state_stamped: EgoStateStamped, lr: float) -> EgoStateStamped:
    """
    Shifts the vehicle's position from the center of gravity (CG) back to the center of the rear axle.
    
    Args:
        state_stamped (EgoStateStamped): The original state referenced at the CG.
        lr (float): The distance from the rear axle to the center of gravity [m].
        
    Returns:
        EgoStateStamped: A new state object with the position shifted to the rear axle.
    """
    # Create a deep copy to avoid modifying the original object in memory
    new_stamped = copy.deepcopy(state_stamped)
    
    yaw = new_stamped.state.yaw
    
    # Shift backward along the vehicle's longitudinal axis
    new_stamped.state.pos.x -= lr * cos(yaw)
    new_stamped.state.pos.y -= lr * sin(yaw)
    
    return new_stamped


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
    """
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


def convert_cfg_to_native(cfg_obj):
    """
    Konvertiert OmegaConf rekursiv in native SimpleNamespace-Objekte.
    Dies ermöglicht C-Level Zugriffsgeschwindigkeiten auf Attribute.
    """
    if hasattr(cfg_obj, "_is_dict"):
        d = OmegaConf.to_container(cfg_obj, resolve=True)
    else:
        d = cfg_obj

    if isinstance(d, dict):
        return SimpleNamespace(**{k: convert_cfg_to_native(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [convert_cfg_to_native(i) for i in d]
    
    return d