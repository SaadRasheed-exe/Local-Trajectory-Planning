import copy
from math import cos, hypot, pi, sin
from typing import List, Optional, Tuple, Union

from core.types.agents import DynamicObjectStamped
from core.types.geometry import Vector2D
from core.types.vehicle import EgoStateStamped

MovingObject = Union[EgoStateStamped, DynamicObjectStamped]


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


def get_x_y_yaw_from_state(state: MovingObject) -> Tuple[float, float, float]:
    """Extract (x, y, yaw) from any stamped moving-object state (ego or dynamic object)."""
    return state.state.pos.x, state.state.pos.y, state.state.yaw


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
    return Vector2D(x=magnitude * cos(direction), y=magnitude * sin(direction))


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
