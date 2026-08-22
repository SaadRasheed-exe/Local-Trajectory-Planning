from typing import Tuple, List
import math
from models.models import EgoStateStamped, Vector2D

def get_x_y_yaw_from_state(state: EgoStateStamped) -> Tuple[float, float, float]:
    """
    Extracts the global x, y, and yaw coordinates from an EgoStateStamped object.

    Args:
        state (EgoStateStamped): The ego vehicle state.

    Returns:
        Tuple[float, float, float]: A tuple containing (x [m], y [m], yaw [rad]).
    """
    return state.state.pos.x, state.state.pos.y, state.state.yaw


def get_bbox_corners(
    center_x: float, 
    center_y: float, 
    yaw: float, 
    length: float, 
    width: float
) -> List[Vector2D]:
    """
    Calculates the global coordinates for the four corners of a bounding box 
    based on its center point, dimensions, and rotation (yaw).

    Args:
        center_x (float): The x-coordinate of the bounding box center [m].
        center_y (float): The y-coordinate of the bounding box center [m].
        yaw (float): The orientation/heading of the bounding box [rad].
        length (float): The total longitudinal length of the bounding box [m].
        width (float): The total lateral width of the bounding box [m].

    Returns:
        List[Vector2D]: A list of the four corner points in global coordinates.
                        Order: [Front-Left, Front-Right, Rear-Right, Rear-Left].
    """
    
    # Precompute trigonometric values for the rotation matrix
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    
    # Calculate half-dimensions for local coordinate offsets
    hl = length / 2.0
    hw = width / 2.0
    
    # Define corners in the local coordinate system (origin at center_x, center_y = 0,0)
    # x points forward, y points left
    local_corners = [
        (hl, hw),    # Front-Left
        (hl, -hw),   # Front-Right
        (-hl, -hw),  # Rear-Right
        (-hl, hw)    # Rear-Left
    ]
    
    corners = []
    for lx, ly in local_corners:
        # Apply 2D rotation matrix and translate to the global center position
        global_x = center_x + (lx * cos_y - ly * sin_y)
        global_y = center_y + (lx * sin_y + ly * cos_y)
        
        corners.append(Vector2D(x=global_x, y=global_y))
        
    return corners