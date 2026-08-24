from dataclasses import dataclass
from typing import List, Optional, Tuple

from core.types.agents import DynamicObjectStamped
from core.types.geometry import Vector2D


@dataclass
class Lane:
    id: int                     # unique lane identifier

    centerline: List[Tuple[Vector2D, float]]  # ordered centerline points with tangents [[m, m], rad]

    width: float                # total lane width [m] (± width/2 from centerline)

    speed_limit: float          # speed limit [m/s]

    cumulative_lengths: Optional[List[float]] = None
    # arc length s at each centerline point [m]
    # used for fast longitudinal position computation


@dataclass
class Environment:
    objects: List[DynamicObjectStamped]
    # list of current object states (no prediction)

    lanes: List[Lane]
    # static road geometry
