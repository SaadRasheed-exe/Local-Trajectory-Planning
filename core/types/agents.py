from dataclasses import dataclass
from enum import Enum

from core.types.geometry import Vector2D


class ObjectType(Enum):
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    STATIC = 4


@dataclass
class DynamicObject:
    id: int
    obj_class: ObjectType       # semantic class of object

    pos: Vector2D               # position [m]
    yaw: float                  # orientation [rad]

    velocity: Vector2D             # velocity (vx, vy) [m/s]
    acceleration: Vector2D         # acceleration (ax, ay) [m/s²]

    width: float                # bounding box width [m]
    length: float               # bounding box length [m]


@dataclass
class DynamicObjectStamped:
    timestamp: int              # time [ms]
    state: DynamicObject        # object state at this time
