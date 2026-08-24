from dataclasses import dataclass

from core.types.geometry import Vector2D


@dataclass
class EgoState:
    pos: Vector2D               # position (x, y) in global coordinate system [m]
    velocity: Vector2D          # velocity (x,y) in global coordinate system [m/s]
    acceleration: Vector2D      # acceleration (x,y) in global coordinate system [m/s]
    yaw: float                  # heading angle in global coordinate system [rad]
    steering_angle: float       # angle of the wheels [rad]


@dataclass
class EgoStateStamped:
    timestamp: int              # time [ms]
    state: EgoState             # ego state at this time


@dataclass
class EgoInput:
    steering_angle: float       # steering input δ [rad]
    acceleration: float         # longitudinal acceleration [m/s²]
