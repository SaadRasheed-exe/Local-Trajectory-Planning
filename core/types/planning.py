from dataclasses import dataclass
from typing import List, Optional

from core.types.geometry import Vector2D
from core.types.perception import PredictedEnvironment
from core.types.vehicle import EgoStateStamped


@dataclass
class GoalRegion:
    center: Vector2D            # center position [m]

    length: float               # longitudinal size [m]
    width: float                # lateral size [m]

    yaw: float                  # desired heading [rad]


@dataclass
class PlanningRequest:
    start_state: EgoStateStamped   # initial state for planning

    goal_region: GoalRegion     # target region
    target_speed: float         # target speed

    environment: PredictedEnvironment


@dataclass
class Trajectory:
    states: List[EgoStateStamped]   # time-parameterized ego states


@dataclass
class PlanResult:
    success: bool                       # True if valid plan found
    goal_region_reached: bool           # True if trajectory reaches goal_region

    trajectory: Optional[Trajectory]    # resulting trajectory (None if failed)
    cost: float                         # total trajectory cost

    status_message: str                 # diagnostic info

    debug_root_node: Optional[object] = None
    # planner-private search-tree root (e.g. Hybrid A* StateNode); typed loosely
    # so that core never depends on a concrete planner implementation.
