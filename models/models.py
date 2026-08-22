from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum


# ============================================================
# Basic Geometry
# ============================================================

@dataclass
class Vector2D:
    x: float                    # x-coordinate [m]
    y: float                    # y-coordinate [m]


# ============================================================
# Ego Vehicle State (Planning)
# ============================================================

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


# ============================================================
# Control Input 
# ============================================================

@dataclass
class EgoInput:
    steering_angle: float       # steering input δ [rad]
    acceleration: float         # longitudinal acceleration [m/s²]


# ============================================================
# Dynamic Objects 
# ============================================================

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





# ============================================================
# Lane Representation
# ============================================================

@dataclass
class Lane:
    id: int                     # unique lane identifier

    centerline: List[Tuple[Vector2D, float]]  # ordered centerline points with tangents [[m, m], rad]

    width: float                # total lane width [m] (± width/2 from centerline)

    speed_limit: float          # speed limit [m/s]

    cumulative_lengths: Optional[List[float]] = None
    # arc length s at each centerline point [m]
    # used for fast longitudinal position computation


# ============================================================
# Environment (Current State from Simulation / Perception)
# ============================================================

@dataclass
class Environment:
    objects: List[DynamicObjectStamped]
    # list of current object states (no prediction)

    lanes: List[Lane]
    # static road geometry


# ============================================================
# Environment (Prediction Output)
# ============================================================

@dataclass
class PredictedEnvironment:
    objects: Dict[int, List[DynamicObjectStamped]] 
    # mapping: object_id → predicted states over time

    lanes: List[Lane]
    # static road geometry

    dt: int
    # time resolution of prediction [ms]

    horizon: int
    # prediction horizon [ms]


# ============================================================
# Goal Definition
# ============================================================

@dataclass
class GoalRegion:
    center: Vector2D            # center position [m]

    length: float               # longitudinal size [m]
    width: float                # lateral size [m]

    yaw: float                  # desired heading [rad]


# ============================================================
# Planner Interface
# ============================================================

@dataclass
class PlanningRequest:
    start_state: EgoStateStamped   # initial state for planning 

    goal_region: GoalRegion     # target region
    target_speed: float         # target speed

    environment: PredictedEnvironment


# ============================================================
# Planner Output
# ============================================================

@dataclass
class Trajectory:
    states: List[EgoStateStamped]   # time-parameterized ego states



from planner.state_node import StateNode


@dataclass
class PlanResult:
    success: bool                       # True if valid plan found
    goal_region_reached: bool         # True if trajectory reaches goal_region
    
    trajectory: Optional[Trajectory]    # resulting trajectory (None if failed)
    cost: float                         # total trajectory cost
    
    status_message: str                 # diagnostic info

    debug_root_node: StateNode