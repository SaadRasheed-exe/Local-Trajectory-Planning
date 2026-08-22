# Hybrid A* Planning Framework – System Overview

## Overview

This project implements a modular trajectory planning system based on **Hybrid A***.
The architecture is designed for **team development**, with clearly separated responsibilities and well-defined interfaces between modules.

The system is structured into the following main components:

* Simulation / Environment (Saad)
* Motion & Prediction (Angelo)
* Planning (Hybrid A*) (Daniel)
* Visualization (Diana)
* Control (Diana)

Each module is responsible for a clearly defined part of the pipeline and communicates only via the defined interfaces.

---

# Folder Structure

```text
project/
│
├── main.py                     # entry point
├── models/                     # shared data structures
│   └── models.py
├── simulation/                 # simulation & environment
│   ├── simulation.py
│   └── environment.py
├── motion/                     # vehicle models
├── prediction/                 # object prediction
├── planner/                    # hybrid A* implementation
├── control/                    # tracking / MPC
├── visualization/              # visualization tools
└── utils/                      # helper functions (math, time, etc.)
```

---

# Data Models (`models/`)

**Responsible: Team (shared)**

This module defines all shared data structures used across the system.

It provides a common representation for:

* vehicle state
* dynamic objects
* environment
* planning inputs and outputs

All modules rely on these definitions to ensure consistent data exchange.

---

# Simulation & Environment

**Responsible: Saad (Simulation / Environment)**

This module represents the world and is responsible for:

* maintaining the current simulation state
* updating ego and object states over time
* providing the current environment to the planner
* executing control commands

It acts as the central data source for the entire system.


### Yaw Explanation

![Yaw Animation](imgs/YawAnimation.gif)


## Simulation Interface

Defines how other modules interact with the simulation.

```python
class SimulationInterface:

    def get_ego_state(self) -> EgoStateStamped:
        """Returns the current ego vehicle state."""
        pass

    def get_environment(self) -> Environment:
        """Returns the current environment representation."""
        pass

    def apply_steer_rate(self, steer_rate: float) -> None:
        """Applies the steering rate."""
        pass

    def apply_acceleration(self, acc: float) -> None:
        """Applies the acceleration."""
        pass

    def step(self, dt: float) -> None:
        """Simulates for dt milliseconds."""
        pass
```

## Collision Checking

Provides functionality to validate whether a state is safe.

```python
def check_collision(
    ego: EgoState,
    obj: DynamicObject
) -> bool:
    """Checks whether the ego vehicle collides with a given object."""
    pass
```

### `get_distance_to_objects`

#### 1. Type Definitions & Function Signature

```python
from typing import List, Tuple, Optional, TypedDict

class ObjectDistance(TypedDict):
    id: int
    min_distance: float  # Minimum distance found in the interval as a float

def get_distance_to_objects(
    current_ego: EgoStateStamped,
    previous_ego: EgoStateStamped,
    predicted_env: PredictedEnvironment,
    vehicle_params: VehicleParameters,
    resolution_ms: int
) -> Tuple[Optional[List[ObjectDistance]], bool]:
    """
    Checks the interval between two planning steps for collisions and 
    calculates aggregated distances to all objects.
    """
```

---

#### 2. Functional Requirements

##### FR1: Temporal Sub-sampling
The function must discretize the time interval between `previous_ego.timestamp` and `current_ego.timestamp` into smaller steps defined by `resolution_ms` (e.g., 10 ms).  
This prevents tunneling effects, where collisions between large discrete steps are missed.

---

##### FR2: State Interpolation
For each sub-step, the function must linearly interpolate the states (position `x, y` and orientation `yaw`) for:
- the ego vehicle  
- all objects in `predicted_env`

**Note:**  
Special care must be taken for `yaw` interpolation to correctly handle angle wrap-around (e.g., transition from `+π` to `-π`).

---

##### FR3: Ego Coordinate Transformation (Rear-to-Center)
The `EgoState` reference point is the center of the rear axle.  
For geometric collision checking, this must be transformed to the vehicle’s geometric center at every sub-step.

---

##### FR4: OBB Distance Calculation
The function must model:
- the ego vehicle  
- all dynamic objects  

as Oriented Bounding Boxes (OBB) using their respective length and width.

It must compute the minimum distance between these rectangles.

---

##### FR5: Performance – Early Exit (Critical)
To optimize Hybrid A* search, the function must implement fail-fast logic:

- If at any sub-step the distance to any object is ≤ 0:
  - Immediately terminate execution  
  - Return:
    ```python
    (None, True)
    ```
- No further sub-steps or objects should be processed after a collision is detected

---

##### FR6: Result Aggregation
If the full interval is processed without any collision:

- Return:
  ```python
  (List[ObjectDistance], False)
  ```

- The list must contain:
  - One entry per object ID  
  - The minimum distance (float) encountered across all interpolated sub-steps  

---


#### 3. Implementation Hints

##### Loop Order
- Iterate time steps first, then objects  
- Ensures early detection of collisions near the start of the trajectory  


---

```python
def get_ego_lane_info(
    ego_state: EgoState
    vehicle_params: VehicleParameters,
    lanes: List[Lane]
    
) -> Tuple[int, float, float, float]:
    """
    Returns:
    lane_id: int,
    distance_to_lane_center: float [m]
    yaw_offset_to_lane_direction: float, [rad]
    lane_occlusion: float [e.g. 0.8]
    """
```



# Motion & Prediction

**Responsible: Angelo (Motion & Prediction)**

This module models how vehicles and objects move over time.

It is responsible for:

* propagating the ego vehicle state (motion models)
* predicting future motion of surrounding objects
* providing time-consistent environment data to the planner

## Kinematic Model (Planner)

Used inside the planner for fast state expansion.

```python
def nonlinear_bicycle_model(
    state: EgoStateStamped,
    control: EgoInput,
    params: VehicleParameters,
    dt: int
) -> EgoStateStamped:
    """
    Computes the next state using a nonlinear kinematic bicycle model.

    Responsibilities:
    - propagate the ego state forward in time
    - respect vehicle geometry (wheelbase, steering limits)

    This model is a simplification of real vehicle dynamics and is used
    inside the planner due to its computational efficiency.
    """
    pass
```

## Dynamic Model (Tracking)

Used for accurate vehicle dynamics in control.

```python
class DynamicBicycleModel:

    def __init__(self, params: VehicleParameters) -> None:
        """
        Initializes the dynamic bicycle model.

        Responsibilities:
        - store vehicle parameters (mass, inertia, tire stiffness, etc.)
        - prepare internal variables required for dynamic simulation

        This model represents the physical behavior of the vehicle more accurately
        than the kinematic model and is used for control and tracking.
        """
        pass

    def step(
        self,
        state: DynamicState,
        control: EgoInput,
        dt: float
    ) -> DynamicState:
        """
        Propagates the vehicle state using a dynamic bicycle model.

        Responsibilities:
        - compute longitudinal and lateral motion
        - update yaw rate and vehicle orientation
        - account for tire forces and vehicle inertia

        Inputs:
        - state: current dynamic state (includes velocities and yaw rate)
        - control: steering and acceleration input
        - dt: time step in seconds

        Output:
        - next dynamic state after dt

        This model is used in the control module to accurately
        follow the planned trajectory and capture real vehicle behavior.
        """
        pass

```

## Prediction

Predicts future motion of surrounding objects.

```python
def predict_constant_velocity(
    obj: DynamicObjectStamped,
    horizon: int,
    dt: int
) -> list[DynamicObjectStamped]:
    """Predicts future object states assuming constant velocity."""
    pass
```

```python
def predict_constant_acceleration(
    obj: DynamicObjectStamped,
    horizon: int,
    dt: int
) -> list[DynamicObjectStamped]:
    """Predicts future object states assuming constant acceleration."""
    pass
```

```python
def predict_environment(
    environment: Environment,
    horizon: int,
    dt: int
) -> PredictedEnvironment:
    """Generates a predicted environment over the planning horizon."""
    pass
```

---

# Planner

**Responsible: Daniel (Hybrid A*)**

The planner computes a feasible trajectory for the ego vehicle.

It is responsible for:

* exploring possible trajectories using Hybrid A*
* evaluating trajectories based on cost and constraints
* ensuring collision-free motion
* returning the best trajectory within the time budget

```python
def plan(request: PlanningRequest) -> PlanResult:
    """
    Computes a trajectory using Hybrid A*.

    Takes the current planning request and returns a planning result.
    """
    pass
```

---

# Control 

**Responsible: Diana**

The control module tracks the planned trajectory and converts it into control inputs.

It is responsible for:

* following the planned trajectory
* handling model inaccuracies
* ensuring smooth and stable vehicle behavior

```python
class Controller:
    def __init__(self, params: VehicleParameters) -> None:
        """
        Initializes the Controller.
        """

    def calc_acceleration(Ego: EgoStateStamped, veh_cfg: DictConfig , Waypoints: Trajectory) -> float:
        """
        Calculate the acceleration, given the Trajectory.
        """


    def calc_steer_rate(Ego: EgoStateStamped, veh_cfg: DictConfig, Waypoints: Trajectory) -> float:
        """
        Calculate the steering-rate, given the Trajectory.
        """
```

---

# Visualization

**Responsible: Diana (Visualization)**

The visualization module is essential for understanding, demonstration and debugging the system.

Responsibilities:

* visualize the ego vehicle state over time
* display dynamic objects and their predicted trajectories
* render lane geometry and environment structure
* display planned trajectories from the planner

This module helps to:

* verify correctness of planning and prediction
* analyze system behavior
* present results in a clear and interpretable way

It visualizes:

* ego vehicle state
* environment and objects
* planned trajectories

Main visualization entry point:

````python
def visualize_scene(
    env: Environment,
    ego: EgoState,
    vehicle_params: VehicleParameters,
    trajectory: Trajectory
) -> None:
    """
    Visualizes the current scene in a continuously updating window.

    Responsibilities:
    - display current environment (lanes and objects)
    - show ego vehicle position, orientation and geometry (length, width)
    - render the planned trajectory
    - update the visualization in real-time as the simulation progresses

    Live Visualization Concept:
    - the function is called repeatedly inside the main loop
    - the plotting backend (e.g. matplotlib) must run in interactive mode
    - the figure should be updated instead of recreated each time
    - typical approach:
        * initialize figure once (outside or inside with static flag)
        * clear/update plot content every call
        * use non-blocking draw (e.g. plt.pause(...))

    Important:
    - the function must be non-blocking (no plt.show() in loop)
    - updates should be fast to allow real-time behavior
    - state is implicitly updated via repeated calls with new inputs

    Inputs:
    - env: current environment
    - ego: current ego state
    - vehicle_params: vehicle geometry for rendering
    - trajectory: planned trajectory
    """
````

---

# Full System Data Flow (with Function Dependencies)

## Overview

```text
Simulation
   ↓
Environment (current)
   ↓
Prediction
   ↓
PredictedEnvironment (future)
   ↓
Planner
   ↓
Trajectory
   ↓
Control
   ↓
Simulation
```

---

# Detailed Flow with Function Dependencies

```text
SIMULATION
    |
    |-- get_ego_state()
    |-- get_environment()
    |
    ↓
[ ENVIRONMENT (t) ]
    |
    ↓
PREDICTION
    |
    |-- predict_environment(...)
    |       |(uses i.a.)
    |       |-- predict_constant_velocity(obj)
    |       |-- predict_constant_acceleration(obj)
    |       |
    |       → builds PredictedEnvironment
    |
    ↓
[ PREDICTED ENVIRONMENT (t → t+horizon) ]
    |
    ↓
PLANNER (Hybrid A*)
    |
    |-- plan(request)
    |       |(uses i.a.)
    |       |-- nonlinear_bicycle_model(state, control)
    |       |       → state propagation for node expansion
    |       |
    |       |-- check_collision(state, obj)
    |       |-- is_collision_free(state, env)
    |       |       → uses check_collision internally
    |       |
    |       → builds trajectory
    |
    ↓
[ TRAJECTORY ]
    |
    ↓
CONTROL (MPC)
    |
    |-- compute_control(trajectory, current_state)
    |       |(uses i.a.)
    |       |-- DynamicBicycleModel.step(...)
    |       |       → accurate vehicle dynamics
    |       |
    |       → outputs EgoInput
    |
    ↓
[ CONTROL INPUT ]
    |
    ↓
SIMULATION
    |
    |-- apply_control(control)
    |-- step(dt)
    |
    ↓
[ NEXT STATE ]

                ↘
                 ↘
                  ↓
        VISUALIZATION
            |
            |-- visualize_scene(...)
            
```
---

# Summary

* Simulation provides the current world state
* Prediction estimates future object motion
* Planner computes a trajectory
* Control executes the trajectory
* Visualization supports debugging and analysis

The system is modular, extensible, and suitable for real-time applications.

---





