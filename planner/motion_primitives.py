from models.models import *
import numpy as np
import math
import timeit
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig, OmegaConf


@dataclass
class MotionPrimitive:
    steering_angle: float       # steering input δ [rad]
    acceleration: float         # longitudinal acceleration [m/s²]  
    dt: int                     # the time the MotionPrimitive is used


def _get_max_steering_angle(velocity: float, veh_cfg: DictConfig, mp_cfg: DictConfig) -> float:
    """
    Compute the maximum feasible steering angle (in radians) based on a lateral
    acceleration constraint using a simplified kinematic bicycle model.

    Model assumption:
        a_lat = (v^2 / L) * tan(delta)

    Solving for the steering angle delta:
        delta = arctan((a_lat * L) / v^2)

    The resulting steering angle is additionally limited by the vehicle's
    physical maximum steering angle (veh_cfg.max_steer).

    Parameters
    ----------
    velocity : float
        Vehicle speed in meters per second (m/s). Must be > 0.
    veh_cfg : DictConfig
        Configuration object containing vehicle dimensions and limits:
        - wheel_base: wheelbase [m]
        - max_steer: maximum steering angle [rad]
    mp_cfg : DictConfig
        Configuration object containing:
        - max_a_lat: Maximum allowable lateral acceleration in meters per second squared (m/s²).

    Returns
    -------
    float
        Maximum steering angle in radians, limited by both the lateral
        acceleration constraint and the vehicle's steering limit.
    """
    # If moving very slowly, the lateral acceleration constraint is irrelevant.
    if abs(velocity) < 0.1: 
        return veh_cfg.max_steer
    
    # Compute steering angle using the rearranged bicycle model formula
    res =  np.arctan((mp_cfg.max_a_lat * veh_cfg.wheel_base) / np.square(velocity))
    return min(res, veh_cfg.max_steer)


def _get_steering_angle_range(velocity: float,
                             steering_angle: float,
                             veh_cfg: DictConfig,
                             mp_cfg: DictConfig,
                             internal_dt: int) -> tuple[float, float]: 
    """
    Compute the feasible steering range based on mechanical limits, 
    actuator rate, and lateral acceleration safety constraints.

    Parameters
    ----------
    velocity : float
        Current vehicle speed in m/s.
    steering_angle : float
        Current Steering angle in radians.
    veh_cfg : DictConfig
        Configuration object containing vehicle constants:
        - wheel_base: wheelbase [m]
        - max_steer: maximum physical steering angle [rad]
        - max_steer_rate: maximum steering rate [rad/s]
    mp_cfg : DictConfig
        Configuration object containing:
        - max_a_lat: Maximum allowable lateral acceleration in m/s².
    internal_dt : int
        Time step duration in milliseconds (ms).

    Returns
    -------
    tuple[float, float]
        Feasible (min_angle, max_angle) in radians.
    """
    
    # 1. Physical Sanity Check
    if abs(steering_angle) > veh_cfg.max_steer:
        raise ValueError("Steering Angle exceeds Vehicle limits (max_steer)")
    
    # 2. Safety & Mechanical: Get max angle allowed by lateral acceleration or physical stop
    max_steering_angle_global = _get_max_steering_angle(velocity, veh_cfg, mp_cfg)

    # 3. Dynamic: Max steering travel possible within the time step (dt)
    max_steering_delta = veh_cfg.max_steer_rate * (internal_dt / 1000)
    min_steering_angle_local = steering_angle - max_steering_delta
    max_steering_angle_local = steering_angle + max_steering_delta

    # 4. Intersection: Limit reachable range by safety and mechanical bounds
    min_steering_angle = max(min_steering_angle_local, -max_steering_angle_global)
    max_steering_angle = min(max_steering_angle_local, max_steering_angle_global)

    # 5. Reachability: Ensure min_steering_angle <= max_steering_angle by collapsing the range 
    # if safety limits force a movement faster than the actuator's rate capacity.
    if min_steering_angle > max_steering_angle:
        max_steering_angle = min_steering_angle
    elif max_steering_angle < min_steering_angle:
        min_steering_angle = max_steering_angle

    return (min_steering_angle, max_steering_angle)


def _get_acceleration_range(velocity: float,
                           velocity_limit: float,
                           acceleration: float,
                           mp_cfg: DictConfig,
                           internal_dt: int) -> tuple[float, float]:
    """
    Calculates the permissible acceleration range [a_min, a_max] for the next time step.
    Takes jerk limits, velocity limits, and tolerance bands into account for predictive 
    and deadbeat control.

    Parameters
    ----------
    velocity : float
        Current vehicle speed in m/s.
    velocity_limit : float
        The target maximum speed limit in m/s.
    acceleration : float
        Current vehicle acceleration in m/s².
    mp_cfg : DictConfig
        Configuration object containing:
        - velocity_limit_tolerance: Upper tolerance above the velocity limit. Exceeding this triggers maximum hard braking.
        - velocity_limit_thresh: Threshold defining the docking zone (below the limit and above 0) to smoothly transition into deadbeat control and prevent chattering.
        - acceleration_limit: Absolute physical maximum acceleration bound (> 0) in m/s².
        - deceleration_limit: Absolute physical maximum deceleration bound (< 0) in m/s².
        - positiv_jerk_limit: Maximum allowed rate of increasing acceleration (> 0) in m/s³.
        - negativ_jerk_limit: Maximum allowed rate of decreasing acceleration (< 0) in m/s³.
    internal_dt : int
        Time step duration in seconds (ms).

    Returns
    -------
    tuple[float, float]
        Feasible (min_acceleration, max_acceleration) range in m/s² for the next time step.
    """
    #Conversion from milliseconds to seconds
    internal_dt_sec = internal_dt / 1000.0

    # Prevent Division by Zero if internal_dt is strictly 0 (failsafe)
    if internal_dt_sec <= 0.0:
        return 0.0, 0.0

    # --- 1. Physical limits for the current time step (Jerk limitation) ---
    # Determine the mechanically achievable acceleration boundaries for this specific time step.
    abs_max_a = min(acceleration + (mp_cfg.positiv_jerk_limit * internal_dt_sec), mp_cfg.acceleration_limit)
    abs_min_a = max(acceleration + (mp_cfg.negativ_jerk_limit * internal_dt_sec), mp_cfg.deceleration_limit)

    # --- 2. Calculate maximum permissible acceleration (a_accel_max) ---
    
    # CASE A: Velocity strictly exceeds the upper tolerance band.
    if velocity > (velocity_limit + mp_cfg.velocity_limit_tolerance):
        # Force maximum possible deceleration (initiate hard braking).
        a_accel_max = abs_min_a
        a_decel_max = abs_min_a

        return float(a_decel_max), float(a_accel_max)

    # CASE B: Velocity is in the upper docking zone or within the tolerance band.
    elif velocity >= (velocity_limit - mp_cfg.velocity_limit_thresh):
        # Apply deadbeat control to hit V_limit exactly, strictly bounded by physical jerk limits.
        req_a_max = (velocity_limit - velocity) / internal_dt_sec    
        a_accel_max = min(req_a_max, abs_max_a)
        a_accel_max = max(a_accel_max, abs_min_a)
        
    # CASE C: Velocity is safely below the limit (Predictive look-ahead).
    else:
        # Simulate: If we start reducing acceleration to 0 right now,
        # what will our resulting velocity be?
        predicted_v = velocity
        temp_a = acceleration
        
        # Calculate the discrete steps required to bring current acceleration down to 0.
        steps_to_zero = math.ceil(abs(temp_a / (mp_cfg.negativ_jerk_limit * internal_dt_sec)))
        
        for _ in range(steps_to_zero):
            # Simulate gradually shedding acceleration towards 0.
            temp_a = max(0.0, temp_a + (mp_cfg.negativ_jerk_limit * internal_dt_sec))
            predicted_v += temp_a * internal_dt_sec

        # If the predicted velocity enters the upper docking zone, enforce maximum deceleration limit.
        if predicted_v > (velocity_limit - mp_cfg.velocity_limit_thresh):
            a_accel_max = abs_min_a
        else:
            a_accel_max = abs_max_a

    # --- 3. Calculate minimum permissible acceleration (a_decel_max) ---
    # Goal: Prevent the vehicle from reversing (v < 0).

    # CASE D: Velocity is in the lower docking zone (approaching standstill).
    if velocity <= mp_cfg.velocity_limit_thresh:
        # Apply deadbeat control to bring velocity exactly to 0, bounded by physical jerk limits.
        req_a_min = (0.0 - velocity) / internal_dt_sec
        a_decel_max = min(req_a_min, abs_max_a)
        a_decel_max = max(a_decel_max, abs_min_a)
    
    # CASE E: Velocity is safely above 0 (Predictive look-ahead for stopping).
    else:
        predicted_v_min = velocity
        temp_a_min = acceleration
        
        # Calculate the discrete steps required to bring negative acceleration (braking) back up to 0.
        steps_to_zero_dec = math.ceil(abs(temp_a_min / (mp_cfg.positiv_jerk_limit * internal_dt_sec)))
        
        for _ in range(steps_to_zero_dec):
            # Simulate gradually reducing the braking force towards 0.
            temp_a_min = min(0.0, temp_a_min + (mp_cfg.positiv_jerk_limit * internal_dt_sec))
            predicted_v_min += temp_a_min * internal_dt_sec 

        # If the predicted velocity drops into the lower docking zone, force an increase in acceleration.
        if predicted_v_min < mp_cfg.velocity_limit_thresh:
            a_decel_max = abs_max_a 
        else:
            a_decel_max = abs_min_a

    # --- 4. Return as [Min, Max] ---
    # Safety check: Ensure the minimum bound never exceeds the maximum bound.
    if a_decel_max > a_accel_max:
        a_decel_max = a_accel_max
        
    return float(a_decel_max), float(a_accel_max)



def get_motion_primitives(velocity: float,
                          steering_angle: float,
                          veh_cfg: DictConfig,
                          velocity_limit: float,
                          acceleration: float,
                          mp_cfg: DictConfig,
                          internal_dt: int) -> list[MotionPrimitive]:
    """
    Generates a deterministic set of feasible motion primitives (action trajectories) 
    for a Hybrid A* planner based on vehicle dynamics and kinematic limits.

    Parameters
    ----------
    velocity : float
        Current vehicle speed in m/s.
    steering_angle : float
        Current steering angle in radians.
    veh_cfg : DictConfig
        Configuration object containing physical vehicle dimensions and constraints 
        (e.g., max_steer, wheel_base).
    velocity_limit : float
        The target maximum speed limit in m/s.
    acceleration : float
        Current vehicle acceleration in m/s².
    mp_cfg : DictConfig
        Configuration object containing:
        - acceleration_split: List of percentages (e.g., [0, 50, 100]) to sample the feasible acceleration range. 
          0% represents max deceleration, 100% represents max acceleration.
        - steering_angle_split: List of percentages (e.g., [-100, 0, 100]) to sample the feasible steering range.
          -100% represents max right/left, 0% is straight, 100% represents max left/right.
        - max_a_lat: Maximum allowable lateral acceleration in m/s².
        - velocity_limit_tolerance: Upper tolerance above the velocity limit. Exceeding this triggers braking.
        - velocity_limit_thresh: Threshold defining the docking zone to smoothly transition into deadbeat control.
        - acceleration_limit: Absolute physical maximum acceleration bound (> 0) in m/s².
        - deceleration_limit: Absolute physical maximum deceleration bound (< 0) in m/s².
        - positiv_jerk_limit: Maximum allowed rate of increasing acceleration (> 0) in m/s³.
        - negativ_jerk_limit: Maximum allowed rate of decreasing acceleration (< 0) in m/s³.
    internal_dt : int
        Time step duration for the primitive in milliseconds (ms).

    Returns
    -------
    list[MotionPrimitive]
        A list of physically feasible and safe motion primitives for the next expansion step.
    """
    
    # 1. Initialize the empty list for the generated motion primitives
    motion_primitives: list[MotionPrimitive] = []
    
    # 2. Get the dynamically feasible minimum and maximum acceleration bounds
    acceleration_min, acceleration_max = _get_acceleration_range(
        velocity = velocity,
        velocity_limit = velocity_limit,
        acceleration = acceleration,
        mp_cfg = mp_cfg,
        internal_dt = internal_dt
    )
    
    # Time conversion for kinematic projection calculations
    internal_dt_sec = internal_dt / 1000.0
    
    # Compute the total available acceleration span
    acceleration_delta = acceleration_max - acceleration_min
    acceleration_values: list[float] = []

    # 3. Generate the discrete acceleration samples based on the provided percentages
    for split in mp_cfg.acceleration_split:
        # Convert the percentage split (0-100) into an actual acceleration offset
        acceleration_delta_split = acceleration_delta * (split / 100.0)
        acceleration_value = acceleration_min + acceleration_delta_split
        
        # Append uniquely to maintain deterministic ordering without duplicates
        if acceleration_value not in acceleration_values:
            acceleration_values.append(acceleration_value)

    # 4. Generate steering samples for each valid acceleration sample
    for acceleration_value in acceleration_values:
        
        # Predict the resulting velocity at the end of this primitive's time step.
        # Bounded to >= 0.0 to prevent evaluating backward motion physics.
        target_velocity = max(0.0, velocity + (acceleration_value * internal_dt_sec))
        
        # Retrieve the dynamically feasible steering bounds for the predicted target velocity
        min_angle, max_angle = _get_steering_angle_range(
            velocity = target_velocity,
            steering_angle = steering_angle,
            veh_cfg = veh_cfg,
            mp_cfg = mp_cfg,
            internal_dt = internal_dt
        )

        # Check if the steering envelope is collapsed (e.g., restricted by lateral acceleration).
        if abs(max_angle - min_angle) < 1e-5:
            
            # Instantiate a single primitive to avoid creating multiple identical duplicates
            motion_primitive = MotionPrimitive(
                steering_angle=min_angle,
                acceleration=acceleration_value,
                dt=internal_dt
            )
            motion_primitives.append(motion_primitive)
            
        else:
            # Calculate the geometric center and radius of the available steering envelope
            center_point = (max_angle + min_angle) / 2.0
            distance_from_center = abs(max_angle - center_point)

            # Generate discrete steering commands based on the provided percentages
            for split in mp_cfg.steering_angle_split:
                # Convert the percentage split (-100 to 100) into a steering offset
                steering_angle_delta_split = distance_from_center * (split / 100.0)
                steering_angle_value = center_point + steering_angle_delta_split
                
                # Instantiate the primitive and add it to the expansion list
                motion_primitive = MotionPrimitive(
                    steering_angle = steering_angle_value,
                    acceleration = acceleration_value,
                    dt = internal_dt
                )
                motion_primitives.append(motion_primitive)

    # Sort primitives to optimize A* expansion (straight and constant speed first):
    # 1. Absolute steering magnitude (closest to 0 straight ahead first).
    # 2. Steering sign preference (positive steering angles before negative ones).
    # 3. Absolute acceleration magnitude (closest to 0 m/s² / constant speed first).
    # 4. Acceleration sign preference (positive acceleration before braking).
    motion_primitives.sort(key=lambda mp: (
        round(abs(mp.steering_angle), 5), 
        -mp.steering_angle, 
        round(abs(mp.acceleration), 5),
        -mp.acceleration
    ))
    
    return motion_primitives



def print_motion_primitives(primitives: list[MotionPrimitive]) -> None:
    """
    Prints a list of MotionPrimitives as a clearly formatted table in the console.
    """
    if not primitives:
        print("The list of Motion Primitives is empty.")
        return

    # 1. Define the table header (with fixed column widths)
    header = f"{'No.':<5} | {'Steering Angle [rad]':<22} | {'Acceleration [m/s²]':<22} | {'dt [ms]':<10}"
    print("\n" + header)
    
    # 2. Dynamically adjust the separator line to match the header length
    print("-" * len(header))

    # 3. Iterate through all primitives and print them with formatting
    for i, mp in enumerate(primitives, start=1):
        steer_str = f"{mp.steering_angle:.5f}"
        accel_str = f"{mp.acceleration:.5f}"
        
        print(f"{i:<5} | {steer_str:<22} | {accel_str:<22} | {mp.dt:<10}")
    
    print(f"{'-' * len(header)}")
    print(f"Total: {len(primitives)} Motion Primitives\n")  
    
    

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    
    ## Benchmarking Acceleration Range
    t  = timeit.timeit(lambda: _get_acceleration_range(
                           velocity = 55.55,
                           velocity_limit = 13.88,
                           acceleration = 1.8,
                           mp_cfg = cfg.motion_primitives,
                           internal_dt = 100),
                           number=100)
    print(f"Average time (_get_acceleration_range): {(t/100) * 1000:.6f} ms")
    
    # Get and print primitives
    res = get_motion_primitives(velocity=55.55,
                                steering_angle=0.7,
                                veh_cfg=cfg.vehicle,        
                                velocity_limit=60,
                                acceleration=0,
                                mp_cfg=cfg.motion_primitives,
                                internal_dt=100)
    
    print_motion_primitives(res)

    ## Benchmarking get_motion_primitives
    t  = timeit.timeit(lambda: get_motion_primitives(velocity=5,
                                                    steering_angle=0,
                                                    veh_cfg=cfg.vehicle,    
                                                    velocity_limit=13.88,
                                                    acceleration=0,
                                                    mp_cfg=cfg.motion_primitives,
                                                    internal_dt=100),
                                                    number = 1000)
    
    print(f"Average time (get_motion_primitives): {(t/1000) * 1000:.6f} ms")


if __name__ == "__main__":
    main()