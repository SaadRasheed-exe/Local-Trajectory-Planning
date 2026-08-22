import math
from planner.cost.cost_utils import *
from models.models import EgoStateStamped
from shapely.geometry import Polygon
from collision.collision import get_distance_to_objects
from models.models import *
from omegaconf import DictConfig
from utils.helper import get_magnitude, get_signed_magnitude


#--- Comfort Costs ---------------------------------------------------------------------

def get_dt_seconds(prev_state: EgoStateStamped, curr_state: EgoStateStamped) -> float:
    """
    Calculates the time difference between two states in seconds.

    Args:
        prev_state (EgoStateStamped): The previous state of the ego vehicle.
        curr_state (EgoStateStamped): The current state of the ego vehicle.

    Returns:
        float: Time difference in seconds, safely clamped to a minimum of 1e-6 to prevent division by zero.
    """
    dt = (curr_state.timestamp - prev_state.timestamp) / 1000.0
    return max(dt, 1e-6)

def cost_jerk(prev_state: EgoStateStamped, curr_state: EgoStateStamped):
    """
    Calculates the squared jerk costs in the vehicle's local longitudinal and lateral directions.

    Args:
        prev_state (EgoStateStamped): The previous state of the ego vehicle.
        curr_state (EgoStateStamped): The current state of the ego vehicle.

    Returns:
        dict: A dictionary containing the squared jerk costs with keys "long" and "lat".
    """
    dt = get_dt_seconds(prev_state, curr_state)
    
    # Calculate global jerk via finite differences
    jerk_x = (curr_state.state.acceleration.x - prev_state.state.acceleration.x) / dt
    jerk_y = (curr_state.state.acceleration.y - prev_state.state.acceleration.y) / dt
    
    # Project into local vehicle coordinate system using current yaw
    yaw = curr_state.state.yaw
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    
    jerk_long = jerk_x * cos_y + jerk_y * sin_y
    jerk_lat = -jerk_x * sin_y + jerk_y * cos_y
    
    # Cost calculation 
    cost_long = jerk_long ** 2
    cost_lat = jerk_lat ** 2
    
    return {"long": cost_long, "lat": cost_lat}

def cost_acceleration(prev_state: EgoStateStamped, curr_state: EgoStateStamped):
    """
    Calculates the squared acceleration costs in local longitudinal and lateral directions 
    using the average acceleration over the time step.

    Args:
        prev_state (EgoStateStamped): The previous state of the ego vehicle.
        curr_state (EgoStateStamped): The current state of the ego vehicle.

    Returns:
        dict: A dictionary containing the squared acceleration costs with keys "long" and "lat".
    """
    dt = get_dt_seconds(prev_state, curr_state)
    
    accel_x = (curr_state.state.velocity.x - prev_state.state.velocity.x) / dt
    accel_y = (curr_state.state.velocity.y - prev_state.state.velocity.y) / dt
    
    yaw = curr_state.state.yaw
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    
    accel_long = accel_x * cos_y + accel_y * sin_y
    accel_lat = -accel_x * sin_y + accel_y * cos_y
    
    # Cost calculation
    cost_long = accel_long ** 2
    cost_lat = accel_lat ** 2
    
    return {"long": cost_long, "lat": cost_lat}


#--- Safety Costs ------------------------------------------------------------------------

def cost_target_speed_delta(
    curr_state: EgoStateStamped, 
    target_speed: float,
    weight_underspeed: float = 1.0,      
    weight_overspeed: float = 5.0,       
    exp_growth_factor: float = 1.0,      
    max_penalty: float = 1e6             
) -> float:
    """
    Calculates the cost for deviating from the target speed. Applies a linear penalty 
    for driving too slow and an exponential penalty barrier for driving too fast.

    Args:
        curr_state (EgoStateStamped): The current state of the ego vehicle.
        target_speed (float): The desired target speed [m/s].
        weight_underspeed (float): Linear weight applied when velocity is below target.
        weight_overspeed (float): Base weight applied when velocity exceeds target.
        exp_growth_factor (float): Factor controlling the steepness of the overspeed exponential curve.
        max_penalty (float): Absolute maximum cost limit to prevent math overflows.

    Returns:
        float: The calculated penalty cost for speed deviation.
    """
    
    current_speed = get_signed_magnitude(curr_state.state.velocity, curr_state.state.yaw)
    
    delta_v = current_speed - target_speed
    
    if delta_v < 0:
        # Linear penalty for underspeed
        return weight_underspeed * abs(delta_v)
        
    # Exponential barrier for overspeed with overflow protection
    safe_exponent = min(exp_growth_factor * delta_v, 50.0)
    raw_cost = weight_overspeed * (math.exp(safe_exponent) - 1.0)
    
    return min(raw_cost, max_penalty) 



def cost_objects_force_field(
    current_ego: EgoStateStamped,
    previous_ego: EgoStateStamped,
    predicted_env: PredictedEnvironment,
    veh_cfg: DictConfig,
    cost_cfg: DictConfig
) -> float:
    """
    Calculates proximity costs to dynamic objects using an asymmetrical dual force-field approach.
    Aggregates costs via LogSumExp to safely center the vehicle in narrow gaps while maintaining worst-case dominance.

    Args:
        current_ego (EgoStateStamped): The current kinematic and spatial state of the ego vehicle.
        previous_ego (EgoStateStamped): The previous state of the ego vehicle, used for continuous collision checking.
        predicted_env (PredictedEnvironment): Environment data containing future trajectory predictions for dynamic objects.
        veh_cfg (DictConfig): Vehicle configuration containing physical dimensions (e.g., length, width, rear_to_wheel).
        cost_cfg (DictConfig): Cost configuration containing all tuning parameters and the collision checker resolution.

    Returns:
        float: The total aggregated proximity cost. Returns float('inf') if a hard collision is detected. 
               Otherwise, returns a LogSumExp aggregated value of exponential (high risk) and quadratic (low risk) costs.
    """
    
    # Extract specific force field parameters for direct access
    ff_cfg = cost_cfg.cost_objects_force_field
    
    # 1. Get exact distances and check for hard collisions using the resolution from config
    distances, is_collision = get_distance_to_objects(
        current_ego = current_ego,
        previous_ego = previous_ego,
        predicted_env = predicted_env, 
        ego_length = veh_cfg.length,
        ego_width = veh_cfg.width,
        ego_rear_to_wheel = veh_cfg.rear_to_wheel,
        resolution_ms = cost_cfg.cost_objects_force_field.resolution_ms,
        calc_exact_distance = cost_cfg.cost_objects_force_field.d_ffl
    )
    
    if is_collision:
        return float('inf')
        
    if not distances:
        return 0.0

    # 2. Construct the Ego's Dynamic "Force Field High" Polygon
    ego_x, ego_y, ego_yaw = get_x_y_yaw_from_state(current_ego)
    
    # Extract Ego kinematics using the helper function
    ego_speed = get_signed_magnitude(current_ego.state.velocity, current_ego.state.yaw)
    ego_accel = get_signed_magnitude(current_ego.state.acceleration, current_ego.state.yaw)
    
    # Calculate dynamic expansion for the ego vehicle
    ego_front_exp = ego_speed * ff_cfg.speed_expansion_factor_front + ego_accel * ff_cfg.accel_expansion_factor_front
    ego_rear_exp = ego_speed * ff_cfg.speed_expansion_factor_rear + ego_accel * ff_cfg.accel_expansion_factor_rear
    
    # Shift reference point from the rear axle to the geometric center
    ego_x_center = ego_x + (veh_cfg.length / 2.0 - veh_cfg.rear_to_wheel) * math.cos(ego_yaw)
    ego_y_center = ego_y + (veh_cfg.length / 2.0 - veh_cfg.rear_to_wheel) * math.sin(ego_yaw)
    
    # Apply asymmetrical dynamic shift
    ego_shift = (ego_front_exp - ego_rear_exp) / 2.0
    ego_ffh_center_x = ego_x_center + ego_shift * math.cos(ego_yaw)
    ego_ffh_center_y = ego_y_center + ego_shift * math.sin(ego_yaw)
    
    # Calculate the total dynamic bounding box dimensions
    ego_ffh_length = veh_cfg.length + 2 * ff_cfg.ego_safe_margin + ego_front_exp + ego_rear_exp
    ego_ffh_width = veh_cfg.width + 2 * ff_cfg.ego_safe_margin
    
    ego_corners = get_bbox_corners(
        ego_ffh_center_x, ego_ffh_center_y, ego_yaw, ego_ffh_length, ego_ffh_width
    )
    ego_ffh_polygon = Polygon([(c.x, c.y) for c in ego_corners])

    sum_exp = 0.0
    
    # 3. Evaluate force fields for each relevant object
    for obj_id, base_dist in distances:
        obj_states = predicted_env.objects.get(obj_id, [])
        if not obj_states:
            continue
            
        # Find the object state that chronologically matches the current ego timestamp
        current_obj_state = obj_states[0]
        for state in obj_states:
            if state.timestamp >= current_ego.timestamp:
                current_obj_state = state
                break

        obj_x = current_obj_state.state.pos.x
        obj_y = current_obj_state.state.pos.y
        obj_yaw = current_obj_state.state.yaw
        
        # Extract object kinematics seamlessly using the Vector2D helper function
        obj_speed = get_signed_magnitude(current_obj_state.state.velocity, current_obj_state.state.yaw)
        obj_accel = get_signed_magnitude(current_obj_state.state.acceleration, current_obj_state.state.yaw)
        
        # Calculate dynamic expansion for the object
        obj_front_exp = obj_speed * ff_cfg.speed_expansion_factor_front + obj_accel * ff_cfg.accel_expansion_factor_front
        obj_rear_exp = obj_speed * ff_cfg.speed_expansion_factor_rear + obj_accel * ff_cfg.accel_expansion_factor_rear
        
        # Base dimensions plus dynamic shift
        base_length = current_obj_state.state.length + 2 * ff_cfg.obj_safe_margin
        ff_high_width = current_obj_state.state.width + 2 * ff_cfg.obj_safe_margin
        
        ff_high_length = base_length + obj_front_exp + obj_rear_exp
        shift_distance = (obj_front_exp - obj_rear_exp) / 2.0
        
        ff_center_x = obj_x + shift_distance * math.cos(obj_yaw)
        ff_center_y = obj_y + shift_distance * math.sin(obj_yaw)
        
        obj_corners = get_bbox_corners(
            ff_center_x, ff_center_y, obj_yaw, ff_high_length, ff_high_width
        )
        obj_ffh_polygon = Polygon([(c.x, c.y) for c in obj_corners])

        # 4. Cost Mapping (Intersection vs. Proximity)
        cost_low = 0.0
        if base_dist < ff_cfg.d_ffl:
            # FFL Low: Quadratic virtual spring for smooth anticipation
            cost_low = ff_cfg.weight_ff_low * ((ff_cfg.d_ffl - base_dist) / ff_cfg.d_ffl) ** 2
            
        cost_high = 0.0
        if ego_ffh_polygon.intersects(obj_ffh_polygon):
            # FFH High: Exponential cost to handle narrow gaps securely
            cost_high = ff_cfg.weight_ff_high * math.exp(-ff_cfg.decay_factor_high * base_dist)

            
        cost = cost_low + cost_high
                
        # 5. LogSumExp Aggregation Preparation
        sum_exp += math.exp(ff_cfg.lse_alpha * cost)
            
    # Final LogSumExp evaluation
    if sum_exp == 0.0:
        return 0.0
        
    total_lse_cost = (1.0 / ff_cfg.lse_alpha) * math.log(sum_exp)
    return total_lse_cost



def cost_lane_center_distance(distance_to_center: float) -> float:
    """
    Calculates the unweighted proximity cost to the lane center.
    Uses a quadratic mapping to penalize large deviations aggressively 
    while allowing slight, smooth deviations.
    """
    return distance_to_center ** 2


def cost_lane_yaw_offset(yaw_offset: float) -> float:
    """
    Calculates the unweighted cost for heading misalignment with the lane.
    Uses a quadratic mapping to strictly penalize driving crossways to the lane direction.
    """
    return yaw_offset ** 2


def cost_lane_occlusion(lane_occlusion: float, exp_growth_factor: float = 5.0) -> float:
    """
    Calculates the unweighted cost for leaving the drivable lane area.
    
    Args:
        lane_occlusion: Percentage of the vehicle currently on the road [0.0 - 1.0].
        exp_growth_factor: Determines how steeply the cost explodes when leaving the road.
        
    Returns:
        Exponentially growing cost based on the off-road ratio.
    """
    # off_road_ratio is 0.0 when fully on the road, 1.0 when fully off-road
    off_road_ratio = 1.0 - lane_occlusion
    
    # Small tolerance to avoid penalizing micro-deviations (e.g., 99% on road)
    if off_road_ratio > 0.01:
        return math.exp(exp_growth_factor * off_road_ratio) - 1.0
        
    return 0.0


def cost_opposite_lane(opposite_lane: bool, penalty: float = 1000.0) -> float:
    """
    Calculates the cost for driving on an oncoming traffic lane.
    Uses a static step-function penalty.
    """
    return penalty if opposite_lane else 0.0