from math import sin, cos
from typing import List
from models.models import DynamicObject, DynamicObjectStamped, Vector2D

def predict_motion_constant_velocity(objects: List[DynamicObjectStamped], prediction_horizon: int, dt: int, last_only: bool = False) -> List[DynamicObjectStamped]:
    """
    Predict the future states of dynamic objects using a simple constant velocity model.
    
    Args:
        objects: A list of DynamicObjectStamped representing the current states of dynamic objects.
        prediction_horizon: The total time (in milliseconds) for which to predict the motion.
        dt: The time step (in milliseconds) between predictions.
        last_only: If True, only the last predicted state for each object will be returned.
    
    Returns:
        A list of DynamicObjectStamped representing the predicted future states of the dynamic objects at each time step.
    """
    predicted_objects = []
    
    for obj in objects:
        current_state = obj.state
        predicted_states = []
        
        for t in range(0, prediction_horizon + dt, dt):
            new_x = current_state.pos.x + current_state.velocity.x * (t / 1000.0)
            new_y = current_state.pos.y + current_state.velocity.y * (t / 1000.0)
            new_pos = Vector2D(x=new_x, y=new_y)
            
            predicted_state = DynamicObjectStamped(
                timestamp=obj.timestamp + t,
                state=DynamicObject(
                    id=current_state.id,
                    obj_class=current_state.obj_class,
                    pos=new_pos,
                    yaw=current_state.yaw,
                    velocity=current_state.velocity,
                    acceleration=Vector2D(x=0.0, y=0.0),  # Assuming constant velocity, so acceleration is zero
                    width=current_state.width,
                    length=current_state.length
                )
            )
            predicted_states.append(predicted_state)
        
        if last_only:
            predicted_objects.append(predicted_states[-1])
        else:
            predicted_objects.extend(predicted_states)
    
    return predicted_objects


def predict_motion_constant_acceleration(objects: List[DynamicObjectStamped], prediction_horizon: int, dt: int) -> List[DynamicObjectStamped]:
    """
    Predict the future states of dynamic objects using a simple constant acceleration model.
    
    Args:
        objects: A list of DynamicObjectStamped representing the current states of dynamic objects.
        prediction_horizon: The total time (in milliseconds) for which to predict the motion.
        dt: The time step (in milliseconds) between predictions.
    
    Returns:
        A list of DynamicObjectStamped representing the predicted future states of the dynamic objects at each time step.
    """
    predicted_objects = []
    
    for obj in objects:
        current_state = obj.state
        predicted_states = []
        
        for t in range(0, prediction_horizon + dt, dt):
            new_x = current_state.pos.x + current_state.velocity.x * (t / 1000.0) + 0.5 * current_state.acceleration.x * ((t / 1000.0) ** 2)
            new_y = current_state.pos.y + current_state.velocity.y * (t / 1000.0) + 0.5 * current_state.acceleration.y * ((t / 1000.0) ** 2)
            new_pos = Vector2D(x=new_x, y=new_y)
            
            new_velocity = Vector2D(x=current_state.velocity.x + current_state.acceleration.x * (t / 1000.0), y=current_state.velocity.y + current_state.acceleration.y * (t / 1000.0))
            
            predicted_state = DynamicObjectStamped(
                timestamp=obj.timestamp + t,
                state=DynamicObject(
                    id=current_state.id,
                    obj_class=current_state.obj_class,
                    pos=new_pos,
                    yaw=current_state.yaw,
                    velocity=new_velocity,
                    acceleration=current_state.acceleration,  # Assuming constant acceleration
                    width=current_state.width,
                    length=current_state.length
                )
            )
            predicted_states.append(predicted_state)
        
        predicted_objects.extend(predicted_states)
    
    return predicted_objects