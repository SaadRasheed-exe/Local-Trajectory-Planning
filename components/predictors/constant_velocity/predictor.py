"""Object motion prediction models for the planning horizon."""

import math
from typing import Dict, List, Literal, Optional

from core.types.agents import DynamicObject, DynamicObjectStamped
from core.types.geometry import Vector2D
from core.types.perception import PredictedEnvironment
from core.types.road import Environment
from core.geometry import get_signed_magnitude, get_vector

PredictionModel = Literal["constant_acceleration", "constant_turn_rate"]

# Guard for near-zero yaw rate in CTRV: below this, use straight-line fallback.
_YAW_RATE_MIN: float = 1e-4  # [rad/s]

# Guard for near-zero acceleration norm in braking clamp.
_ACCEL_NORM_SQ_MIN: float = 1e-9


def _validate_prediction_inputs(prediction_horizon: int, dt: int) -> None:
    """Validate prediction timing inputs, both expressed in milliseconds."""
    if prediction_horizon < 0:
        raise ValueError("prediction_horizon must be >= 0")
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if prediction_horizon % dt != 0:
        raise ValueError(
            f"prediction_horizon ({prediction_horizon}) must be divisible by dt ({dt})"
        )


def _predict_constant_acceleration_step(
    position: Vector2D,
    velocity: Vector2D,
    acceleration: Vector2D,
    time_s: float,
) -> tuple[Vector2D, Vector2D]:
    """
    Predict one step from t=0 and clamp braking objects exactly at their stop point.

    Constant acceleration kinematic equations:
        x(t)  = x0 + vx0*t + 0.5*ax*t^2
        y(t)  = y0 + vy0*t + 0.5*ay*t^2
        vx(t) = vx0 + ax*t
        vy(t) = vy0 + ay*t

    Stop-time derivation:
        The object decelerates when dot(v, a) < 0.
        Setting |v(t)| = 0 along the motion axis gives:
            t_stop = -dot(v0, a) / dot(a, a)
        If t_stop falls within [0, time_s], the object has stopped.
        Position is evaluated at t_stop and velocity is clamped to zero.
    """
    velocity_dot_accel = velocity.x * acceleration.x + velocity.y * acceleration.y
    acceleration_norm_sq = acceleration.x**2 + acceleration.y**2

    if velocity_dot_accel < 0.0 and acceleration_norm_sq > _ACCEL_NORM_SQ_MIN:
        stop_time_s = -velocity_dot_accel / acceleration_norm_sq
        if 0.0 <= stop_time_s <= time_s:
            stop_pos = Vector2D(
                x=position.x
                + velocity.x * stop_time_s
                + 0.5 * acceleration.x * stop_time_s**2,
                y=position.y
                + velocity.y * stop_time_s
                + 0.5 * acceleration.y * stop_time_s**2,
            )
            return stop_pos, Vector2D(x=0.0, y=0.0)

    new_pos = Vector2D(
        x=position.x + velocity.x * time_s + 0.5 * acceleration.x * time_s**2,
        y=position.y + velocity.y * time_s + 0.5 * acceleration.y * time_s**2,
    )
    new_velocity = Vector2D(
        x=velocity.x + acceleration.x * time_s,
        y=velocity.y + acceleration.y * time_s,
    )
    return new_pos, new_velocity


def _predict_ctrv_step(
    position: Vector2D,
    yaw: float,
    speed: float,
    yaw_rate: float,
    time_s: float,
) -> tuple[Vector2D, float]:
    """
    Predict one step using the Constant Turn Rate and Velocity model.

    When |yaw_rate| is almost zero, the model falls back to straight-line
    motion. Speed is held constant.
    """
    if abs(yaw_rate) >= _YAW_RATE_MIN:
        new_yaw = yaw + yaw_rate * time_s
        radius = speed / yaw_rate
        new_pos = Vector2D(
            x=position.x + radius * (math.sin(new_yaw) - math.sin(yaw)),
            y=position.y + radius * (math.cos(yaw) - math.cos(new_yaw)),
        )
    else:
        new_yaw = yaw
        new_pos = Vector2D(
            x=position.x + speed * math.cos(yaw) * time_s,
            y=position.y + speed * math.sin(yaw) * time_s,
        )

    return new_pos, new_yaw


def predict_motion_constant_velocity(
    objects: List[DynamicObjectStamped],
    prediction_horizon: int,
    dt: int,
    last_only: bool = False,
) -> List[DynamicObjectStamped]:
    """
    Predict object motion using a constant velocity (straight-line) model.

    Args:
        objects: Current states of the dynamic objects.
        prediction_horizon: Total prediction time [ms].
        dt: Time step between predictions [ms].
        last_only: If True, only the final predicted state per object is returned.
    """
    _validate_prediction_inputs(prediction_horizon, dt)
    predicted_objects: List[DynamicObjectStamped] = []

    for obj in objects:
        cs = obj.state
        velocity = (
            cs.velocity
            if isinstance(cs.velocity, Vector2D)
            else get_vector(float(cs.velocity), cs.yaw)
        )

        predicted_states = []
        for t in range(0, prediction_horizon + dt, dt):
            time_s = t / 1000.0
            new_pos = Vector2D(
                x=cs.pos.x + velocity.x * time_s,
                y=cs.pos.y + velocity.y * time_s,
            )
            predicted_states.append(
                DynamicObjectStamped(
                    timestamp=obj.timestamp + t,
                    state=DynamicObject(
                        id=cs.id,
                        obj_class=cs.obj_class,
                        pos=new_pos,
                        yaw=cs.yaw,
                        velocity=velocity,
                        acceleration=Vector2D(x=0.0, y=0.0),
                        width=cs.width,
                        length=cs.length,
                    ),
                )
            )

        if last_only:
            predicted_objects.append(predicted_states[-1])
        else:
            predicted_objects.extend(predicted_states)

    return predicted_objects


def predict_motion_constant_acceleration(
    objects: List[DynamicObjectStamped],
    prediction_horizon: int,
    dt: int,
    accelerations: Optional[Dict[int, Vector2D | float]] = None,
) -> List[DynamicObjectStamped]:
    """
    Predict object motion over the horizon using constant acceleration.

    If accelerations is provided, each object uses the acceleration from
    accelerations[object_id] instead of obj.state.acceleration.
    """
    _validate_prediction_inputs(prediction_horizon, dt)
    predicted_objects: List[DynamicObjectStamped] = []

    for obj in objects:
        cs = obj.state
        velocity = (
            cs.velocity
            if isinstance(cs.velocity, Vector2D)
            else get_vector(float(cs.velocity), cs.yaw)
        )
        acceleration_value = cs.acceleration
        if accelerations is not None:
            if cs.id not in accelerations:
                raise ValueError(f"Missing acceleration for object id {cs.id}")
            acceleration_value = accelerations[cs.id]

        acceleration = (
            acceleration_value
            if isinstance(acceleration_value, Vector2D)
            else get_vector(float(acceleration_value), cs.yaw)
        )

        for t in range(0, prediction_horizon + dt, dt):
            time_s = t / 1000.0
            new_pos, new_velocity = _predict_constant_acceleration_step(
                position=cs.pos,
                velocity=velocity,
                acceleration=acceleration,
                time_s=time_s,
            )
            predicted_objects.append(
                DynamicObjectStamped(
                    timestamp=obj.timestamp + t,
                    state=DynamicObject(
                        id=cs.id,
                        obj_class=cs.obj_class,
                        pos=new_pos,
                        yaw=cs.yaw,
                        velocity=new_velocity,
                        acceleration=acceleration,
                        width=cs.width,
                        length=cs.length,
                    ),
                )
            )

    return predicted_objects


def predict_constant_turn(
    objects: List[DynamicObjectStamped],
    prediction_horizon: int,
    dt: int,
    yaw_rates: Optional[Dict[int, float]] = None,
) -> List[DynamicObjectStamped]:
    """
    Predict object motion over the horizon using CTRV.

    yaw_rates maps each object id to a measured yaw rate in rad/s. If an object
    id is absent, yaw_rate = 0 is assumed, which degenerates to straight-line
    constant-velocity motion.
    """
    _validate_prediction_inputs(prediction_horizon, dt)

    if yaw_rates is None:
        yaw_rates = {}

    predicted_objects: List[DynamicObjectStamped] = []

    for obj in objects:
        cs = obj.state
        speed = (
            get_signed_magnitude(cs.velocity, cs.yaw)
            if isinstance(cs.velocity, Vector2D)
            else abs(float(cs.velocity))
        )
        yaw_rate = yaw_rates.get(cs.id, 0.0)

        for t in range(0, prediction_horizon + dt, dt):
            time_s = t / 1000.0
            new_pos, new_yaw = _predict_ctrv_step(
                position=cs.pos,
                yaw=cs.yaw,
                speed=speed,
                yaw_rate=yaw_rate,
                time_s=time_s,
            )
            new_velocity = get_vector(speed, new_yaw)
            predicted_objects.append(
                DynamicObjectStamped(
                    timestamp=obj.timestamp + t,
                    state=DynamicObject(
                        id=cs.id,
                        obj_class=cs.obj_class,
                        pos=new_pos,
                        yaw=new_yaw,
                        velocity=new_velocity,
                        acceleration=get_vector(0.0, new_yaw),
                        width=cs.width,
                        length=cs.length,
                    ),
                )
            )

    return predicted_objects


def group_predictions_by_object(
    predicted_objects: List[DynamicObjectStamped],
) -> Dict[int, List[DynamicObjectStamped]]:
    """Convert flat predictions into object_id -> predicted states."""
    grouped: Dict[int, List[DynamicObjectStamped]] = {}
    for obj in predicted_objects:
        grouped.setdefault(obj.state.id, []).append(obj)
    return grouped


def predict_environment(
    environment: Environment,
    prediction_horizon: int,
    dt: int,
    model: PredictionModel = "constant_acceleration",
    accelerations: Optional[Dict[int, Vector2D | float]] = None,
    yaw_rates: Optional[Dict[int, float]] = None,
) -> PredictedEnvironment:
    """
    Predict every object in the environment and return a PredictedEnvironment.

    model = "constant_acceleration" : straight-line, handles braking correctly.
    model = "constant_turn_rate"    : CTRV, handles curved trajectories.
    """
    if not environment.objects:
        raise ValueError("Empty object list - prediction would be vacuously safe.")

    if model == "constant_acceleration":
        raw = predict_motion_constant_acceleration(
            environment.objects,
            prediction_horizon,
            dt,
            accelerations=accelerations,
        )
    elif model == "constant_turn_rate":
        raw = predict_constant_turn(
            environment.objects,
            prediction_horizon,
            dt,
            yaw_rates=yaw_rates,
        )
    else:
        raise ValueError(f"Unknown prediction model: {model!r}")

    return PredictedEnvironment(
        objects=group_predictions_by_object(raw),
        lanes=environment.lanes,
        dt=dt,
        horizon=prediction_horizon,
    )
