import math
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.models import (
    DynamicObject,
    DynamicObjectStamped,
    EgoStateStamped,
    Environment,
    PredictedEnvironment,
    Vector2D,
)
from utils.helper import (
    get_signed_magnitude,
    get_vector,
    global_to_ego_axis,
)

PredictionModel = Literal["constant_acceleration", "constant_turn_rate"]

# Guard for near-zero yaw rate in CTRV: below this, use straight-line fallback.
_YAW_RATE_MIN: float = 1e-4  # [rad/s]

# Guard for near-zero acceleration norm in braking clamp.
_ACCEL_NORM_SQ_MIN: float = 1e-9

# Prediction duration used by the visual ization.
_PLOT_PREDICTION_HORIZON_MS: int = 7500

# Time step used by the visualization prediction.
_PLOT_PREDICTION_DT_MS: int = 100

# Visible global x window for the prediction plot.
_PLOT_FORWARD_WINDOW_M: float = 250.0


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


def find_lead_vehicle(
    environment: Environment,
    ego_state: EgoStateStamped,
    lane_y_tolerance: float = 2.0,
    max_lookahead_m: float = 150.0,
) -> Optional[DynamicObjectStamped]:
    """Find the closest object ahead of ego in the same lane."""
    candidates = []
    for obj in environment.objects:
        rel_x, rel_y, _ = global_to_ego_axis(
            obj.state.pos.x,
            obj.state.pos.y,
            ego_state.state.pos.x,
            ego_state.state.pos.y,
            ego_state.state.yaw,
        )
        if 0.0 < rel_x <= max_lookahead_m and abs(rel_y) <= lane_y_tolerance:
            candidates.append((rel_x, obj))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def predict_lead_environment(
    environment: Environment,
    ego_state: EgoStateStamped,
    prediction_horizon: int,
    dt: int,
    model: PredictionModel = "constant_acceleration",
    accelerations: Optional[Dict[int, Vector2D | float]] = None,
    yaw_rates: Optional[Dict[int, float]] = None,
) -> PredictedEnvironment:
    """Predict only the lead vehicle relevant to ego."""
    lead = find_lead_vehicle(environment, ego_state)
    if lead is None:
        raise ValueError("No lead vehicle found.")
    return predict_environment(
        Environment(objects=[lead], lanes=environment.lanes),
        prediction_horizon=prediction_horizon,
        dt=dt,
        model=model,
        accelerations=accelerations,
        yaw_rates=yaw_rates,
    )


def _load_prediction_visualization_cfg(config_overrides: Optional[List[str]] = None):
    """Load the project config exactly like the simulation stack does."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(
            config_name="config",
            overrides=list(config_overrides or []),
        )


def _plot_prediction_dt_ms(planner_dt_ms: int) -> int:
    """
    Use a finer plot prediction step without changing planner behavior.

    The project planner can run at 1000 ms, but this plot needs denser samples
    to show the prediction window clearly.
    """
    return min(planner_dt_ms, _PLOT_PREDICTION_DT_MS)


def visualize_overtake_decision_replay(
    output_path: str = "prediction_visualization.png",
    show: bool = True,
    config_overrides: Optional[List[str]] = None,
) -> None:
    """
    Plot the lead vehicle prediction from the current scenario.

    Only the lead vehicle is predicted here. Other scenario objects are
    intentionally ignored in this visualization.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from simulation.simulate import Simulation

    cfg = _load_prediction_visualization_cfg(
        config_overrides or ["scenario=overtake_scenario"]
    )
    simulation = Simulation(cfg)
    initial_environment = simulation.get_environment(noise_level=0.0)
    initial_ego = simulation.get_ego_state(noise_level=0.0)

    dt = _plot_prediction_dt_ms(int(cfg.planner.dt_sim))
    prediction_horizon = int(math.ceil(_PLOT_PREDICTION_HORIZON_MS / dt) * dt)

    lead = find_lead_vehicle(initial_environment, initial_ego)
    if lead is None:
        raise ValueError("Cannot visualize lead prediction: no lead vehicle found.")
    lead_id = lead.state.id

    predicted_environment = predict_lead_environment(
        environment=initial_environment,
        ego_state=initial_ego,
        prediction_horizon=prediction_horizon,
        dt=dt,
        model="constant_acceleration",
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for lane in initial_environment.lanes:
        if lane.centerline:
            lane_xs = [point.x for point, _ in lane.centerline]
            lane_ys = [point.y for point, _ in lane.centerline]
            ax.plot(
                lane_xs,
                lane_ys,
                color="#b8b8b8",
                linestyle="--",
                linewidth=1.0,
                label=f"Lane {lane.id}",
                zorder=1,
            )

    color_norm = colors.Normalize(
        vmin=0.0,
        vmax=float(_PLOT_PREDICTION_HORIZON_MS),
    )
    prediction_scatter = None
    for object_id, predictions in sorted(predicted_environment.objects.items()):
        predictions = sorted(predictions, key=lambda obj: obj.timestamp)
        elapsed_from_now = np.array(
            [obj.timestamp - initial_ego.timestamp for obj in predictions]
        )
        window_mask = elapsed_from_now <= _PLOT_PREDICTION_HORIZON_MS
        predictions = [obj for obj, keep in zip(predictions, window_mask) if keep]
        elapsed = elapsed_from_now[window_mask]
        if not predictions:
            continue
        xs = np.array([obj.state.pos.x for obj in predictions])
        ys = np.array([obj.state.pos.y for obj in predictions])

        if object_id == lead_id:
            label = "Lead predicted from environment"
            marker = "s"
            line_color = "tab:purple"
        else:
            label = f"Object {object_id} predicted from environment"
            marker = "o"
            line_color = "tab:gray"

        ax.plot(xs, ys, color=line_color, linewidth=1.5, alpha=0.45, zorder=2)
        prediction_scatter = ax.scatter(
            xs,
            ys,
            c=elapsed,
            cmap=plt.cm.viridis,
            norm=color_norm,
            marker=marker,
            s=42,
            edgecolors="black",
            linewidths=0.3,
            label=label,
            zorder=3,
        )

    ax.scatter(
        [initial_ego.state.pos.x],
        [initial_ego.state.pos.y],
        color="black",
        s=70,
        label="Current ego",
        zorder=5,
    )
    if prediction_scatter is not None:
        color_bar = fig.colorbar(prediction_scatter, ax=ax, pad=0.01)
        color_bar.set_label("Prediction time [ms]")

    ax.set_xlabel("global x [m]")
    ax.set_ylabel("global y [m]")
    ax.set_xlim(
        initial_ego.state.pos.x,
        initial_ego.state.pos.x + _PLOT_FORWARD_WINDOW_M,
    )
    ax.set_title("Predicted lead vehicle")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_prediction(
    output_path: str = "prediction_visualization.png",
    show: bool = True,
    config_overrides: Optional[List[str]] = None,
) -> None:
    """Backward-compatible entry point for the current-scenario prediction plot."""
    visualize_overtake_decision_replay(
        output_path=output_path,
        show=show,
        config_overrides=config_overrides,
    )


if __name__ == "__main__":
    visualize_overtake_decision_replay(config_overrides=["scenario=overtake_scenario"])
