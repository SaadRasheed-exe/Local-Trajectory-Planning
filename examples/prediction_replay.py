"""Standalone visualization: lead-vehicle prediction for the overtake scenario.

Predicts only the lead vehicle ahead of ego and plots the prediction window:

    python examples/prediction_replay.py [output_path]
"""

import math
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from models.models import (
    DynamicObjectStamped,
    Environment,
    EgoStateStamped,
    PredictedEnvironment,
)
from omegaconf import DictConfig
from utils.helper import global_to_ego_axis

from prediction.predictivity import PredictionModel, predict_environment

# Prediction duration used by the visualization.
_PLOT_PREDICTION_HORIZON_MS: int = 7500

# Time step used by the visualization prediction.
_PLOT_PREDICTION_DT_MS: int = 100

# Visible global x window for the prediction plot.
_PLOT_FORWARD_WINDOW_M: float = 250.0


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
    accelerations=None,
    yaw_rates=None,
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

    config_dir = REPO_ROOT / "configs"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(
            config_name="config",
            overrides=list(config_overrides or []),
        )


def _plot_prediction_dt_ms(planner_dt_ms: int) -> int:
    """Use a finer plot prediction step without changing planner behavior."""
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


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "prediction_visualization.png"
    visualize_overtake_decision_replay(output_path=output, show=False)
