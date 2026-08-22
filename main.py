"""Unified entry point for the autonomous driving stack.

Runtime behavior is controlled by configs/runtime/default_runtime_config.yaml:

    mode: sequential  - plan, follow the plan for plan_execution_ms, then replan
    mode: threaded    - real-time controller thread plus continuous planner loop

    debug: true       - planner KPIs, search-tree HTML export, object-distance logging

Examples:
    python main.py runtime.mode=sequential
    python main.py runtime.mode=threaded runtime.debug=true
"""

import threading
import time
from typing import Optional, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from collision.collision import get_distance_to_objects, get_ego_lane_info
from controllers.controllers import MPCController
from models.models import EgoStateStamped, PlanningRequest, PredictedEnvironment, Trajectory
from planner.planner import plan
from prediction.predictivity import predict_environment
from simulation.simulate import Simulation
from utils.helper import get_goal_region
from visualization.visualizer import visualize_scene


FALLBACK_TARGET_SPEED = 10.0   # [m/s], used when ego is not matched to any lane
FALLBACK_GOAL_LENGTH = 3.0     # [m]
FALLBACK_GOAL_WIDTH = 3.0      # [m]

VALID_MODES = ("sequential", "threaded")


class SharedState:
    """Thread-safe handover of the latest trajectory from planner to controller."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.trajectory: Optional[Trajectory] = None
        self.is_running = True


# ---------------------------------------------------------------------------
# Planning pipeline (shared by both modes)
# ---------------------------------------------------------------------------

def _build_predicted_environment(curr_env, cfg: DictConfig) -> PredictedEnvironment:
    try:
        return predict_environment(
            environment=curr_env,
            prediction_horizon=cfg.planner.horizon,
            dt=cfg.planner.dt_sim,
        )
    except ValueError:
        # Scenarios without dynamic objects: nothing to predict.
        return PredictedEnvironment(
            objects={},
            lanes=curr_env.lanes,
            dt=cfg.planner.dt_sim,
            horizon=cfg.planner.horizon,
        )


def _get_target_speed(ego_state: EgoStateStamped, curr_env, cfg: DictConfig) -> float:
    _, _, _, _, _, lane_speed_limit = get_ego_lane_info(
        ego_state=ego_state.state,
        ego_length=cfg.vehicle.length,
        ego_width=cfg.vehicle.width,
        ego_rear_to_wheel=cfg.vehicle.rear_to_wheel,
        lanes=curr_env.lanes,
    )
    return lane_speed_limit if lane_speed_limit else FALLBACK_TARGET_SPEED


def _prepare_planning(sim: Simulation, cfg: DictConfig):
    """Fetch world state, build predictions and assemble a PlanningRequest."""
    curr_env = sim.get_environment()
    ego_state = sim.get_ego_state()
    pred_env = _build_predicted_environment(curr_env, cfg)

    target_speed = _get_target_speed(ego_state, curr_env, cfg)

    goal_cfg = OmegaConf.select(cfg, "scenario.goal")
    goal_region = get_goal_region(
        curr_ego_state=ego_state,
        lanes=curr_env.lanes,
        horizon=int(goal_cfg.horizon) if goal_cfg is not None else cfg.planner.horizon,
        length=float(goal_cfg.length) if goal_cfg is not None else FALLBACK_GOAL_LENGTH,
        width=float(goal_cfg.width) if goal_cfg is not None else FALLBACK_GOAL_WIDTH,
        target_speed=target_speed,
    )

    request = PlanningRequest(
        start_state=ego_state,
        goal_region=goal_region,
        target_speed=target_speed,
        environment=pred_env,
    )
    return curr_env, ego_state, pred_env, goal_region, request


# ---------------------------------------------------------------------------
# Control helpers
# ---------------------------------------------------------------------------

def _brake_command(cfg: DictConfig) -> Tuple[float, float]:
    # vehicle.max_deceleration is negative by convention.
    return float(cfg.vehicle.max_deceleration), 0.0


def _compute_control_command(
    controller: MPCController,
    ego_state: EgoStateStamped,
    trajectory: Optional[Trajectory],
    cfg: DictConfig,
) -> Tuple[float, float]:
    """Returns (acceleration, steer_rate); falls back to emergency braking."""
    if trajectory is None:
        print("No trajectory available. Emergency braking.")
        return _brake_command(cfg)
    try:
        acc, steer_rate = controller.compute_control(ego_state, trajectory)[0]
        return acc, steer_rate
    except Exception as exc:
        print(f"Controller failed ({exc}). Emergency braking.")
        return _brake_command(cfg)


def _pace_real_time(loop_start: float, step_ms: int) -> None:
    remaining_s = step_ms / 1000.0 - (time.perf_counter() - loop_start)
    if remaining_s > 0.0:
        time.sleep(remaining_s)
    else:
        print(f"Warning: loop overran its {step_ms} ms budget by {-remaining_s * 1000.0:.1f} ms")


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _log_object_distances(
    previous_ego: EgoStateStamped,
    current_ego: EgoStateStamped,
    pred_env: PredictedEnvironment,
    cfg: DictConfig,
) -> None:
    ff_cfg = cfg.cost.cost_objects_force_field
    distances, is_collision = get_distance_to_objects(
        current_ego=current_ego,
        previous_ego=previous_ego,
        predicted_env=pred_env,
        ego_length=cfg.vehicle.length,
        ego_width=cfg.vehicle.width,
        ego_rear_to_wheel=cfg.vehicle.rear_to_wheel,
        resolution_ms=int(ff_cfg.resolution_ms),
        calc_exact_distance=float(ff_cfg.d_ffl),
    )
    print(f"[debug] Object distances: {distances} | collision: {is_collision}")


def _export_search_tree(plan_result, pred_env, goal_region, cfg: DictConfig) -> None:
    # Lazy import: bokeh is only needed when debugging.
    from visualization.tree_visualizer import visualize_search_tree

    visualize_search_tree(
        plan_result=plan_result,
        pred_env=pred_env,
        goal_region=goal_region,
        cfg=cfg,
        vehicle_cfg=cfg.vehicle,
        output_filename="debug_tree.html",
    )


# ---------------------------------------------------------------------------
# Sequential mode: plan -> follow -> replan in a single thread
# ---------------------------------------------------------------------------

def _run_sequential(sim: Simulation, controller: MPCController, cfg: DictConfig) -> None:
    newest_trajectory: Optional[Trajectory] = None
    previous_ego = sim.get_ego_state()

    while True:
        curr_env, ego_state, pred_env, goal_region, request = _prepare_planning(sim, cfg)

        plan_result = plan(request, cfg, debug=bool(cfg.runtime.debug))
        if plan_result.success and plan_result.trajectory is not None:
            newest_trajectory = plan_result.trajectory
        else:
            print(f"No path found: {plan_result.status_message}. Following last trajectory.")

        if cfg.runtime.debug:
            _log_object_distances(previous_ego, ego_state, pred_env, cfg)
            _export_search_tree(plan_result, pred_env, goal_region, cfg)
        previous_ego = ego_state

        steps_per_plan = int(cfg.runtime.plan_execution_ms / cfg.runtime.sim_step_ms)
        for _ in range(steps_per_plan):
            loop_start = time.perf_counter()

            ego_state = sim.get_ego_state()
            acc, steer_rate = _compute_control_command(controller, ego_state, newest_trajectory, cfg)
            sim.step(acc, steer_rate, int(cfg.runtime.sim_step_ms))

            visualize_scene(
                env=sim.get_environment(),
                ego=ego_state,
                vehicle_params=cfg.vehicle,
                trajectory=newest_trajectory,
                goal_region=goal_region,
                path=sim.ego_history,
            )
            _pace_real_time(loop_start, int(cfg.runtime.sim_step_ms))

        if bool(plan_result.goal_region_reached):
            print("Goal reached.")
            break


# ---------------------------------------------------------------------------
# Threaded mode: real-time controller thread + continuous planner loop
# ---------------------------------------------------------------------------

def _controller_worker(sim: Simulation, controller: MPCController, shared_state: SharedState, cfg: DictConfig) -> None:
    dt_sec = cfg.runtime.sim_step_ms / 1000.0
    while shared_state.is_running:
        loop_start = time.perf_counter()

        ego_state = sim.get_ego_state()
        with shared_state.lock:
            trajectory = shared_state.trajectory

        acc, steer_rate = _compute_control_command(controller, ego_state, trajectory, cfg)
        sim.step(acc, steer_rate, int(cfg.runtime.sim_step_ms))

        remaining_s = dt_sec - (time.perf_counter() - loop_start)
        if remaining_s > 0.0:
            time.sleep(remaining_s)


def _run_threaded(sim: Simulation, controller: MPCController, cfg: DictConfig) -> None:
    shared_state = SharedState()
    ctrl_thread = threading.Thread(
        target=_controller_worker,
        args=(sim, controller, shared_state, cfg),
        daemon=True,
    )
    ctrl_thread.start()

    previous_ego = sim.get_ego_state()
    try:
        while True:
            curr_env, ego_state, pred_env, goal_region, request = _prepare_planning(sim, cfg)

            plan_result = plan(request, cfg, debug=bool(cfg.runtime.debug))
            if plan_result.success and plan_result.trajectory is not None:
                with shared_state.lock:
                    shared_state.trajectory = plan_result.trajectory
            else:
                # Keep following the last trajectory; the worker brakes once it runs out.
                print(f"No path found: {plan_result.status_message}. Following last trajectory.")

            if cfg.runtime.debug:
                _log_object_distances(previous_ego, ego_state, pred_env, cfg)
                _export_search_tree(plan_result, pred_env, goal_region, cfg)
            previous_ego = ego_state

            visualize_scene(
                env=curr_env,
                ego=ego_state,
                vehicle_params=cfg.vehicle,
                trajectory=plan_result.trajectory,
                goal_region=goal_region,
                path=sim.ego_history,
            )
    except KeyboardInterrupt:
        print("Shutting down simulation...")
    finally:
        shared_state.is_running = False
        ctrl_thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.runtime.mode not in VALID_MODES:
        raise ValueError(f"Unknown runtime.mode: {cfg.runtime.mode!r} (expected one of {VALID_MODES})")

    sim = Simulation(cfg)
    controller = MPCController(cfg.vehicle, cfg.controller)

    if cfg.runtime.mode == "sequential":
        _run_sequential(sim, controller, cfg)
    else:
        _run_threaded(sim, controller, cfg)


if __name__ == "__main__":
    main()
