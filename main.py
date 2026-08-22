"""Unified entry point for the autonomous driving stack.

This is a local trajectory planner: the goal region is relative to the ego
vehicle (always 'horizon seconds ahead' on its lane) and is re-anchored on
every planning loop.

Runtime behavior is controlled by configs/runtime/default_runtime_config.yaml:

    mode: sequential  - plan, follow the plan for plan_execution_ms, then replan
    mode: threaded    - real-time controller thread plus continuous planner loop

    debug: true       - planner KPIs, search-tree HTML export, object-distance logging

A run stops when the ego collides with another vehicle, when the ego reaches
the end of its lane, after runtime.max_duration_ms of simulated time (if > 0),
or on Ctrl-C. In sequential mode, a failed planning attempt pauses the
simulation until a trajectory is found.

Examples:
    python main.py runtime.mode=sequential
    python main.py runtime.mode=threaded runtime.debug=true
"""

import threading
import time
from typing import Optional, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from collision.collision import (
    get_colliding_object_ids,
    get_distance_to_objects,
    get_ego_lane_info,
)
from controllers.controllers import MPCController
from models.models import EgoStateStamped, Environment, PlanningRequest, PredictedEnvironment, Trajectory
from planner.planner import plan
from prediction.predictivity import predict_environment
from simulation.simulate import Simulation
from utils.helper import get_goal_region, get_nearest_lane_end_distance
from visualization.visualizer import visualize_scene


FALLBACK_TARGET_SPEED = 10.0   # [m/s], used when ego is not matched to any lane
FALLBACK_GOAL_LENGTH = 5.0     # [m]
FALLBACK_GOAL_WIDTH = 5.0      # [m]

# The run stops once the ego is this close to its lane's last centerline point.
LANE_END_STOP_MARGIN_M = 5.0

VALID_MODES = ("sequential", "threaded")


class SharedState:
    """Thread-safe handover of the latest trajectory from planner to controller."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.trajectory: Optional[Trajectory] = None
        self.is_running = True
        self.stop_reason: Optional[str] = None


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


def _build_goal_region(
    lanes,
    ego_state: EgoStateStamped,
    cfg: DictConfig,
    target_speed: float,
):
    """
    Anchor the rolling goal region relative to the ego pose.

    This is a local planner: the goal is always 'horizon seconds ahead on the
    current lane' (see utils.helper.get_goal_region), never a fixed map
    coordinate. Called once per planning loop for the actual planning target
    and once per animation frame for the live display.
    """
    goal_cfg = OmegaConf.select(cfg, "scenario.goal")
    return get_goal_region(
        curr_ego_state=ego_state,
        lanes=lanes,
        horizon=int(goal_cfg.horizon) if goal_cfg is not None else cfg.planner.horizon,
        length=float(goal_cfg.length) if goal_cfg is not None else FALLBACK_GOAL_LENGTH,
        width=float(goal_cfg.width) if goal_cfg is not None else FALLBACK_GOAL_WIDTH,
        target_speed=target_speed,
    )


def _prepare_planning(sim: Simulation, cfg: DictConfig):
    """Fetch world state, build predictions and assemble a PlanningRequest."""
    curr_env = sim.get_environment()
    ego_state = sim.get_ego_state()
    pred_env = _build_predicted_environment(curr_env, cfg)

    target_speed = _get_target_speed(ego_state, curr_env, cfg)
    goal_region = _build_goal_region(curr_env.lanes, ego_state, cfg, target_speed)

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


def _stop_reason_for_state(
    ego_state: EgoStateStamped,
    environment: Environment,
    lanes,
    cfg: DictConfig,
) -> Optional[str]:
    """
    Returns a shutdown reason if the run must stop now, else None.

    Stop conditions: the ego bounding box intersects another vehicle, or the
    ego has reached the end of its lane. Evaluated once per simulation step.
    """
    colliding_ids = get_colliding_object_ids(
        ego_state=ego_state,
        environment=environment,
        ego_length=float(cfg.vehicle.length),
        ego_width=float(cfg.vehicle.width),
        ego_rear_to_wheel=float(cfg.vehicle.rear_to_wheel),
    )
    if colliding_ids:
        return f"Collision with object(s) {colliding_ids}."

    if get_nearest_lane_end_distance(ego_state, lanes) <= LANE_END_STOP_MARGIN_M:
        return "Lane end reached."

    return None


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
    previous_ego = sim.get_ego_state()
    start_timestamp = previous_ego.timestamp
    max_duration_ms = int(cfg.runtime.max_duration_ms)
    sim_step_ms = int(cfg.runtime.sim_step_ms)

    try:
        while True:
            curr_env, ego_state, pred_env, goal_region, request = _prepare_planning(sim, cfg)

            plan_result = plan(request, cfg, debug=bool(cfg.runtime.debug))
            if not (plan_result.success and plan_result.trajectory is not None):
                print(f"No path found: {plan_result.status_message}. "
                      f"Simulation paused until a trajectory is found.")

            # Sequential mode contract: the simulation never advances without a
            # valid plan. Freeze everything (objects included) and retry planning
            # until success; prediction noise is redrawn per attempt.
            while not (plan_result.success and plan_result.trajectory is not None):
                ego_state = sim.get_ego_state()
                visualize_scene(
                    env=sim.get_environment(),
                    ego=ego_state,
                    vehicle_params=cfg.vehicle,
                    trajectory=None,
                    goal_region=_build_goal_region(
                        curr_env.lanes,
                        ego_state,
                        cfg,
                        _get_target_speed(ego_state, curr_env, cfg),
                    ),
                    path=sim.ego_history,
                )
                time.sleep(0.05)

                curr_env, ego_state, pred_env, goal_region, request = _prepare_planning(sim, cfg)
                plan_result = plan(request, cfg, debug=bool(cfg.runtime.debug))

            newest_trajectory = plan_result.trajectory

            if cfg.runtime.debug:
                _log_object_distances(previous_ego, ego_state, pred_env, cfg)
                _export_search_tree(plan_result, pred_env, goal_region, cfg)
            previous_ego = ego_state

            steps_per_plan = int(cfg.runtime.plan_execution_ms / sim_step_ms)
            # Never execute beyond the end of the current plan.
            traj_span_steps = int(
                (newest_trajectory.states[-1].timestamp - newest_trajectory.states[0].timestamp)
                / sim_step_ms
            )
            steps_per_plan = max(1, min(steps_per_plan, traj_span_steps))

            stop_reason: Optional[str] = None
            for _ in range(steps_per_plan):
                loop_start = time.perf_counter()

                ego_state = sim.get_ego_state()
                acc, steer_rate = _compute_control_command(controller, ego_state, newest_trajectory, cfg)
                sim.step(acc, steer_rate, sim_step_ms)

                env_frame = sim.get_environment()
                stop_reason = _stop_reason_for_state(ego_state, env_frame, curr_env.lanes, cfg)

                # Display goal: re-anchored to the current pose every frame so the
                # box slides ahead of the vehicle. The planning target stays fixed
                # for the whole execution window (request.goal_region).
                visualize_scene(
                    env=env_frame,
                    ego=ego_state,
                    vehicle_params=cfg.vehicle,
                    trajectory=newest_trajectory,
                    goal_region=_build_goal_region(
                        curr_env.lanes,
                        ego_state,
                        cfg,
                        _get_target_speed(ego_state, curr_env, cfg),
                    ),
                    path=sim.ego_history,
                )
                _pace_real_time(loop_start, int(cfg.runtime.sim_step_ms))

                if stop_reason is not None:
                    break

            if stop_reason is not None:
                print(f"{stop_reason} Shutting down.")
                break

            # Re-fetch so the check sees the state after the final step,
            # not the pre-step snapshot from the last iteration.
            ego_state = sim.get_ego_state()

            # This is a local planner with a rolling goal: there is no final
            # destination, so the run continues until explicitly stopped.
            if max_duration_ms > 0 and (ego_state.timestamp - start_timestamp) >= max_duration_ms:
                print(f"Simulation duration limit reached ({max_duration_ms} ms). Shutting down.")
                break
    except KeyboardInterrupt:
        print("Shutting down simulation...")


# ---------------------------------------------------------------------------
# Threaded mode: real-time controller thread + continuous planner loop
# ---------------------------------------------------------------------------

def _controller_worker(
    sim: Simulation,
    controller: MPCController,
    shared_state: SharedState,
    cfg: DictConfig,
    lanes,
) -> None:
    dt_sec = cfg.runtime.sim_step_ms / 1000.0
    while shared_state.is_running:
        loop_start = time.perf_counter()

        ego_state = sim.get_ego_state()
        with shared_state.lock:
            trajectory = shared_state.trajectory

        acc, steer_rate = _compute_control_command(controller, ego_state, trajectory, cfg)
        sim.step(acc, steer_rate, int(cfg.runtime.sim_step_ms))

        stop_reason = _stop_reason_for_state(ego_state, sim.get_environment(), lanes, cfg)
        if stop_reason is not None:
            with shared_state.lock:
                shared_state.stop_reason = stop_reason
            shared_state.is_running = False
            break

        remaining_s = dt_sec - (time.perf_counter() - loop_start)
        if remaining_s > 0.0:
            time.sleep(remaining_s)


def _run_threaded(sim: Simulation, controller: MPCController, cfg: DictConfig) -> None:
    shared_state = SharedState()

    # Seed the first trajectory synchronously, before the controller starts.
    # Otherwise the worker emergency-brakes while the first (slow) plan runs,
    # the ego comes to a standstill, and the planner can no longer expand
    # motion primitives from v = 0 - a permanent deadlock.
    curr_env, ego_state, pred_env, goal_region, request = _prepare_planning(sim, cfg)
    seed_result = plan(request, cfg, debug=bool(cfg.runtime.debug))
    if seed_result.success and seed_result.trajectory is not None:
        with shared_state.lock:
            shared_state.trajectory = seed_result.trajectory
    else:
        print(f"Initial plan failed: {seed_result.status_message}. Starting without trajectory.")

    ctrl_thread = threading.Thread(
        target=_controller_worker,
        args=(sim, controller, shared_state, cfg, curr_env.lanes),
        daemon=True,
    )
    ctrl_thread.start()

    previous_ego = sim.get_ego_state()
    start_timestamp = previous_ego.timestamp
    max_duration_ms = int(cfg.runtime.max_duration_ms)
    try:
        while True:
            if shared_state.stop_reason is not None:
                print(f"{shared_state.stop_reason} Shutting down.")
                break

            curr_env, ego_state, pred_env, goal_region, request = _prepare_planning(sim, cfg)

            plan_result = plan(request, cfg, debug=bool(cfg.runtime.debug))
            if plan_result.success and plan_result.trajectory is not None:
                with shared_state.lock:
                    shared_state.trajectory = plan_result.trajectory
            else:
                # Keep following the last trajectory; the worker brakes once it runs out.
                print(f"No path found: {plan_result.status_message}. Following last trajectory.")

            if max_duration_ms > 0 and (ego_state.timestamp - start_timestamp) >= max_duration_ms:
                print(f"Simulation duration limit reached ({max_duration_ms} ms). Shutting down.")
                break

            if cfg.runtime.debug:
                _log_object_distances(previous_ego, ego_state, pred_env, cfg)
                _export_search_tree(plan_result, pred_env, goal_region, cfg)
            previous_ego = ego_state

            # Display goal: re-anchored to the freshest pose so the box slides
            # ahead of the vehicle between planning iterations.
            display_ego = sim.get_ego_state()
            visualize_scene(
                env=curr_env,
                ego=ego_state,
                vehicle_params=cfg.vehicle,
                trajectory=plan_result.trajectory,
                goal_region=_build_goal_region(
                    curr_env.lanes,
                    display_ego,
                    cfg,
                    _get_target_speed(display_ego, curr_env, cfg),
                ),
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
