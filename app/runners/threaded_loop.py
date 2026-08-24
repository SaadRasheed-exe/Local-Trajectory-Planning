"""Threaded mode: real-time controller thread + continuous planner process."""
import multiprocessing
import threading
import time

from omegaconf import DictConfig

from components.collision.collision_queries import build_lane_polygons
from simulation.simulate import Simulation
from visualization.visualizer import visualize_scene

from app.pipeline import (
    _build_goal_region,
    _build_planning_pipeline,
    _compute_control_command,
    _export_search_tree,
    _get_target_speed,
    _log_object_distances,
    _pace_real_time,
    _prepare_planning,
    _stop_reason_for_state,
)
from app.runners.shared import SharedState


def _controller_worker(
    sim: Simulation,
    controller,
    shared_state: SharedState,
    cfg: DictConfig,
    lanes,
    lane_polygons,
) -> None:
    dt_sec = cfg.runtime.sim_step_ms / 1000.0
    while shared_state.is_running:
        loop_start = time.perf_counter()

        ego_state = sim.get_ego_state()
        with shared_state.lock:
            trajectory = shared_state.trajectory

        acc, steer_rate = _compute_control_command(controller, ego_state, trajectory, cfg)
        sim.step(acc, steer_rate, int(cfg.runtime.sim_step_ms))

        stop_reason = _stop_reason_for_state(ego_state, sim.get_environment(), lanes, lane_polygons, cfg)
        if stop_reason is not None:
            with shared_state.lock:
                shared_state.stop_reason = stop_reason
            shared_state.is_running = False
            break

        remaining_s = dt_sec - (time.perf_counter() - loop_start)
        if remaining_s > 0.0:
            time.sleep(remaining_s)


def _planner_process(pipe, planner, cfg: DictConfig) -> None:
    """
    Dedicated planning process for threaded mode.

    Runs in its own interpreter so its GIL cannot starve the renderer thread.
    Protocol over the duplex pipe:

        -> {'type': 'req'}                       ask the renderer for a snapshot
        <- {'type': 'snapshot', 'ego': ..., 'env': ...}
        -> {'type': 'traj', 'success', 'status_message', 'trajectory'}
        <- {'type': 'stop'}                      shutdown request
    """
    period_s = max(int(cfg.runtime.get("planner_period_ms", 50)), 0) / 1000.0
    previous_ego = None
    while True:
        try:
            pipe.send({"type": "req"})
            msg = pipe.recv()
        except (EOFError, BrokenPipeError, OSError, KeyboardInterrupt):
            return

        if not isinstance(msg, dict) or msg.get("type") != "snapshot":
            return  # 'stop' or dead peer

        ego_state = msg["ego"]
        curr_env = msg["env"]

        pred_env, goal_region, request = _build_planning_pipeline(curr_env, ego_state, cfg)
        plan_result = planner.plan(request)

        try:
            pipe.send({
                "type": "traj",
                "success": bool(plan_result.success),
                "status_message": plan_result.status_message,
                "trajectory": plan_result.trajectory if plan_result.success else None,
            })
        except (BrokenPipeError, OSError):
            return

        if cfg.runtime.debug and previous_ego is not None:
            _log_object_distances(previous_ego, ego_state, pred_env, cfg)
            _export_search_tree(plan_result, pred_env, goal_region, cfg)
        previous_ego = ego_state

        if period_s > 0:
            time.sleep(period_s)


def run(sim: Simulation, planner, controller, cfg: DictConfig) -> None:
    shared_state = SharedState()

    # Seed the first trajectory synchronously, before any thread starts.
    # Otherwise the worker emergency-brakes while the first (slow) plan runs,
    # the ego comes to a standstill, and the planner can no longer expand
    # motion primitives from v = 0 - a permanent deadlock.
    curr_env, ego_state, pred_env, goal_region, request = _prepare_planning(sim, cfg)
    seed_result = planner.plan(request)
    if seed_result.success and seed_result.trajectory is not None:
        with shared_state.lock:
            shared_state.trajectory = seed_result.trajectory
    else:
        print(f"Initial plan failed: {seed_result.status_message}. Starting without trajectory.")

    lanes = curr_env.lanes
    # Lanes are static scenario content; buffer the drivable area only once.
    lane_polygons = build_lane_polygons(lanes)

    # Planner runs in a separate process (own GIL): a Python-heavy planning
    # loop inside this interpreter starves the renderer ~70x through GIL
    # handoff. Forked before the controller thread starts (fork-safety).
    parent_conn, child_conn = multiprocessing.get_context("fork").Pipe(duplex=True)
    planner_proc = multiprocessing.get_context("fork").Process(
        target=_planner_process,
        args=(child_conn, planner, cfg),
        daemon=True,
    )
    planner_proc.start()
    child_conn.close()  # this process only talks over its own end

    ctrl_thread = threading.Thread(
        target=_controller_worker,
        args=(sim, controller, shared_state, cfg, lanes, lane_polygons),
        daemon=True,
    )
    ctrl_thread.start()

    start_timestamp = sim.get_ego_state().timestamp
    max_duration_ms = int(cfg.runtime.max_duration_ms)
    sim_step_ms = int(cfg.runtime.sim_step_ms)
    latest_trajectory = shared_state.trajectory  # seed result, for display
    planner_alive = True

    # Main thread: dedicated render loop at sim-tick cadence. Drawing stays on
    # the main thread (matplotlib is not thread-safe) while the controller
    # thread steps physics and the planner process computes trajectories.
    try:
        while True:
            if shared_state.stop_reason is not None:
                print(f"{shared_state.stop_reason} Shutting down.")
                break

            loop_start = time.perf_counter()

            ego_state = sim.get_ego_state()
            env_frame = sim.get_environment()

            # Serve planner requests / collect results without blocking.
            if planner_alive:
                try:
                    while parent_conn.poll():
                        msg = parent_conn.recv()
                        if msg["type"] == "req":
                            parent_conn.send({
                                "type": "snapshot",
                                "ego": ego_state,
                                "env": env_frame,
                            })
                        elif msg["type"] == "traj":
                            if msg["success"] and msg["trajectory"] is not None:
                                latest_trajectory = msg["trajectory"]
                                with shared_state.lock:
                                    shared_state.trajectory = msg["trajectory"]
                            else:
                                print(f"No path found: {msg['status_message']}. Following last trajectory.")
                except (EOFError, BrokenPipeError, OSError):
                    print("Planner process died; continuing on the last trajectory.")
                    planner_alive = False

            with shared_state.lock:
                trajectory = shared_state.trajectory

            visualize_scene(
                env=env_frame,
                ego=ego_state,
                vehicle_params=cfg.vehicle,
                trajectory=trajectory if trajectory is not None else latest_trajectory,
                goal_region=_build_goal_region(
                    lanes,
                    ego_state,
                    cfg,
                    _get_target_speed(ego_state, env_frame, cfg),
                ),
                path=sim.ego_history,
            )

            if max_duration_ms > 0 and (ego_state.timestamp - start_timestamp) >= max_duration_ms:
                print(f"Simulation duration limit reached ({max_duration_ms} ms). Shutting down.")
                break

            _pace_real_time(loop_start, sim_step_ms)
    except KeyboardInterrupt:
        print("Shutting down simulation...")
    finally:
        shared_state.is_running = False
        try:
            parent_conn.send({"type": "stop"})
        except (BrokenPipeError, OSError):
            pass
        planner_proc.join(timeout=1.5)
        if planner_proc.is_alive():
            planner_proc.terminate()
            planner_proc.join(timeout=1.0)
        ctrl_thread.join(timeout=2.0)


