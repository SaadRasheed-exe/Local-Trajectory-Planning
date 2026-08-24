"""Sequential mode: plan -> follow -> replan in a single thread."""
import time
from typing import Optional

from omegaconf import DictConfig

from components.collision.collision_queries import build_lane_polygons
from simulation.simulate import Simulation
from visualization.visualizer import visualize_scene

from app.pipeline import (
    _build_goal_region,
    _compute_control_command,
    _export_search_tree,
    _get_target_speed,
    _log_object_distances,
    _pace_real_time,
    _prepare_planning,
    _stop_reason_for_state,
)


def run(sim: Simulation, planner, controller, cfg: DictConfig) -> None:
    previous_ego = sim.get_ego_state()
    start_timestamp = previous_ego.timestamp
    max_duration_ms = int(cfg.runtime.max_duration_ms)
    sim_step_ms = int(cfg.runtime.sim_step_ms)
    # Lanes are static scenario content; buffer the drivable area only once.
    lane_polygons = build_lane_polygons(sim.get_environment().lanes)

    try:
        while True:
            curr_env, ego_state, pred_env, goal_region, request = _prepare_planning(sim, cfg)

            plan_result = planner.plan(request)
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
                plan_result = planner.plan(request)

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
                stop_reason = _stop_reason_for_state(ego_state, env_frame, curr_env.lanes, lane_polygons, cfg)

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


