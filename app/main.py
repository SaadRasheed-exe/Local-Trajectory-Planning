"""Unified entry point for the autonomous driving stack.

This is a local trajectory planner: the goal region is relative to the ego
vehicle (always 'horizon seconds ahead' on its lane) and is re-anchored on
every planning loop.

Runtime behavior is controlled by configs/runtime/default_runtime_config.yaml:

    mode: sequential  - plan, follow the plan for plan_execution_ms, then replan
    mode: threaded    - real-time controller thread plus a planner running in
                        its own process (keeps the renderer responsive)

    debug: true       - planner KPIs, search-tree HTML export, object-distance logging

A run stops when the ego collides with another vehicle, when the ego reaches
the end of its lane, after runtime.max_duration_ms of simulated time (if > 0),
or on Ctrl-C. In sequential mode, a failed planning attempt pauses the
simulation until a trajectory is found.

Examples:
    python main.py runtime.mode=sequential
    python main.py runtime.mode=threaded runtime.debug=true
"""
from omegaconf import DictConfig

VALID_MODES = ("sequential", "threaded")


def run(cfg: DictConfig) -> None:
    """Assemble the stack from config and dispatch to the selected runner."""
    if cfg.runtime.mode not in VALID_MODES:
        raise ValueError(f"Unknown runtime.mode: {cfg.runtime.mode!r} (expected one of {VALID_MODES})")

    from app.factory import build_controller, build_planner
    from app.runners import sequential_loop, threaded_loop
    from simulation.simulate import Simulation

    sim = Simulation(cfg)
    planner = build_planner(cfg)
    controller = build_controller(cfg)

    if cfg.runtime.mode == "sequential":
        sequential_loop.run(sim, planner, controller, cfg)
    else:
        threaded_loop.run(sim, planner, controller, cfg)


__all__ = ["run", "VALID_MODES"]
