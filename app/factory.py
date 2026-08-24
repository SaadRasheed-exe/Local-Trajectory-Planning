"""Explicit component registry.

Selection is data-driven from config (cfg.*.name); adding a new planner or
controller means implementing the core interface and registering it here.
"""
from core.config_tools import convert_cfg_to_native
from core.interfaces.controller import Controller
from core.interfaces.planner import Planner

from components.controllers.mpc.controller import MPCController
from components.planners.hybrid_a_star import HybridAStarPlanner


PLANNERS = {
    HybridAStarPlanner.name: HybridAStarPlanner,
}

CONTROLLERS = {
    "mpc": MPCController,
}


def build_planner(cfg) -> Planner:
    name = getattr(cfg.planner, "name", None) or "hybrid_a_star"
    if name not in PLANNERS:
        raise ValueError(f"Unknown planner.name: {name!r} (available: {sorted(PLANNERS)})")
    return PLANNERS[name](convert_cfg_to_native(cfg), debug=bool(cfg.runtime.debug))


def build_controller(cfg) -> Controller:
    name = getattr(cfg.controller, "name", None) or "mpc"
    if name not in CONTROLLERS:
        raise ValueError(f"Unknown controller.name: {name!r} (available: {sorted(CONTROLLERS)})")
    return CONTROLLERS[name](cfg.vehicle, cfg.controller)
