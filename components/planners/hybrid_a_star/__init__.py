"""Hybrid A* planner: search internals + public Planner implementation."""
from core.interfaces.planner import Planner
from core.types.planning import PlanResult, PlanningRequest

from components.planners.hybrid_a_star import search


class HybridAStarPlanner(Planner):
    """Wraps the functional Hybrid A* search behind the Planner interface."""

    name = "hybrid_a_star"

    def __init__(self, cfg, debug: bool = False) -> None:
        self._cfg = cfg          # native config (converted by the factory)
        self._debug = debug

    def plan(self, request: PlanningRequest) -> PlanResult:
        return search.plan(request, self._cfg, debug=self._debug)
