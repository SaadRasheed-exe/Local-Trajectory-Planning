from abc import ABC, abstractmethod

from core.types.planning import PlanResult, PlanningRequest


class Planner(ABC):
    """Contract every trajectory planner implementation must satisfy.

    Implementations are constructed with their fully converted native
    configuration (no OmegaConf objects) and must be pickle-safe so they can
    run in a separate process.
    """

    name: str

    @abstractmethod
    def plan(self, request: PlanningRequest) -> PlanResult:
        """Compute a trajectory for the given request.

        Never raises for ordinary failures; communicates them through
        PlanResult.success / status_message so the runner can apply uniform
        fallback policy (e.g. emergency braking) regardless of implementation.
        """
