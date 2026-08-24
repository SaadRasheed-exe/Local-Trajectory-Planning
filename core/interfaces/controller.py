from abc import ABC, abstractmethod
from typing import Optional

from core.types.control import ControlCommand
from core.types.planning import Trajectory
from core.types.vehicle import EgoStateStamped


class Controller(ABC):
    """Contract every controller implementation must satisfy.

    Constructed with native vehicle/controller configuration. A None
    trajectory means "no valid plan available"; the implementation applies
    its own fallback policy (typically braking) and reports it through
    ControlCommand.emergency_stop.
    """

    name: str

    @abstractmethod
    def compute_control(
        self,
        ego_state: EgoStateStamped,
        trajectory: Optional[Trajectory],
    ) -> ControlCommand:
        ...
