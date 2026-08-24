from abc import ABC, abstractmethod

from core.types.geometry import Vector2D
from core.types.perception import PredictedEnvironment
from core.types.road import Lane


class CollisionChecker(ABC):
    """Contract for spatial-reasoning implementations (collision, clearance)."""

    @abstractmethod
    def build_road_polygons(self, lanes: list[Lane]) -> None:
        """Precompute drivable-area polygons from the static road model."""

    @abstractmethod
    def has_exited_lanes(self, x: float, y: float, yaw: float,
                         length: float, width: float) -> bool:
        """True when the ego footprint lies fully outside the drivable area."""

    @abstractmethod
    def min_distance_to_agents(self, ego_center_x: float, ego_center_y: float,
                               ego_yaw: float, length: float, width: float,
                               environment: PredictedEnvironment) -> float:
        """Minimum distance [m] between ego footprint and any agent state."""
