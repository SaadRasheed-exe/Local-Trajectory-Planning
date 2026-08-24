from abc import ABC, abstractmethod

from core.types.perception import PredictedEnvironment
from core.types.road import Environment


class Predictor(ABC):
    """Contract for motion-prediction implementations."""

    name: str

    @abstractmethod
    def predict(self, frame: Environment, horizon_ms: int, dt_ms: int) -> PredictedEnvironment:
        """Extrapolate current observations into future states."""
