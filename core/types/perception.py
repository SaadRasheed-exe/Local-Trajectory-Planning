from dataclasses import dataclass
from typing import Dict, List

from core.types.agents import DynamicObjectStamped
from core.types.road import Lane


@dataclass
class PredictedEnvironment:
    objects: Dict[int, List[DynamicObjectStamped]]
    # mapping: object_id → predicted states over time

    lanes: List[Lane]
    # static road geometry

    dt: int
    # time resolution of prediction [ms]

    horizon: int
    # prediction horizon [ms]
