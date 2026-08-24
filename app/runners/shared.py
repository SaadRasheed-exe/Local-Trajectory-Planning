"""State shared between runtime threads."""
import threading
from typing import Optional

from core.types.planning import Trajectory


class SharedState:
    """Thread-safe handover of the latest trajectory from planner to controller."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.trajectory: Optional[Trajectory] = None
        self.is_running = True
        self.stop_reason: Optional[str] = None
