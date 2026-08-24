"""Contract tests for Controller implementations (plain python, no pytest).

Run:  python tests/contracts/test_controller_contract.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math

from omegaconf import OmegaConf

from components.controllers.mpc.controller import MPCController
from core.types.control import ControlCommand
from core.types.planning import Trajectory
from core.types.vehicle import EgoState, EgoStateStamped
from core.types.geometry import Vector2D


def _ego(x: float, v: float, yaw: float = 0.0, t_ms: int = 0) -> EgoStateStamped:
    return EgoStateStamped(
        timestamp=t_ms,
        state=EgoState(
            pos=Vector2D(x=x, y=2.0),
            velocity=Vector2D(x=v * math.cos(yaw), y=v * math.sin(yaw)),
            acceleration=Vector2D(x=0.0, y=0.0),
            yaw=yaw,
            steering_angle=0.0,
        ),
    )


def _straight_trajectory(v: float, horizon_s: float = 3.0, dt_s: float = 0.1) -> Trajectory:
    states = []
    for i in range(int(horizon_s / dt_s) + 1):
        states.append(_ego(x=v * i * dt_s + 10.0, v=v, t_ms=int(i * dt_s * 1000)))
    return Trajectory(states=states)


def main() -> None:
    vehicle_cfg = OmegaConf.load("configs/vehicle/renault_twizy.yaml")
    controller_cfg = OmegaConf.load("configs/controller/mpc.yaml")
    controller = MPCController(vehicle_cfg, controller_cfg)

    ego = _ego(10.0, 12.0)
    traj = _straight_trajectory(12.0)

    # 1. valid tracking returns a well-formed ControlCommand within actuator bounds
    cmd = controller.compute_control(ego, traj)
    assert isinstance(cmd, ControlCommand)
    assert math.isfinite(cmd.acceleration) and math.isfinite(cmd.steer_rate)
    assert -abs(vehicle_cfg.max_deceleration) - 1e-6 <= cmd.acceleration <= vehicle_cfg.max_acceleration + 1e-6
    assert abs(cmd.steer_rate) <= vehicle_cfg.max_steer_rate + 1e-6
    assert not cmd.emergency_stop

    # 2. missing trajectory must degrade to an emergency stop, never raise
    stop = controller.compute_control(ego, None)
    assert isinstance(stop, ControlCommand)
    assert stop.emergency_stop is True
    assert stop.acceleration <= 0.0

    print("controller contract OK")


if __name__ == "__main__":
    main()
