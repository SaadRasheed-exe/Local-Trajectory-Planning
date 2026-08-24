"""Contract tests for Planner implementations (plain python, no pytest).

Run:  python tests/contracts/test_planner_contract.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math
import pickle

from hydra import compose, initialize

from app.factory import build_planner
from app.pipeline import get_goal_region, get_nearest_lane_end_distance
from core.config_tools import convert_cfg_to_native
from core.types import Lane, PlanningRequest, PredictedEnvironment
from core.types.geometry import Vector2D
from core.types.vehicle import EgoState, EgoStateStamped


def _straight_lane(y: float = 2.0, length_m: float = 300.0, step_m: float = 2.0) -> Lane:
    n = int(length_m / step_m) + 1
    return Lane(
        id="lane_1",
        centerline=[(Vector2D(x=i * step_m, y=y), 0.0) for i in range(n)],
        width=4.0,
        speed_limit=20.0,
    )


def _ego(x: float, y: float, v: float, yaw: float = 0.0, t_ms: int = 0) -> EgoStateStamped:
    return EgoStateStamped(
        timestamp=t_ms,
        state=EgoState(
            pos=Vector2D(x=x, y=y),
            velocity=Vector2D(x=v * math.cos(yaw), y=v * math.sin(yaw)),
            acceleration=Vector2D(x=0.0, y=0.0),
            yaw=yaw,
            steering_angle=0.0,
        ),
    )


def _make_request(cfg) -> PlanningRequest:
    lanes = [_straight_lane()]
    ego = _ego(20.0, 2.0, 10.0)
    target_speed = 15.0
    goal = get_goal_region(
        curr_ego_state=ego,
        lanes=lanes,
        horizon=int(cfg.planner.horizon),
        length=float(cfg.scenario.goal.length) if cfg.scenario.goal else 5.0,
        width=float(cfg.scenario.goal.width) if cfg.scenario.goal else 5.0,
        target_speed=target_speed,
    )
    pred_env = PredictedEnvironment(objects={}, lanes=lanes, dt=cfg.planner.dt_sim, horizon=cfg.planner.horizon)
    return PlanningRequest(start_state=ego, goal_region=goal, target_speed=target_speed, environment=pred_env)


def main() -> None:
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="config")

    planner = build_planner(cfg)
    native = convert_cfg_to_native(cfg)

    # 1. identity + registry name attribute
    assert planner.name == "hybrid_a_star", planner.name

    # 2. plan() returns a well-formed PlanResult on a trivial free-road request
    request = _make_request(cfg)
    result = planner.plan(request)
    assert result.status_message is None or isinstance(result.status_message, str)
    assert result.success is True, result.status_message
    assert result.trajectory is not None and len(result.trajectory.states) > 1
    assert all(math.isfinite(s.state.pos.x) for s in result.trajectory.states)

    # 3. picklable (threaded mode forks/pickles the planner)
    clone = pickle.loads(pickle.dumps(planner))
    r2 = clone.plan(_make_request(cfg))
    assert r2.success is True, r2.status_message

    # 4. repeat planning is stateless-safe
    r3 = planner.plan(_make_request(cfg))
    assert r3.success is True, r3.status_message

    print("planner contract OK")


if __name__ == "__main__":
    main()
