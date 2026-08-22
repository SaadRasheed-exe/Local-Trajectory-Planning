from visualization.visualizer import visualize_scene
from simulation.simulate import Simulation
from planner.planner import plan
from models.models import PlanningRequest, Trajectory, PredictedEnvironment
from utils.helper import get_goal_region
from motion.motion_prediction import predict_motion_constant_velocity
from controllers.controllers import MPCController

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    sim = Simulation(cfg)

    goal_reached = False
    full_trajectory = Trajectory(states=[])
    controller = MPCController(cfg.vehicle, cfg.controller.mpc)

    while not goal_reached:

        curr_env = sim.get_environment()
        pred_objs = predict_motion_constant_velocity(curr_env.objects, prediction_horizon=cfg.planner.horizon, dt=cfg.planner.dt_sim)
        pred_env = PredictedEnvironment(
            objects={obj.state.id: [obj] for obj in pred_objs},
            lanes=curr_env.lanes,
            dt=cfg.planner.dt_sim,
            horizon=cfg.planner.horizon
        )

        planning_request = PlanningRequest(
            start_state=sim.get_ego_state(),
            goal_region=get_goal_region(
                curr_ego_state=sim.get_ego_state(),
                lanes=curr_env.lanes,
                horizon=cfg.scenario.goal.horizon,
                length=cfg.scenario.goal.length,
                width=cfg.scenario.goal.width
            ),
            target_speed=13+8/9,
            environment=pred_env
        )

        plan_result = plan(planning_request, cfg)

        if plan_result.success and plan_result.trajectory is not None:
            acceleration, steering_rate = controller.compute_control(sim.get_ego_state(), plan_result.trajectory)
            sim.step(acceleration, steering_rate, cfg.controller.mpc.dt)

            if goal_reached:
                print("Goal reached!")
                break
        
        else:
            print(f"Planning failed: {plan_result.status_message}")
            break
        
    
    visualize_scene(
        env=sim.get_environment(0.0),
        ego=sim.get_ego_state(0.0),
        vehicle_params=cfg.vehicle,
        trajectory=full_trajectory
    )

if __name__ == "__main__":
    main()
