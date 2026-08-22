import time
import hydra
from omegaconf import DictConfig

# --- System Components ---
from simulation.simulate import Simulation
from controllers.controllers import MPCController
from planner.planner import plan
from prediction.predictivity import predict_environment

# --- Visualization ---
from visualization.visualizer import visualize_scene
from visualization.tree_visualizer import visualize_search_tree

# --- Utilities & Physics ---
from collision.collision import get_ego_lane_info, get_distance_to_objects
from utils.helper import get_goal_region
from models.models import PlanningRequest


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the autonomous driving stack (Debug Environment).
    Initializes all components, runs the high-level planner, and executes 
    the control/simulation loop in a defined temporal sequence.
    """
    
    # 1. Initialize core modules
    sim = Simulation(cfg)
    controller = MPCController(cfg.vehicle, cfg.controller)
    
    newest_trajectory = None
    
    # Timing constants for the simulation loop
    SIM_STEP_MS = 50
    PLAN_EXECUTION_MS = 4000
    MAX_SIM_STEPS = int(PLAN_EXECUTION_MS / SIM_STEP_MS)

    while True:
        # ==========================================
        # PHASE 1: PREPARATION & PREDICTION
        # ==========================================
        curr_env = sim.get_environment()
        ego_state_stamped = sim.get_ego_state()
        
        # Predict environment for the planning horizon
        pred_env = predict_environment(
            environment=curr_env,
            prediction_horizon=cfg.planner.horizon,
            dt=cfg.planner.dt_sim
        )
        
        # Extract lane information and dynamic speed limits
        _, _, _, _, _, velocity_limit_lane = get_ego_lane_info(
            ego_state=ego_state_stamped.state,
            ego_length=cfg.vehicle.length,
            ego_width=cfg.vehicle.width,
            ego_rear_to_wheel=cfg.vehicle.rear_to_wheel,  
            lanes=curr_env.lanes
        )
        
        target_speed = velocity_limit_lane if velocity_limit_lane is not None else 10.0

        # Define the goal region for the A* search
        goal_region = get_goal_region(
            curr_ego_state=ego_state_stamped,
            lanes=curr_env.lanes,
            horizon=cfg.planner.horizon,
            length=3.0,
            width=3.0,
            target_speed=target_speed
        )
        
        # ==========================================
        # PHASE 2: TRAJECTORY PLANNING
        # ==========================================
        planning_request = PlanningRequest(
            start_state=ego_state_stamped,
            goal_region=goal_region,
            target_speed=target_speed,
            environment=pred_env
        )
        
        # Compute the trajectory
        plan_result = plan(planning_request, cfg, debug=True)
        
        if plan_result.success and plan_result.trajectory is not None:
            print("Path found successfully.")
            newest_trajectory = plan_result.trajectory
        else:
            # Keep the old trajectory as fallback 
            print(f"No path found: {plan_result.status_message}. Using fallback trajectory.")
        
        # Debugging: Print current distance to all objects
        distances, is_collision = get_distance_to_objects(
            current_ego=ego_state_stamped,
            previous_ego=ego_state_stamped,
            predicted_env=pred_env,
            ego_length=cfg.vehicle.length,
            ego_width=cfg.vehicle.width,
            ego_rear_to_wheel=cfg.vehicle.rear_to_wheel,  
            resolution_ms=10,
            calc_exact_distance=5.0
        )
        print(f"Distance to objects: {distances} | Collision: {is_collision}")
        
        # Export search tree visualization to HTML
        visualize_search_tree(
            plan_result=plan_result,
            pred_env=pred_env,
            goal_region=goal_region,
            cfg=cfg,
            vehicle_cfg=cfg.vehicle,
            output_filename="debug_tree.html"
        )

        # ==========================================
        # PHASE 3: CONTROL & SIMULATION LOOP
        # ==========================================
        step_counter = 0
        
        while step_counter < MAX_SIM_STEPS:
            start_time = time.perf_counter()
            
            # 1. Fetch current simulation state
            ego_state = sim.get_ego_state()
            curr_env = sim.get_environment()
            
            # 2. Compute control commands via MPC
            acc, steer_rate = 0.0, 0.0
            
            if newest_trajectory is not None:
                try:
                    acc, steer_rate = controller.compute_control(ego_state, newest_trajectory)[0]
                except Exception as e:
                    print(f"Controller failed ({e}). Triggering emergency brake.")
                    acc = cfg.vehicle.max_deceleration
                    steer_rate = 0.0
            else:
                print("No trajectory available. Triggering emergency brake.")
                acc = cfg.vehicle.max_deceleration
                
            # 3. Advance the physics simulation
            sim.step(acc, steer_rate, SIM_STEP_MS)
            
            # 4. Update real-time visualization
            visualize_scene(
                env=curr_env,
                ego=ego_state,
                vehicle_params=cfg.vehicle,
                trajectory=newest_trajectory,
                goal_region=goal_region,
                path=sim.ego_history
            )
            
            # 5. Enforce real-time execution limits
            compute_duration = time.perf_counter() - start_time
            sleep_duration = (SIM_STEP_MS / 1000.0) - compute_duration
            
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                print(f"Warning: Compute too slow for real-time tracking (overtime: {-sleep_duration:.4f}s)")
                
            step_counter += 1


if __name__ == "__main__":
    main()