from visualization.visualizer import visualize_scene
from simulation.simulate import Simulation
from planner.planner import plan
from models.models import PlanningRequest, GoalRegion, Vector2D, Trajectory, PredictedEnvironment
from motion.motion_prediction import predict_motion_constant_velocity
from visualization.visualizer import visualize_scene
import hydra
from omegaconf import DictConfig
import threading
import time
from controllers.controllers import MPCController
from utils.helper import get_goal_region
from prediction.predictivity import predict_environment

class SharedState:
    """
    A thread-safe shared state object used for communication between the 
    high-level planner (producer) and the real-time controller (consumer).

    Attributes:
        lock (threading.Lock): Mutex lock to ensure thread-safe read/write operations.
        trajectory (Optional[Trajectory]): The most recent trajectory computed by the planner.
        is_running (bool): Flag to indicate whether the background threads should continue running.
    """
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.trajectory = Trajectory([])
        self.is_running = True



def controller_worker(
    sim: Simulation, 
    controller: MPCController, 
    shared_state: SharedState, 
    dt_sec: float,
    cfg: DictConfig
) -> None:
    """
    Background worker function that runs the MPC controller in a strict real-time loop.
    It reads the latest trajectory, computes control commands, and advances the simulation.

    Args:
        sim (Simulation): The simulation environment instance.
        controller (MPCController): The Model Predictive Control instance.
        shared_state (SharedState): The thread-safe object containing the latest trajectory.
        dt_sec (float): The time step for the controller control loop in seconds.

    Returns:
        None
    """
    while shared_state.is_running:
        loop_start = time.perf_counter()

        # 1. Get current state from simulation
        ego_state = sim.get_ego_state()

        # 2. Safely read the latest trajectory using the lock
        current_trajectory = None
        with shared_state.lock:
            current_trajectory = shared_state.trajectory

        # 3. Compute control commands and apply fallback if necessary
        acc, steer_rate = 0.0, 0.0
        if current_trajectory is not None:
            try:
                acc, steer_rate = controller.compute_control(ego_state, current_trajectory)
            except Exception:
                # Fallback: Emergency braking if the MPC fails to find a solution
                acc, steer_rate = cfg.vehicle.max_deceleration, 0.0 
        else:
            # No trajectory available yet -> standstill / emergency brake
            acc = cfg.vehicle.max_deceleration #TODO aus config maximale verzögerung auslesen

        # 4. Advance the simulation (the controller drives the simulation time)
        dt_ms = int(dt_sec * 1000)
        sim.step(acc, steer_rate, dt_ms)

        # 5. Enforce strict frequency (sleep for the remainder of the clock cycle)
        elapsed = time.perf_counter() - loop_start
        sleep_time = dt_sec - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the autonomous driving stack. Initializes all components, 
    starts the background controller thread, and runs the high-level planner continuously.

    Args:
        cfg (DictConfig): The Hydra configuration object containing system and vehicle parameters.

    Returns:
        None
    """
    sim = Simulation(cfg)
    controller = MPCController(cfg.vehicle, cfg.controller)
    
    # Initialize the communication object
    shared_state = SharedState()

    # Start the background controller thread
    # TODO: argumente anschauen vom controller
    dt_controller_sec = cfg.controller.dt / 1000.0
    ctrl_thread = threading.Thread(
        target=controller_worker, 
        args=(sim, controller, shared_state, dt_controller_sec, cfg),
        daemon=True  # Automatically stops when the main program exits
    )
    ctrl_thread.start()

    try:
        while True:
            #TODO possibly stop time and wait at the end for remaining time
            # 1. Fetch environment and state for the planner
            curr_env = sim.get_environment()
            ego_state_stamped = sim.get_ego_state()

            # TODO: predict environment
            pred_env = predict_environment(environment = curr_env,
                                           prediction_horizon = cfg.planner.horizon,
                                           dt = cfg.planner.dt_sim)
            
            # TODO: calculate goal region (dummy arguments used)
            goal_region = get_goal_region(curr_ego_state=ego_state_stamped,
                                          lanes=curr_env.lanes,
                                          horizon=cfg.planner.horizon * cfg.planner.horizon_increment,
                                          length=3.0,
                                          width=3.0) 
            
            # TODO: get dynamic velocity limit
            velocity_limit = 10.0

            planning_request = PlanningRequest(
                start_state=ego_state_stamped,
                goal_region=goal_region,
                target_speed=velocity_limit,
                environment=pred_env
            )

            # 2. Compute the plan 
            plan_result = plan(planning_request, cfg)

            # 3. Pass the new trajectory to the controller
            if plan_result.success and plan_result.trajectory is not None:
                with shared_state.lock:
                    shared_state.trajectory = plan_result.trajectory
            else:
                print(f"No path found: {plan_result.status_message}")
                # If no path is found, the old trajectory remains in shared_state.
                # The MPC will simply continue following it (default fallback behavior).
            
            # 4. Visualization (runs safely in the main thread)
            visualize_scene(
                env=curr_env,
                ego=ego_state_stamped,
                vehicle_params=cfg.vehicle,
                trajectory=plan_result.trajectory
            )
            
    except KeyboardInterrupt:
        print("Shutting down simulation...")
        # Shut down threads cleanly
        shared_state.is_running = False
        ctrl_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()


#@hydra.main(version_base=None, config_path="configs", config_name="config")
def _main(cfg: DictConfig):
    sim = Simulation()
    controller = None #TODO init of the controller
    #TODO start thread for controller

    while True:

        curr_env = sim.get_environment()
        ego_state_stamped = sim.get_ego_state()

        #TODO predict environment
        pred_env = curr_env
        ############

        #TODO calculate goal region
        goal_region = GoalRegion(...)
        #############

        #TODO get velocity_limit -> target_speed
        velocity_limit = None
        

        planning_request = PlanningRequest(start_state=ego_state_stamped,
                                           goal_region=goal_region,
                                           target_speed=velocity_limit,
                                           environment=pred_env)

        plan_result = plan(planning_request, cfg)

        if plan_result.success and plan_result.trajectory is not None:
              #TODO speak with controller thread
              ...

        else:
            print(f"No path found: {plan_result.status_message}")
            #TODO: Vehicle should follow the last caluclated trajectorie and break 
        
        #TODO change visualizier to get vehicle config
        visualize_scene(
            env=curr_env,
            ego=ego_state_stamped,
            vehicle_params=vehicle_params,
            trajectory=plan_result.trajectory
        )
        