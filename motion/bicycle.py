import math
from models.models import EgoInput, EgoState, EgoStateStamped, Vector2D
from omegaconf import DictConfig
from utils.helper import get_vector, get_magnitude, get_signed_magnitude
from dataclasses import dataclass


def kinematic_bicycle(
    stamped_state: EgoStateStamped,
    control: EgoInput,
    dt: int,
    vehicle_params: DictConfig,
) -> EgoStateStamped:
    """
    Lightweight kinematic bicycle propagation.

    Arguments:
        stamped_state: Current stamped state of the ego vehicle.
        control: Control input for the ego vehicle.
        dt: Time step in milliseconds.
        vehicle_params: Vehicle parameters including max acceleration, deceleration, steering angle, and wheel base.

    Returns:
        EgoStateStamped: The next stamped state of the ego vehicle with updated timestamp.
    """
    
    # Extract the inner state for easier access
    current_state = stamped_state.state
    
    x = current_state.pos.x
    y = current_state.pos.y
    yaw = current_state.yaw
    speed = get_signed_magnitude(current_state.velocity, current_state.yaw)
    dt_sec = dt / 1000.0

    # 1. Clip inputs
    accel = max(vehicle_params.max_deceleration,
                min(vehicle_params.max_acceleration, control.acceleration))

    new_steer = max(-vehicle_params.max_steer,
                    min(vehicle_params.max_steer, control.steering_angle))

    # 2. Calculate new speed (clip at 0 to prevent reversing)
    speed_next = max(0.0, speed + accel * dt_sec)
    
    # Average speed for more precise position integration during acceleration
    v_avg = (speed + speed_next) / 2.0

    # 3. Calculate omega using the CURRENT (new) steering angle and average speed
    omega = v_avg * math.tan(new_steer) / vehicle_params.wheel_base

    # 4. Integration
    if abs(omega) > 1e-6:
        yaw_next = yaw + omega * dt_sec

        x_next = x + (v_avg / omega) * (
            math.sin(yaw_next) - math.sin(yaw)
        )
        y_next = y - (v_avg / omega) * (
            math.cos(yaw_next) - math.cos(yaw)
        )
    else:
        yaw_next = yaw
        x_next = x + v_avg * math.cos(yaw) * dt_sec
        y_next = y + v_avg * math.sin(yaw) * dt_sec

    # 5. Velocity vector (at the end of the time step)
    vx_next = speed_next * math.cos(yaw_next)
    vy_next = speed_next * math.sin(yaw_next)

    # 6. Acceleration vector (including centripetal acceleration for correct prediction models)
    # a_long = accel, a_lat = v * omega
    lat_accel = speed_next * omega
    ax_next = accel * math.cos(yaw_next) - lat_accel * math.sin(yaw_next)
    ay_next = accel * math.sin(yaw_next) + lat_accel * math.cos(yaw_next)

    # 7. Create the new inner state
    next_state = EgoState(
        pos=Vector2D(x_next, y_next),
        velocity=Vector2D(vx_next, vy_next),
        acceleration=Vector2D(ax_next, ay_next),
        yaw=yaw_next,
        steering_angle=new_steer,
    )
    
    # 8. Update the timestamp and return the stamped state wrapper
    next_timestamp = stamped_state.timestamp + dt

    return EgoStateStamped(
        timestamp=next_timestamp,
        state=next_state
    )

class DynamicBicycleModel:

    def __init__(
        self,
        ego_state: EgoState,
        vehicle_params: DictConfig,
    ):
        """
        Initialize from EgoState.

        EgoState velocities are assumed to be in the
        global frame and are converted into body frame.
        """

        # Vehicle parameters
        self.m = vehicle_params.m
        self.Iz = vehicle_params.Iz

        self.lf = vehicle_params.lf
        self.lr = vehicle_params.lr

        self.Cf = vehicle_params.Cf
        self.Cr = vehicle_params.Cr

        self.mu = vehicle_params.mu
        self.g = 9.81

        self.max_steer = vehicle_params.max_steer
        self.max_steer_rate = vehicle_params.max_steer_rate

        self.max_acceleration = vehicle_params.max_acceleration
        self.max_deceleration = vehicle_params.max_deceleration

        # Convert global velocity -> body velocity
        c = math.cos(ego_state.yaw)
        s = math.sin(ego_state.yaw)

        vx_body = (
            c * ego_state.velocity.x +
            s * ego_state.velocity.y
        )

        vy_body = 0.0

        self.internal_state = {
            'x': ego_state.pos.x,
            'y': ego_state.pos.y,
            'yaw': ego_state.yaw,
            'vx': vx_body,
            'vy': vy_body,
            'yaw_rate': 0.0,
            'steer': ego_state.steering_angle,
        }
    
    def step(self, acceleration: float, steer_rate: float, dt: int) -> EgoState:
        """
        Advance the vehicle by dt milliseconds using a blended Kinematic/Dynamic model.
        """
        s = self.internal_state
        dt_sec = dt / 1000.0

        # 1. Clip inputs
        accel_cmd = max(
            self.max_deceleration,
            min(self.max_acceleration, acceleration)
        )

        steer_rate_cmd = max(
            -self.max_steer_rate,
            min(self.max_steer_rate, steer_rate)
        )

        delta = s['steer'] + steer_rate_cmd * dt_sec
        delta = max(-self.max_steer, min(self.max_steer, delta))

        # We load the REAL velocity, no more max(0.1, ...) hack!
        vx = s['vx'] 
        vy = s['vy']
        r = s['yaw_rate']
        
        wheel_base = self.lf + self.lr

        # ==========================================
        # KINEMATIC FALLBACK (Low Speed & Standstill)
        # ==========================================
        if abs(vx) < 1.0:
            # Bypass tire forces completely to prevent singularity and phantom drifting
            Fyf, Fyr = 0.0, 0.0
            
            # Longitudinal dynamics
            dvx = accel_cmd
            
            # Lateral dynamics (forced to zero at low speed to prevent slide)
            dvy = 0.0
            vy = 0.0 
            
            # Yaw dynamics (calculated analytically without slip)
            dr = 0.0
            r = vx * math.tan(delta) / wheel_base
            
        # ==========================================
        # FULL DYNAMIC MODEL (High Speed)
        # ==========================================
        else:
            alpha_f = delta - math.atan2(vy + self.lf * r, vx)
            alpha_r = -math.atan2(vy - self.lr * r, vx)

            Fyf = self.Cf * alpha_f
            Fyr = self.Cr * alpha_r

            Fzf = self.m * self.g * self.lr / wheel_base
            Fzr = self.m * self.g * self.lf / wheel_base

            # Apply friction limits
            Fyf = max(-self.mu * Fzf, min(self.mu * Fzf, Fyf))
            Fyr = max(-self.mu * Fzr, min(self.mu * Fzr, Fyr))

            # Calculate accelerations
            dvx = accel_cmd + r * vy - Fyf * math.sin(delta) / self.m
            dvy = -r * vx + (Fyf * math.cos(delta) + Fyr) / self.m
            dr = (self.lf * Fyf * math.cos(delta) - self.lr * Fyr) / self.Iz

        # 2. Integration
        vx += dvx * dt_sec
        vy += dvy * dt_sec
        
        # Only integrate yaw rate if we used the dynamic model 
        # (in kinematic mode, we set it analytically above)
        if abs(vx) >= 1.0:
            r += dr * dt_sec

        # Prevent reversing (if the system does not support negative longitudinal velocity)
        vx = max(0.0, vx)

        # 3. Global pose update
        yaw = s['yaw'] + r * dt_sec

        c = math.cos(yaw)
        ss = math.sin(yaw)

        x = s['x'] + (vx * c - vy * ss) * dt_sec
        y = s['y'] + (vx * ss + vy * c) * dt_sec

        # 4. Save internal state
        self.internal_state = {
            'x': x,
            'y': y,
            'yaw': yaw,
            'vx': vx,
            'vy': vy,
            'yaw_rate': r,
            'steer': delta,
        }

        ax_body = dvx - r * vy
        ay_body = dvy + r * vx

        # 5. Return updated EgoState
        return EgoState(
            pos=Vector2D(x, y),
            velocity=Vector2D(
                vx * c - vy * ss,
                vx * ss + vy * c
            ),
            acceleration=Vector2D(
                ax_body * c - ay_body * ss,
                ax_body * ss + ay_body * c
            ),
            yaw=yaw,
            steering_angle=delta,
        )