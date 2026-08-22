import math
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.models import EgoInput, EgoState, EgoStateStamped, Vector2D
from omegaconf import DictConfig, OmegaConf
from utils.helper import get_magnitude, get_vector, global_to_ego_axis


_MIN_SPEED_FOR_SLIP = 0.1


# Role: fast stateless model for planner trajectory rollout.
def kinematic_bicycle_model(
    state: EgoStateStamped,
    control: EgoInput,
    dt_ms: int,
    veh_cfg: DictConfig,
    limit_steer_rate: bool = False,
) -> EgoStateStamped:
    """Fast stateless bicycle model intended for planner rollout."""

    current_state = state.state
    dt_sec = float(dt_ms) / 1000.0

    max_acceleration = float(veh_cfg.max_acceleration)
    max_deceleration = float(veh_cfg.max_deceleration)
    if max_deceleration > 0.0:
        max_deceleration = -max_deceleration
    acceleration = max(
        max_deceleration,
        min(max_acceleration, float(control.acceleration)),
    )

    steering_angle = max(
        -float(veh_cfg.max_steer),
        min(float(veh_cfg.max_steer), float(control.steering_angle)),
    )

    if limit_steer_rate:
        max_steering_delta = float(veh_cfg.max_steer_rate) * dt_sec
        steering_angle = max(
            current_state.steering_angle - max_steering_delta,
            min(
                current_state.steering_angle + max_steering_delta,
                steering_angle,
            ),
        )

    speed = get_magnitude(current_state.velocity)
    next_speed = max(0.0, speed + acceleration * dt_sec)
    average_speed = 0.5 * (speed + next_speed)

    yaw = current_state.yaw
    yaw_rate = average_speed * math.tan(steering_angle) / veh_cfg.wheel_base

    if abs(yaw_rate) > 1e-9:
        next_yaw = (yaw + yaw_rate * dt_sec + math.pi) % (2.0 * math.pi) - math.pi
        turn_radius = average_speed / yaw_rate
        next_x = current_state.pos.x + turn_radius * (
            math.sin(next_yaw) - math.sin(yaw)
        )
        next_y = current_state.pos.y - turn_radius * (
            math.cos(next_yaw) - math.cos(yaw)
        )
    else:
        next_yaw = yaw
        next_x = current_state.pos.x + average_speed * math.cos(yaw) * dt_sec
        next_y = current_state.pos.y + average_speed * math.sin(yaw) * dt_sec

    next_velocity = get_vector(next_speed, next_yaw)

    lateral_acceleration = next_speed * yaw_rate
    acceleration_x, acceleration_y, _ = global_to_ego_axis(
        acceleration,
        lateral_acceleration,
        0.0,
        0.0,
        -next_yaw,
    )
    next_acceleration = Vector2D(x=acceleration_x, y=acceleration_y)

    return EgoStateStamped(
        timestamp=state.timestamp + dt_ms,
        state=EgoState(
            pos=Vector2D(x=next_x, y=next_y),
            velocity=next_velocity,
            acceleration=next_acceleration,
            yaw=next_yaw,
            steering_angle=steering_angle,
        ),
    )

#  ─ OVERTAKE MANOEUVRE ─

OVERTAKE_PHASES = [
    (1.0,   0.0),   # approach: drive straight for 1.0s
    (1.0,   3.0),   # pull out:  steer left  3°
    (1.0,  -3.0),   # pass:      straighten up
    (1.0,  -3.0),   # pull in:   steer right 3°
    (1.0,   3.0),   # traighten up
    (1.0,   0.0),   # Going back
    (1.0,   0.0),   # Straighten up
    (1.0,   0.0),   # Complete:  straighten up
]

OVERTAKE_PHASE_NAMES = [
    "approach",
    "pull out",
    "pass",
    "pull in",
    "straighten",
    "return",
    "complete",
    "done",
]


#  ─ VISUALIZER ─

class Visualizer:

    def __init__(self, vehicle: 'Vehicle'):
        self.vehicle = vehicle

        self.trail_x:    List[float] = []
        self.trail_y:    List[float] = []
        self.steer_hist: List[float] = []
        self.yaw_hist:   List[float] = []
        self.beta_hist:  List[float] = []
        self.v_hist:     List[float] = []
        self.yr_hist:    List[float] = []
        self.t_hist:     List[float] = []

        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6))
        self.ax_world = self.axes[0]
        self.ax_stats = self.axes[1]
        self.fig.suptitle('Bicycle model — overtake manoeuvre', fontsize=12)
        self.fig.tight_layout()
        plt.ion()

    def record(self, state: EgoStateStamped):
        ego_state = state.state
        self.trail_x.append(ego_state.pos.x)
        self.trail_y.append(ego_state.pos.y)
        self.steer_hist.append(math.degrees(ego_state.steering_angle))
        self.yaw_hist.append(math.degrees(ego_state.yaw))
        self.beta_hist.append(math.degrees(self.vehicle.beta))
        self.v_hist.append(get_magnitude(ego_state.velocity))
        self.yr_hist.append(math.degrees(self.vehicle.yaw_rate))
        self.t_hist.append(state.timestamp / 1000.0)

    def _draw_wheel(self, ax, cx, cy, angle_rad):
        wl = self.vehicle.wheel_length
        ww = self.vehicle.wheel_width
        wheel = patches.Rectangle(
            (-wl / 2, -ww / 2), wl, ww,
            linewidth=0.8,
            edgecolor='#5F5E5A',
            facecolor='#2C2C2A'
        )
        t = (transforms.Affine2D()
             .rotate(angle_rad)
             .translate(cx, cy)
             + ax.transData)
        wheel.set_transform(t)
        ax.add_patch(wheel)

    def _draw_car(self, ax, state: EgoStateStamped):
        ego_state = state.state
        x, y, yaw = ego_state.pos.x, ego_state.pos.y, ego_state.yaw
        steering_angle = ego_state.steering_angle
        p  = self.vehicle.veh_cfg
        ht = self.vehicle.track / 2
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        def world(dx, dy):
            return (x + dx * cos_y - dy * sin_y,
                    y + dx * sin_y + dy * cos_y)

        # Car body
        body = patches.FancyBboxPatch(
            (-p.lr, -p.width / 2), p.wheel_base, p.width,
            boxstyle="round,pad=0.05",
            linewidth=1.2,
            edgecolor='#185FA5',
            facecolor='#B5D4F4'
        )
        body.set_transform(
            transforms.Affine2D()
            .rotate(yaw).translate(x, y) + ax.transData
        )
        ax.add_patch(body)

        # Windscreen
        wscreen = patches.FancyBboxPatch(
            (p.lf * 0.1, -p.width * 0.28),
            p.lf * 0.75, p.width * 0.56,
            boxstyle="round,pad=0.02",
            linewidth=0,
            facecolor='#85B7EB',
            alpha=0.6
        )
        wscreen.set_transform(
            transforms.Affine2D()
            .rotate(yaw).translate(x, y) + ax.transData
        )
        ax.add_patch(wscreen)

        # Four wheels
        wheel_defs = [
            ( p.lf,  ht,  True),
            ( p.lf, -ht,  True),
            (-p.lr,  ht,  False),
            (-p.lr, -ht,  False),
        ]
        for dx, dy, steered in wheel_defs:
            wx, wy = world(dx, dy)
            angle  = yaw + (steering_angle if steered else 0.0)
            self._draw_wheel(ax, wx, wy, angle)

        # Heading arrow
        arrow_l = p.length * 0.65
        ax.annotate('',
            xy=(x + arrow_l * cos_y, y + arrow_l * sin_y),
            xytext=(x, y),
            arrowprops=dict(arrowstyle='->', color='#185FA5', lw=2.0)
        )

        # Sideslip arrow
        if abs(self.vehicle.beta) > 0.003:
            va = yaw + self.vehicle.beta
            ax.annotate('',
                xy=(x + arrow_l * math.cos(va),
                    y + arrow_l * math.sin(va)),
                xytext=(x, y),
                arrowprops=dict(arrowstyle='->', color='#D85A30',
                                lw=1.8, linestyle='dashed')
            )

        # Steering arc
        if abs(steering_angle) > 0.01:
            fax, fay = world(p.lf, 0)
            arc = patches.Arc(
                (fax, fay), 1.0, 1.0,
                angle=math.degrees(yaw),
                theta1=0,
                theta2=math.degrees(steering_angle),
                color='#378ADD', linewidth=1.2
            )
            ax.add_patch(arc)

        ax.plot(x, y, 'o', color='#E24B4A', markersize=5, zorder=6)

    def _draw_world(self, state: EgoStateStamped):
        ego_state = state.state
        ax = self.ax_world
        ax.cla()
        ax.set_facecolor('#F1EFE8')
        ax.grid(True, color='white', linewidth=0.8, zorder=0)
        ax.set_aspect('equal')

        # Road lanes — two dashed lines 3.5m apart (standard lane width)
        for lane_y in [-3.5, 0.0, 3.5]:
            style = '--' if lane_y == 0.0 else '-'
            color = '#BA7517' if lane_y == 0.0 else '#888780'
            ax.axhline(lane_y, color=color, linewidth=1.0,
                       linestyle=style, zorder=0, alpha=0.6)

        if len(self.trail_x) > 1:
            ax.plot(self.trail_x, self.trail_y,
                    '--', color='#1D9E75', linewidth=1.4,
                    alpha=0.7, label='path', zorder=1)

        self._draw_car(ax, state)

        # Fixed view — wide enough to see the full overtake
        ax.set_xlim(ego_state.pos.x - 15, ego_state.pos.x + 40)
        ax.set_ylim(-10, 10)

        ax.set_title(
            f't={state.timestamp / 1000.0:.2f}s   '
            f'v={get_magnitude(ego_state.velocity):.1f} m/s   '
            f'steer={math.degrees(ego_state.steering_angle):.1f}°   '
            f'β={math.degrees(self.vehicle.beta):.2f}°',
            fontsize=9
        )
        ax.set_xlabel('x [m]  (forward)')
        ax.set_ylabel('y [m]  (lateral)')
        ax.plot([], [], '-',  color='#185FA5', label='heading ψ')
        ax.plot([], [], '--', color='#D85A30', label='velocity (β)')
        ax.plot([], [], 'o',  color='#E24B4A', label='CoG', markersize=5)
        ax.legend(loc='upper left', fontsize=8)

    def _draw_stats(self):
        ax = self.ax_stats
        ax.cla()
        ax.set_facecolor('#F1EFE8')
        t = self.t_hist

        ax.plot(t, self.steer_hist, color='#185FA5',
                linewidth=1.8, label='steer δ [deg]')
        ax.plot(t, self.yaw_hist,   color='#0F6E56',
                linewidth=1.6, label='yaw ψ [deg]')
        ax.plot(t, self.yr_hist,    color='#534AB7',
                linewidth=1.4, linestyle='-.', label='yaw rate [deg/s]')
        ax.plot(t, self.beta_hist,  color='#D85A30',
                linewidth=1.4, linestyle='--', label='sideslip β [deg]')
        ax.plot(t, self.v_hist,     color='#888780',
                linewidth=1.2, linestyle=':',  label='speed [m/s]')

        ax.axhline(0, color='#B4B2A9', linewidth=0.5)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('value')
        ax.set_title('State history', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, color='white', linewidth=0.8)

    def update(self, state: EgoStateStamped):
        self.record(state)
        self._draw_world(state)
        self._draw_stats()
        if plt.get_backend().lower() != "agg":
            plt.pause(0.001)

    def show(self):
        plt.ioff()
        if plt.get_backend().lower() != "agg":
            plt.show()


#  ─ VEHICLE ─

class Vehicle:
    def __init__(self, x, y, yaw, v):

        self.length        = 2.338
        self.width         = 1.381
        self.rear_to_wheel = 0.339
        self.wheel_length  = 0.531
        self.wheel_width   = 0.125
        self.track         = 1.094
        self.wheel_base    = 1.686

        self.Caf  = 2 * 32857.5
        self.Car  = 2 * 32857.5
        self.mass = 633.0
        self.lf   = 0.9442
        self.lr   = 0.7417
        self.Iz   = 430.166

        self.veh_cfg = OmegaConf.create({
            "max_steer": 0.6,
            "max_steer_rate": 0.5,
            "max_steer_acceleration": 2.0,
            "steer_response": 4.0,
            "lf": self.lf,
            "lr": self.lr,
            "wheel_base": self.wheel_base,
            "wheel_length": self.wheel_length,
            "wheel_width": self.wheel_width,
            "track": self.track,
            "rear_to_wheel": self.rear_to_wheel,
            "width": self.width,
            "length": self.length,
            "m": self.mass,
            "Iz": self.Iz,
            "Cf": self.Caf,
            "Cr": self.Car,
            "max_acceleration": 3.0,
            "max_deceleration": 5.0,
            "mu": 0.85,
        })

        self.state = EgoStateStamped(
            timestamp=0,
            state=EgoState(
                pos=Vector2D(x=x, y=y),
                velocity=get_vector(v, yaw),
                acceleration=Vector2D(x=0.0, y=0.0),
                yaw=yaw,
                steering_angle=0.0,
            ),
        )
        self.beta = 0.0
        self.yaw_rate = 0.0

        self.control = EgoInput(steering_angle=0.0, acceleration=0.0)
        self.viz     = Visualizer(self)

        self.dynamic_model = DynamicBicycleModel(self.state.state, self.veh_cfg)

        # Overtake phase tracker
        self._phase_index    = 0       # which phase we are in
        self._phase_elapsed  = 0.0     # how long we have been in this phase
        
    def _get_target_steer(self, dt_sec: float) -> float:
        """
        Reads OVERTAKE_PHASES and returns the target steering angle
        for the current moment in time.
        """
        
        # Advance phase timer
        self._phase_elapsed += dt_sec

        # If current phase has finished, move to next one
        if self._phase_index < len(OVERTAKE_PHASES) - 1:
            phase_duration = OVERTAKE_PHASES[self._phase_index][0]
            if self._phase_elapsed >= phase_duration:
                self._phase_index   += 1
                self._phase_elapsed  = 0.0

        # Return target steer angle for the current phase (convert to radians)
        target_deg = OVERTAKE_PHASES[self._phase_index][1]
        return math.radians(target_deg)

    def Motion_control(self, dt_ms: int = 50):
        """Track the overtake phase target with the dynamic model."""
        dt_sec = dt_ms / 1000.0
        target_steer = self._get_target_steer(dt_sec)
        self.control = EgoInput(steering_angle=target_steer, acceleration=0.0)
        self.Motion_model(dt_ms)

    def Motion_model(self, dt_ms: int = 50):
        ego_state = self.dynamic_model.step_linear_control(self.control, dt_ms)
        self.state = EgoStateStamped(
            timestamp=self.state.timestamp + dt_ms,
            state=ego_state,
        )
        dynamic_state = self.dynamic_model.dynamic_state
        self.yaw_rate = self.dynamic_model.yaw_rate
        self.beta = math.atan2(dynamic_state.velocity.y, dynamic_state.velocity.x)
        self.viz.update(self.state)

    def Print_ego(self):
        s = self.state.state
        phase_index = min(self._phase_index, len(OVERTAKE_PHASE_NAMES) - 1)
        print(
            f"  [{OVERTAKE_PHASE_NAMES[phase_index]}]  "
            f"x={s.pos.x:.2f}  y={s.pos.y:.2f}  "
            f"yaw={math.degrees(s.yaw):.1f}°  "
            f"steer={math.degrees(s.steering_angle):.1f}°  "
            f"v={get_magnitude(s.velocity):.2f} m/s  "
            f"β={math.degrees(self.beta):.3f}°"
        )

class DynamicBicycleModel:
    """Stateful dynamic bicycle model intended for simulation."""

    # Role: initialize simulator state from an EgoState and vehicle config.
    def __init__(
        self,
        ego_state: EgoState,
        veh_cfg: DictConfig,
        yaw_rate: float = 0.0,
    ) -> None:
        self.veh_cfg = veh_cfg
        vx_body, vy_body, _ = global_to_ego_axis(
            ego_state.velocity.x,
            ego_state.velocity.y,
            0.0,
            0.0,
            ego_state.yaw,
        )
        # Internal velocity is stored in the vehicle/body frame.
        self.dynamic_state = EgoState(
            pos=Vector2D(x=ego_state.pos.x, y=ego_state.pos.y),
            velocity=Vector2D(x=max(0.0, vx_body), y=vy_body),
            acceleration=Vector2D(x=0.0, y=0.0),
            yaw=ego_state.yaw,
            steering_angle=ego_state.steering_angle,
        )
        self.yaw_rate = yaw_rate
        self.ego_state = EgoState(
            pos=Vector2D(x=ego_state.pos.x, y=ego_state.pos.y),
            velocity=ego_state.velocity,
            acceleration=ego_state.acceleration,
            yaw=ego_state.yaw,
            steering_angle=ego_state.steering_angle,
        )
        self.steering_rate = 0.0

    # Role: replace the internal simulator state with a new EgoState.
    #new state without creating a new model.
    def reset(self, ego_state: EgoState, yaw_rate: float = 0.0) -> None:
        vx_body, vy_body, _ = global_to_ego_axis(
            ego_state.velocity.x,
            ego_state.velocity.y,
            0.0,
            0.0,
            ego_state.yaw,
        )

        self.dynamic_state = EgoState(
            pos=Vector2D(x=ego_state.pos.x, y=ego_state.pos.y),
            velocity=Vector2D(x=max(0.0, vx_body), y=vy_body),
            acceleration=Vector2D(x=0.0, y=0.0),
            yaw=ego_state.yaw,
            steering_angle=ego_state.steering_angle,
        )
        self.yaw_rate = yaw_rate
        self.ego_state = EgoState(
            pos=Vector2D(x=ego_state.pos.x, y=ego_state.pos.y),
            velocity=ego_state.velocity,
            acceleration=ego_state.acceleration,
            yaw=ego_state.yaw,
            steering_angle=ego_state.steering_angle,
        )
        self.steering_rate = 0.0

    # Role: convert target steering input into steer rate, then use nonlinear dynamics.
    def step_control(self, control: EgoInput, dt_ms: int) -> EgoState:
        """This accepts target steering angle and acceleration."""

        dt_sec = float(dt_ms) / 1000.0
        target_steering_angle = max(
            -float(self.veh_cfg.max_steer),
            min(float(self.veh_cfg.max_steer), float(control.steering_angle)),
        )
        steering_error = target_steering_angle - self.ego_state.steering_angle
        steer_response = float(self.veh_cfg.get("steer_response", 4.0))
        steer_rate = steer_response * steering_error if dt_sec > 0.0 else 0.0

        return self.step(
            acceleration=control.acceleration,
            steer_rate=steer_rate,
            dt_ms=dt_ms,
        )

    # Role: convert target steering input into steer rate, then use linear dynamics.
    def step_linear_control(self, control: EgoInput, dt_ms: int) -> EgoState:
        """Step with the linear bicycle formula and a target steering command."""

        dt_sec = float(dt_ms) / 1000.0
        target_steering_angle = max(
            -float(self.veh_cfg.max_steer),
            min(float(self.veh_cfg.max_steer), float(control.steering_angle)),
        )
        steering_error = target_steering_angle - self.ego_state.steering_angle
        steer_response = float(self.veh_cfg.get("steer_response", 4.0))
        steer_rate = steer_response * steering_error if dt_sec > 0.0 else 0.0

        return self.step_linear(
            acceleration=control.acceleration,
            steer_rate=steer_rate,
            dt_ms=dt_ms,
        )

    def _apply_steering_actuator(self, steer_rate: float, dt_sec: float) -> float:
        max_steer = float(self.veh_cfg.max_steer)
        max_steer_rate = float(self.veh_cfg.max_steer_rate)
        requested_rate = max(
            -max_steer_rate,
            min(max_steer_rate, float(steer_rate)),
        )

        max_steer_acceleration = float(
            self.veh_cfg.get(
                "max_steer_acceleration",
                max_steer_rate / max(dt_sec, 1e-9),
            )
        )
        max_rate_delta = max(0.0, max_steer_acceleration) * dt_sec
        self.steering_rate = max(
            self.steering_rate - max_rate_delta,
            min(self.steering_rate + max_rate_delta, requested_rate),
        )

        steering_angle = self.ego_state.steering_angle + self.steering_rate * dt_sec
        if steering_angle > max_steer:
            steering_angle = max_steer
            if self.steering_rate > 0.0:
                self.steering_rate = 0.0
        elif steering_angle < -max_steer:
            steering_angle = -max_steer
            if self.steering_rate < 0.0:
                self.steering_rate = 0.0

        return steering_angle

    # Role: propagate simulator state with the bicycle formula.
    def step_linear(
        self,
        acceleration: float,
        steer_rate: float,
        dt_ms: int,
    ) -> EgoState:
        """Advance the vehicle with the linear dynamic bicycle formula."""

        dt_sec = float(dt_ms) / 1000.0
        if dt_sec <= 0.0:
            return self.to_ego_state(Vector2D(x=0.0, y=0.0))

        max_acceleration = float(self.veh_cfg.max_acceleration)
        max_deceleration = float(self.veh_cfg.max_deceleration)
        if max_deceleration > 0.0:
            max_deceleration = -max_deceleration
        acceleration = max(
            max_deceleration,
            min(max_acceleration, float(acceleration)),
        )

        previous = self.dynamic_state
        steering_angle = self._apply_steering_actuator(steer_rate, dt_sec)

        speed = max(
            0.0,
            get_magnitude(previous.velocity) + acceleration * dt_sec,
        )

        if speed > 1e-3:
            beta = math.atan2(previous.velocity.y, previous.velocity.x)
            yaw_rate = self.yaw_rate

            cf = float(self.veh_cfg.Cf)
            cr = float(self.veh_cfg.Cr)
            lf = float(self.veh_cfg.lf)
            lr = float(self.veh_cfg.lr)
            mass = float(self.veh_cfg.m)
            inertia_z = float(self.veh_cfg.Iz)

            a11 = -(cf + cr) / (mass * speed)
            a12 = (-lf * cf + lr * cr) / (mass * speed**2) - 1.0
            a21 = (-lf * cf + lr * cr) / inertia_z
            a22 = -(lf**2 * cf + lr**2 * cr) / (inertia_z * speed)
            b1 = cf / (mass * speed)
            b2 = lf * cf / inertia_z

            beta_dot = a11 * beta + a12 * yaw_rate + b1 * steering_angle
            yaw_acceleration = a21 * beta + a22 * yaw_rate + b2 * steering_angle

            next_beta = beta + beta_dot * dt_sec
            next_yaw_rate = yaw_rate + yaw_acceleration * dt_sec
        else:
            next_beta = 0.0
            next_yaw_rate = 0.0

        next_yaw = (
            previous.yaw + next_yaw_rate * dt_sec + math.pi
        ) % (2.0 * math.pi) - math.pi

        next_vx = speed * math.cos(next_beta)
        next_vy = speed * math.sin(next_beta)

        next_velocity_x, next_velocity_y, _ = global_to_ego_axis(
            next_vx,
            next_vy,
            0.0,
            0.0,
            -next_yaw,
        )
        next_velocity_global = Vector2D(x=next_velocity_x, y=next_velocity_y)

        next_x = previous.pos.x + next_velocity_global.x * dt_sec
        next_y = previous.pos.y + next_velocity_global.y * dt_sec

        acceleration_x, acceleration_y, _ = global_to_ego_axis(
            acceleration,
            speed * next_yaw_rate,
            0.0,
            0.0,
            -next_yaw,
        )
        acceleration_global = Vector2D(x=acceleration_x, y=acceleration_y)

        self.dynamic_state = EgoState(
            pos=Vector2D(x=next_x, y=next_y),
            velocity=Vector2D(x=next_vx, y=next_vy),
            acceleration=acceleration_global,
            yaw=next_yaw,
            steering_angle=steering_angle,
        )
        self.yaw_rate = next_yaw_rate
        self.ego_state = EgoState(
            pos=Vector2D(x=next_x, y=next_y),
            velocity=next_velocity_global,
            acceleration=acceleration_global,
            yaw=next_yaw,
            steering_angle=steering_angle,
        )

        return self.ego_state

    # Role: propagate simulator state with the nonlinear tire-force bicycle model.
    def step(self, acceleration: float, steer_rate: float, dt_ms: int) -> EgoState:
        """Advance the internal vehicle state by dt_ms."""

        dt_sec = float(dt_ms) / 1000.0
        if dt_sec <= 0.0:
            return self.to_ego_state(Vector2D(x=0.0, y=0.0))

        max_acceleration = float(self.veh_cfg.max_acceleration)
        max_deceleration = float(self.veh_cfg.max_deceleration)
        if max_deceleration > 0.0:
            max_deceleration = -max_deceleration
        acceleration = max(
            max_deceleration,
            min(max_acceleration, float(acceleration)),
        )

        previous = self.dynamic_state
        steering_angle = self._apply_steering_actuator(steer_rate, dt_sec)

        vx = previous.velocity.x
        vy = previous.velocity.y
        yaw_rate = self.yaw_rate
        slip_vx = max(abs(vx), _MIN_SPEED_FOR_SLIP)

        alpha_front = steering_angle - math.atan2(
            vy + float(self.veh_cfg.lf) * yaw_rate,
            slip_vx,
        )
        alpha_rear = -math.atan2(
            vy - float(self.veh_cfg.lr) * yaw_rate,
            slip_vx,
        )

        front_lateral_force = float(self.veh_cfg.Cf) * alpha_front
        rear_lateral_force = float(self.veh_cfg.Cr) * alpha_rear

        #Normal force means vertical load from gravity. Heavier load allows more friction force.
        wheel_base = float(self.veh_cfg.lf) + float(self.veh_cfg.lr)
        front_normal_force = (
            float(self.veh_cfg.m) * 9.81 * float(self.veh_cfg.lr) / wheel_base
        )
        rear_normal_force = (
            float(self.veh_cfg.m) * 9.81 * float(self.veh_cfg.lf) / wheel_base
        )

        force_scale = min(1.0, math.hypot(vx, vy) / _MIN_SPEED_FOR_SLIP)
        friction = float(self.veh_cfg.mu)
        front_lateral_force = force_scale * max(
            -friction * front_normal_force,
            min(friction * front_normal_force, front_lateral_force),
        )
        rear_lateral_force = force_scale * max(
            -friction * rear_normal_force,
            min(friction * rear_normal_force, rear_lateral_force),
        )

        dvx = (
            acceleration
            + yaw_rate * vy
            - front_lateral_force * math.sin(steering_angle) / float(self.veh_cfg.m)
        )
        dvy = (
            -yaw_rate * vx
            + (
                front_lateral_force * math.cos(steering_angle)
                + rear_lateral_force
            )
            / float(self.veh_cfg.m)
        )
        yaw_acceleration = (
            float(self.veh_cfg.lf) * front_lateral_force * math.cos(steering_angle)
            - float(self.veh_cfg.lr) * rear_lateral_force
        ) / float(self.veh_cfg.Iz)

        next_vx = max(0.0, vx + dvx * dt_sec)
        next_vy = vy + dvy * dt_sec
        next_yaw_rate = yaw_rate + yaw_acceleration * dt_sec
        next_yaw = (
            previous.yaw + next_yaw_rate * dt_sec + math.pi
        ) % (2.0 * math.pi) - math.pi

        next_velocity_x, next_velocity_y, _ = global_to_ego_axis(
            next_vx,
            next_vy,
            0.0,
            0.0,
            -next_yaw,
        )
        next_velocity_global = Vector2D(x=next_velocity_x, y=next_velocity_y)

        next_x = previous.pos.x + next_velocity_global.x * dt_sec
        next_y = previous.pos.y + next_velocity_global.y * dt_sec

        acceleration_x, acceleration_y, _ = global_to_ego_axis(
            dvx - yaw_rate * vy,
            dvy + yaw_rate * vx,
            0.0,
            0.0,
            -next_yaw,
        )
        acceleration_global = Vector2D(x=acceleration_x, y=acceleration_y)

        self.dynamic_state = EgoState(
            pos=Vector2D(x=next_x, y=next_y),
            velocity=Vector2D(x=next_vx, y=next_vy),
            acceleration=acceleration_global,
            yaw=next_yaw,
            steering_angle=steering_angle,
        )
        self.yaw_rate = next_yaw_rate

        self.ego_state = EgoState(
            pos=Vector2D(x=next_x, y=next_y),
            velocity=next_velocity_global,
            acceleration=acceleration_global,
            yaw=next_yaw,
            steering_angle=steering_angle,
        )

        return self.ego_state

    # Role: convert the internal body-frame state back into the public EgoState format.
    def to_ego_state(self, acceleration: Vector2D | None = None) -> EgoState:
        if acceleration is None:
            acceleration = Vector2D(x=0.0, y=0.0)

        velocity_x, velocity_y, _ = global_to_ego_axis(
            self.dynamic_state.velocity.x,
            self.dynamic_state.velocity.y,
            0.0,
            0.0,
            -self.dynamic_state.yaw,
        )

        return EgoState(
            pos=Vector2D(x=self.dynamic_state.pos.x, y=self.dynamic_state.pos.y),
            velocity=Vector2D(x=velocity_x, y=velocity_y),
            acceleration=acceleration,
            yaw=self.dynamic_state.yaw,
            steering_angle=self.ego_state.steering_angle,
        )

#  ─ MAIN ─

if __name__ == '__main__':

    ego = Vehicle(x=0, y=0, yaw=0, v=10)
    dt_ms = 50
    dt_sec = dt_ms / 1000.0

    # Total simulation time = sum of all phase durations
    total_time = sum(d for d, _ in OVERTAKE_PHASES)
    steps      = int(total_time / dt_sec)

    print("Running overtake simulation...")
    for step in range(steps):
        ego.Motion_control(dt_ms=dt_ms)
        if step % 40 == 0:        
            ego.Print_ego()

    ego.viz.show()
