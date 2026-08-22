from __future__ import annotations

import casadi as ca
import math
import numpy as np

from typing import Optional, Tuple, List
from omegaconf import DictConfig

from models.models import (
    EgoStateStamped,
    EgoState,
    Trajectory,
    Vector2D
)
from utils.helper import get_magnitude, get_signed_magnitude, get_vector


class MPCController:
    """
    Nonlinear MPC based on a kinematic bicycle model.

    State:
        [x, y, yaw, v, delta]

    Control:
        [acceleration, steering_rate]
    """

    def __init__(
        self,
        vehicle_params: DictConfig,
        controller_cfg: DictConfig,
    ):

        self.L = vehicle_params.wheel_base

        self.dt = controller_cfg.dt / 1000.0
        self.N = int(math.ceil(controller_cfg.horizon / controller_cfg.dt))

        self.max_acc = vehicle_params.max_acceleration
        self.max_dec = -abs(vehicle_params.max_deceleration)

        self.max_delta = vehicle_params.max_steer
        self.max_delta_rate = vehicle_params.max_steer_rate

        # Tracking weights
        self.Qx = controller_cfg.get("Q_x", 5.0)
        self.Qy = controller_cfg.get("Q_y", 5.0)
        self.Qyaw = controller_cfg.get("Q_yaw", 10.0)
        self.Qv = controller_cfg.get("Q_v", 1.0)
        self.Qdelta = controller_cfg.get("Q_delta", 0.1)

        # Control effort penalties
        self.Ra = controller_cfg.get("R_a", 0.1)
        self.Rsr = controller_cfg.get("R_sr", 0.1)

        # Lateral acceleration shaping
        self.max_a_lat = controller_cfg.get("max_a_lat", 3.0)
        self.Ralat = controller_cfg.get("R_alat", 5.0)

        # Input rate penalties
        self.Rda = controller_cfg.get("R_da", 1.0)
        self.Rdsr = controller_cfg.get("R_dsr", 1.0)

        self._build_solver()

    def _build_solver(self):

        opti = ca.Opti()

        nx = 5
        nu = 2

        X = opti.variable(nx, self.N + 1)
        U = opti.variable(nu, self.N)

        X0 = opti.parameter(nx)

        REF = opti.parameter(nx, self.N + 1)

        cost = 0

        for k in range(self.N):

            x = X[:, k]
            u = U[:, k]

            px = x[0]
            py = x[1]
            yaw = x[2]
            v = x[3]
            delta = x[4]

            a = u[0]
            delta_rate = u[1]

            xdot = ca.vertcat(
                v * ca.cos(yaw),
                v * ca.sin(yaw),
                v / self.L * ca.tan(delta),
                a,
                delta_rate,
            )

            x_next = x + self.dt * xdot

            opti.subject_to(X[:, k + 1] == x_next)

            e = X[:, k] - REF[:, k]

            yaw_error = ca.atan2(
                ca.sin(X[2, k] - REF[2, k]),
                ca.cos(X[2, k] - REF[2, k]),
            )

            cost += (
                self.Qx * e[0] ** 2
                + self.Qy * e[1] ** 2
                + self.Qyaw * yaw_error ** 2
                + self.Qv * e[3] ** 2
                + self.Qdelta * e[4] ** 2
            )

            cost += (
                self.Ra * a**2
                + self.Rsr * delta_rate**2
            )

            a_lat = (v**2 / self.L) * ca.tan(delta)

            cost += self.Ralat * a_lat**2

            opti.subject_to(a_lat <= self.max_a_lat)
            opti.subject_to(a_lat >= -self.max_a_lat)

            if k > 0:

                du = U[:, k] - U[:, k - 1]

                cost += (
                    self.Rda * du[0] ** 2
                    + self.Rdsr * du[1] ** 2
                )

            opti.subject_to(a <= self.max_acc)
            opti.subject_to(a >= self.max_dec)

            opti.subject_to(
                delta_rate <= self.max_delta_rate
            )

            opti.subject_to(
                delta_rate >= -self.max_delta_rate
            )

            opti.subject_to(
                X[4, k] <= self.max_delta
            )

            opti.subject_to(
                X[4, k] >= -self.max_delta
            )

        terminal_error = X[:, self.N] - REF[:, self.N]

        terminal_yaw_error = ca.atan2(
            ca.sin(X[2, self.N] - REF[2, self.N]),
            ca.cos(X[2, self.N] - REF[2, self.N]),
        )

        cost += (
            self.Qx * terminal_error[0] ** 2
            + self.Qy * terminal_error[1] ** 2
            + self.Qyaw * terminal_yaw_error ** 2
            + self.Qv * terminal_error[3] ** 2
            + self.Qdelta * terminal_error[4] ** 2
        )

        opti.subject_to(
            X[4, self.N] <= self.max_delta
        )

        opti.subject_to(
            X[4, self.N] >= -self.max_delta
        )

        opti.subject_to(X[:, 0] == X0)

        opti.minimize(cost)

        options = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
        }

        opti.solver("ipopt", options)

        self.opti = opti
        self.X0 = X0
        self.REF = REF
        self.U = U

    def _interpolate_trajectory(self, trajectory: Trajectory, target_time_ms: int) -> EgoState:
        """
        Interpolates the trajectory to return an exact EgoState for a given target timestamp.
        """
        states = trajectory.states
        
        # 1. Boundary check: Return first or last state if outside of trajectory timeframe
        if target_time_ms <= states[0].timestamp:
            return states[0].state
        if target_time_ms >= states[-1].timestamp:
            return states[-1].state
            
        # 2. Find the correct time slot for interpolation
        for i in range(len(states) - 1):
            s1 = states[i]
            s2 = states[i + 1]
            
            if s1.timestamp <= target_time_ms <= s2.timestamp:
                dt_total = s2.timestamp - s1.timestamp
                
                # Safety check to avoid division by zero
                if dt_total == 0:
                    return s1.state
                    
                ratio = (target_time_ms - s1.timestamp) / float(dt_total)
                
                # Spatial interpolation (x, y)
                x = s1.state.pos.x + ratio * (s2.state.pos.x - s1.state.pos.x)
                y = s1.state.pos.y + ratio * (s2.state.pos.y - s1.state.pos.y)
                
                # Yaw interpolation (with modulo to prevent 360-degree jumps)
                yaw_diff = (s2.state.yaw - s1.state.yaw + math.pi) % (2 * math.pi) - math.pi
                yaw = s1.state.yaw + ratio * yaw_diff
                
                # Velocity interpolation using signed magnitude
                v1 = get_signed_magnitude(s1.state.velocity, s1.state.yaw)
                v2 = get_signed_magnitude(s2.state.velocity, s2.state.yaw)
                v = v1 + ratio * (v2 - v1)
                
                return EgoState(
                    pos=Vector2D(x=x, y=y),
                    yaw=yaw,
                    velocity=get_vector(v, yaw),
                    acceleration=Vector2D(x=0.0, y=0.0),
                    steering_angle=0.0
                )
                
        return states[-1].state

    def compute_control(
        self,
        ego_state: EgoStateStamped,
        trajectory: Trajectory,
        first_only: bool = True,
    ) -> List[Tuple[float, float]]:

        px = ego_state.state.pos.x
        py = ego_state.state.pos.y

        yaw = ego_state.state.yaw
        v = get_signed_magnitude(
            ego_state.state.velocity, ego_state.state.yaw
        )

        delta = getattr(
            ego_state.state,
            "steering_angle",
            0.0,
        )

        x0 = np.array([
            px,
            py,
            yaw,
            v,
            delta,
        ])

        ref = np.zeros((5, self.N + 1))
        
        # Base time of the ego vehicle to start the prediction
        current_time_ms = ego_state.timestamp
        dt_ms = int(self.dt * 1000)

        for k in range(self.N + 1):
            # Calculate the exact target time for the current MPC node
            target_time_ms = current_time_ms + k * dt_ms
            
            # Interpolate the trajectory state for the target time
            s = self._interpolate_trajectory(trajectory, target_time_ms)

            ref[:, k] = [
                s.pos.x,
                s.pos.y,
                s.yaw,
                get_signed_magnitude(s.velocity, s.yaw),
                0.0,
            ]

        self.opti.set_value(
            self.X0,
            x0,
        )

        self.opti.set_value(
            self.REF,
            ref,
        )

        sol = self.opti.solve()
        
        if first_only:
            u0 = sol.value(
                self.U[:, 0]
            )

            acceleration = float(u0[0])
            steering_rate = float(u0[1])

            return [(
                acceleration,
                steering_rate,
            )]
        
        else:

            u = []
            
            for i in range(self.N):
                ui = sol.value(self.U[:, i])
                acc, steer = float(ui[0]), float(ui[1])
                u.append((acc, steer))
            
            return u