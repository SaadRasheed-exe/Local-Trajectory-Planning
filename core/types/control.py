from dataclasses import dataclass


@dataclass(frozen=True)
class ControlCommand:
    """Canonical actuator output crossing the controller boundary.

    steer_rate is canonical because the plant integrates steering angle from
    a rate; implementations that natively compute an absolute steering angle
    derive the rate internally.
    """

    acceleration: float         # longitudinal acceleration [m/s²]
    steer_rate: float           # steering-angle rate [rad/s]

    emergency_stop: bool = False
