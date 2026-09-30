"""Robot-independent conditioning for streamed joint-position references.

The model-facing trajectory contract deliberately stops at desired tool poses.
After IK, a robot driver still needs a continuous joint reference.  This
module implements that boundary in NumPy only so simulation and hardware
drivers can use the same units and tuning without importing Isaac Sim.

This is a reference governor, not a certified safety controller.  A Franka
hardware driver must still run its normal real-time safety/rate limiter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


FRANKA_LIMIT_EPS = 1e-3


def _positive_vector(value: Iterable[float], size: int, name: str) -> np.ndarray:
    vector = np.asarray(tuple(value), dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all() or np.any(vector <= 0.0):
        raise ValueError(f"{name} must contain {size} positive finite values")
    return vector


@dataclass(frozen=True)
class JointReferenceProfile:
    """Physical limits and response shape for a joint reference governor."""

    name: str
    max_velocity_rad_s: tuple[float, ...]
    max_acceleration_rad_s2: tuple[float, ...]
    max_jerk_rad_s3: tuple[float, ...]
    position_gain_s_inv: float
    velocity_limit_model: str = "symmetric"
    servo_period_s: float = 1e-3

    def validate(self, size: int) -> None:
        _positive_vector(self.max_velocity_rad_s, size, "max_velocity_rad_s")
        _positive_vector(self.max_acceleration_rad_s2, size, "max_acceleration_rad_s2")
        _positive_vector(self.max_jerk_rad_s3, size, "max_jerk_rad_s3")
        if not np.isfinite(self.position_gain_s_inv) or self.position_gain_s_inv <= 0.0:
            raise ValueError("position_gain_s_inv must be positive and finite")
        if self.velocity_limit_model != "symmetric":
            raise ValueError("unknown velocity_limit_model")
        if not np.isfinite(self.servo_period_s) or self.servo_period_s <= 0.0:
            raise ValueError("servo_period_s must be positive and finite")

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "max_velocity_rad_s": list(self.max_velocity_rad_s),
            "max_acceleration_rad_s2": list(self.max_acceleration_rad_s2),
            "max_jerk_rad_s3": list(self.max_jerk_rad_s3),
            "position_gain_s_inv": self.position_gain_s_inv,
            "velocity_limit_model": self.velocity_limit_model,
            "servo_period_s": self.servo_period_s,
        }


FRANKA_PANDA_INTERFACE_PROFILE = JointReferenceProfile(
    name="franka-panda-interface",
    # Official Panda FCI joint trajectory limits. Subtract the same numerical
    # epsilon used by libfranka so a command remains strictly inside the
    # documented necessary inequalities without changing its time scale.
    max_velocity_rad_s=(
        2.175 - FRANKA_LIMIT_EPS,
        2.175 - FRANKA_LIMIT_EPS,
        2.175 - FRANKA_LIMIT_EPS,
        2.175 - FRANKA_LIMIT_EPS,
        2.610 - FRANKA_LIMIT_EPS,
        2.610 - FRANKA_LIMIT_EPS,
        2.610 - FRANKA_LIMIT_EPS,
    ),
    max_acceleration_rad_s2=(
        15.0 - FRANKA_LIMIT_EPS,
        7.5 - FRANKA_LIMIT_EPS,
        10.0 - FRANKA_LIMIT_EPS,
        12.5 - FRANKA_LIMIT_EPS,
        15.0 - FRANKA_LIMIT_EPS,
        20.0 - FRANKA_LIMIT_EPS,
        20.0 - FRANKA_LIMIT_EPS,
    ),
    max_jerk_rad_s3=(
        7500.0 - FRANKA_LIMIT_EPS,
        3750.0 - FRANKA_LIMIT_EPS,
        5000.0 - FRANKA_LIMIT_EPS,
        6250.0 - FRANKA_LIMIT_EPS,
        7500.0 - FRANKA_LIMIT_EPS,
        10000.0 - FRANKA_LIMIT_EPS,
        10000.0 - FRANKA_LIMIT_EPS,
    ),
    # The 10 Hz waypoint timing remains authoritative. This gain removes the
    # previous 125 ms artificial response lag; velocity/acceleration/jerk are
    # still clipped by the physical envelopes above on every 1 ms tick.
    position_gain_s_inv=30.0,
    velocity_limit_model="symmetric",
    servo_period_s=1e-3,
)

JOINT_REFERENCE_PROFILES = {
    FRANKA_PANDA_INTERFACE_PROFILE.name: FRANKA_PANDA_INTERFACE_PROFILE,
}


def get_joint_reference_profile(name: str) -> JointReferenceProfile | None:
    """Return a named profile; ``direct`` disables reference conditioning."""
    if name == "direct":
        return None
    try:
        return JOINT_REFERENCE_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(("direct", *JOINT_REFERENCE_PROFILES))
        raise ValueError(f"Unknown joint command profile {name!r}; choose {choices}") from exc


class JointReferenceGovernor:
    """Generate a smooth position reference at a fixed servo period.

    The state is the *last commanded* position, velocity, and acceleration,
    mirroring the quantities used by hardware-side rate limiters. A
    proportional position error generates a damped target velocity; the
    velocity is then limited with the same safe-acceleration-window structure
    used by libfranka's scalar rate limiter. Call :meth:`update` once per
    servo period.
    """

    def __init__(self, profile: JointReferenceProfile, *, dt_s: float, size: int = 7) -> None:
        profile.validate(size)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        self.profile = profile
        self.dt_s = float(dt_s)
        self.size = int(size)
        self.max_velocity = _positive_vector(profile.max_velocity_rad_s, size, "max_velocity_rad_s")
        self.max_acceleration = _positive_vector(
            profile.max_acceleration_rad_s2, size, "max_acceleration_rad_s2"
        )
        self.max_jerk = _positive_vector(profile.max_jerk_rad_s3, size, "max_jerk_rad_s3")
        self.position: np.ndarray | None = None
        self.velocity: np.ndarray | None = None
        self.acceleration: np.ndarray | None = None

    def velocity_limits(self, position_rad: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        return -self.max_velocity, self.max_velocity

    def reset(self, position_rad: Iterable[float], velocity_rad_s: Iterable[float] | None = None) -> None:
        position = np.asarray(tuple(position_rad), dtype=np.float64)
        velocity = np.zeros(self.size, dtype=np.float64) if velocity_rad_s is None else np.asarray(
            tuple(velocity_rad_s), dtype=np.float64
        )
        if position.shape != (self.size,) or not np.isfinite(position).all():
            raise ValueError(f"position_rad must contain {self.size} finite values")
        if velocity.shape != (self.size,) or not np.isfinite(velocity).all():
            raise ValueError(f"velocity_rad_s must contain {self.size} finite values")
        self.position = position.copy()
        lower_velocity, upper_velocity = self.velocity_limits(position)
        self.velocity = np.clip(velocity, lower_velocity, upper_velocity)
        self.acceleration = np.zeros(self.size, dtype=np.float64)

    def update(self, target_position_rad: Iterable[float]) -> np.ndarray:
        target = np.asarray(tuple(target_position_rad), dtype=np.float64)
        if target.shape != (self.size,) or not np.isfinite(target).all():
            raise ValueError(f"target_position_rad must contain {self.size} finite values")
        if self.position is None or self.velocity is None or self.acceleration is None:
            raise RuntimeError("Call reset before update")

        position_error = target - self.position
        # Request the original target as quickly as the Panda envelope allows,
        # while reserving braking distance for acceleration and discrete jerk.
        # This prevents high response gain from crossing a waypoint and
        # oscillating; it does not rescale the policy timeline.
        stopping_velocity = np.sqrt(
            2.0 * 0.8 * self.max_acceleration * np.abs(position_error)
        )
        commanded_velocity = np.sign(position_error) * np.minimum(
            self.profile.position_gain_s_inv * np.abs(position_error),
            stopping_velocity,
        )
        commanded_jerk = (
            (commanded_velocity - self.velocity) / self.dt_s - self.acceleration
        ) / self.dt_s
        acceleration = self.acceleration + np.clip(
            commanded_jerk, -self.max_jerk, self.max_jerk
        ) * self.dt_s

        # These velocity-dependent bounds leave enough room to reduce the
        # acceleration under the configured jerk limit before reaching the
        # velocity ceiling. This is the key detail that avoids a hidden jerk
        # discontinuity at a hard velocity clip.
        lower_velocity, upper_velocity = self.velocity_limits(self.position)
        safe_max_acceleration = np.minimum(
            (self.max_jerk / self.max_acceleration) * (upper_velocity - self.velocity),
            self.max_acceleration,
        )
        safe_min_acceleration = np.maximum(
            (self.max_jerk / self.max_acceleration) * (lower_velocity - self.velocity),
            -self.max_acceleration,
        )
        acceleration = np.clip(acceleration, safe_min_acceleration, safe_max_acceleration)
        velocity = self.velocity + acceleration * self.dt_s
        position = self.position + velocity * self.dt_s

        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        return position.copy()


__all__ = [
    "FRANKA_PANDA_INTERFACE_PROFILE",
    "FRANKA_LIMIT_EPS",
    "JOINT_REFERENCE_PROFILES",
    "JointReferenceGovernor",
    "JointReferenceProfile",
    "get_joint_reference_profile",
]
