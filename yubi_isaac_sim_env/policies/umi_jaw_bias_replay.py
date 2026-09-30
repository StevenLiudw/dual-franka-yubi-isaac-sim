"""Replay-only fixed jaw zero offset; every source-to-source delta is intact.

The output still uses the normal single-fraction gripper API. The offset is
applied to every waypoint, including the approach, with no clipping.
"""
import os

import numpy as np

from yubi_isaac_sim_env.policies.umi_fixed_anchor_replay import predict as base_predict
from yubi_isaac_sim_env.policies.umi_paired_replay import SIM_JAW_CLOSED_RAD, SIM_JAW_OPEN_RAD


DEFAULT_JAW_BIAS_RAD = -0.02


def shift_fraction(open_fraction: float, bias_rad: float) -> float:
    closed, opened = SIM_JAW_CLOSED_RAD, SIM_JAW_OPEN_RAD
    if not np.isfinite([open_fraction, bias_rad]).all():
        raise ValueError("Jaw fraction and bias must be finite")
    target = closed + (opened - closed) * open_fraction + bias_rad
    if target < closed - 1e-9 or target > opened + 1e-9:
        raise ValueError(f"Jaw bias target {target:.6f} rad exceeds [{closed}, {opened}] rad; refusing to clip deltas")
    return float((target - closed) / (opened - closed))


def predict(observation, step, episode):
    action = base_predict(observation, step, episode)
    bias_rad = float(os.environ.get("UMI_JAW_BIAS_RAD", DEFAULT_JAW_BIAS_RAD))
    for waypoint in action["waypoints"]:
        for side in ("left", "right"):
            command = waypoint[side]
            command["gripper_open_fraction"] = shift_fraction(command["gripper_open_fraction"], bias_rad)
    return action
