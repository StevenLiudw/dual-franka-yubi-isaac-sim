"""Opt-in left second-episode grasp-height trial without moving either base.

The left TCP gains a smooth world-Z offset by its first closure; both robot
mounts, the complete right trajectory, and the replay time axis stay intact.
An optional extra left jaw closure permits a separate height-plus-aperture
diagnostic without changing the shared gripper mapping.
"""

import os

import numpy as np

from yubi_isaac_sim_env.policies.umi_jaw_bias_replay import shift_fraction
from yubi_isaac_sim_env.policies.umi_left_second_offset_replay import (
    APPROACH_STEPS, LEFT_CLOSE_FRAME, POLICY_FPS, POSES, SECOND_START_FRAME,
    SOURCE_FPS, predict as spacing_predict,
)


def predict(observation, step, episode):
    action = spacing_predict(observation, step, episode)
    height_mm = float(os.environ.get("UMI_LEFT_SECOND_HEIGHT_MM", "10"))
    extra_close_rad = float(os.environ.get("UMI_LEFT_SECOND_EXTRA_CLOSE_RAD", "0"))
    close_start_frame = int(os.environ.get("UMI_LEFT_SECOND_EXTRA_CLOSE_START_FRAME", SECOND_START_FRAME))
    close_end_frame = int(os.environ.get("UMI_LEFT_SECOND_EXTRA_CLOSE_END_FRAME", LEFT_CLOSE_FRAME))
    if not np.isfinite(height_mm) or not 0 <= height_mm <= 30:
        raise ValueError("UMI_LEFT_SECOND_HEIGHT_MM must be finite and in [0, 30]")
    if not np.isfinite(extra_close_rad) or not 0 <= extra_close_rad <= 0.08:
        raise ValueError("UMI_LEFT_SECOND_EXTRA_CLOSE_RAD must be finite and in [0, 0.08]")
    if not SECOND_START_FRAME <= close_start_frame < close_end_frame < len(POSES):
        raise ValueError("Extra close ramp must lie within the second source episode")

    stride = SOURCE_FPS // POLICY_FPS
    for waypoint_index, waypoint in enumerate(action["waypoints"][APPROACH_STEPS:]):
        source_frame = (waypoint_index + 1) * stride
        if source_frame < SECOND_START_FRAME:
            continue
        height_progress = np.clip(
            (source_frame - SECOND_START_FRAME) / (LEFT_CLOSE_FRAME - SECOND_START_FRAME),
            0.0, 1.0,
        )
        height_blend = height_progress * height_progress * (3.0 - 2.0 * height_progress)
        left = waypoint["left"]
        left["position_m"][2] += height_mm * 0.001 * height_blend
        if extra_close_rad and source_frame >= close_start_frame:
            close_progress = np.clip(
                (source_frame - close_start_frame) / (close_end_frame - close_start_frame),
                0.0, 1.0,
            )
            close_blend = close_progress * close_progress * (3.0 - 2.0 * close_progress)
            left["gripper_open_fraction"] = shift_fraction(
                left["gripper_open_fraction"], -extra_close_rad * close_blend
            )
    return action
