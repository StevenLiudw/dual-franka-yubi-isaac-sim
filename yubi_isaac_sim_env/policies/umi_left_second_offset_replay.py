"""Isolated second-episode left TCP spacing experiment.

Move only the left TCP in +world Y (away from the right TCP at the episode
boundary). Ramp the offset from the second episode's first frame to the left
first-close frame; leave all orientations, right-arm targets and grippers intact.
This is a visualization/contact trial, not a calibrated source-data replay.
"""

import os

import numpy as np

from yubi_isaac_sim_env.policies.umi_jaw_bias_replay import predict as base_predict
from yubi_isaac_sim_env.policies.umi_paired_replay import (
    APPROACH_STEPS, POLICY_FPS, POSES, SOURCE_EPISODES, SOURCE_FPS,
    _first_grasp_index, _opening_progress,
)


SECOND_START_FRAME = int(np.flatnonzero(POSES[:, 0] == SOURCE_EPISODES[1])[0])
LEFT_CLOSE_FRAME = _first_grasp_index("left", _opening_progress("left"))


def predict(observation, step, episode):
    action = base_predict(observation, step, episode)
    millimeters = float(os.environ.get("UMI_LEFT_SECOND_OFFSET_MM", "30"))
    if not np.isfinite(millimeters) or not 0 <= millimeters <= 80:
        raise ValueError("UMI_LEFT_SECOND_OFFSET_MM must be finite and in [0, 80]")
    offset_m = millimeters / 1000.0
    stride = SOURCE_FPS // POLICY_FPS
    for waypoint_index, waypoint in enumerate(action["waypoints"][APPROACH_STEPS:]):
        source_frame = (waypoint_index + 1) * stride
        if source_frame < SECOND_START_FRAME:
            continue
        progress = np.clip(
            (source_frame - SECOND_START_FRAME) / (LEFT_CLOSE_FRAME - SECOND_START_FRAME),
            0.0, 1.0,
        )
        blend = progress * progress * (3.0 - 2.0 * progress)
        waypoint["left"]["position_m"][1] += offset_m * blend
    return action
