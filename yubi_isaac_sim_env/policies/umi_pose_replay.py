"""Full 6D paired replay using one table transform for both operator hands.

Use the dual_franka_yubi_replay scene and replay head-camera calibration.
Only the initial right grasp registers the shared source frame to the cup.
Neither the left path nor source orientations are independently re-anchored.
"""
from __future__ import annotations

import numpy as np

from yubi_isaac_sim_env.policies.umi_paired_replay import (
    POSES, ANGLES, SOURCE_EPISODES, APPROACH_STEPS, SOURCE_FPS, POLICY_FPS,
    SIDES, _opening_progress, _physical_opening, _first_grasp_index,
)
from yubi_isaac_sim_env.umi_pose_mapping import source_to_tool, interpolate_orientation

POSE_COLUMNS = {"left": slice(2, 9), "right": slice(9, 16)}


def replay_poses(cup_position_m):
    mapped = {side: source_to_tool(POSES[:, POSE_COLUMNS[side]]) for side in SIDES}
    grasp = _first_grasp_index("right", _opening_progress("right"))
    offset = np.asarray(cup_position_m) + [0., 0., 0.0375] - mapped["right"][0][grasp]
    return {side: (xyz + offset, quat) for side, (xyz, quat) in mapped.items()}


def predict(observation: dict, step: int, episode: int) -> dict:
    if step != 0:
        raise RuntimeError("Full pose replay supplies one chunk; use its declared step count")
    mapped = replay_poses(observation["objects"]["cup"]["position_m"])
    opening = {side: _physical_opening(side) for side in SIDES}
    current = {side: observation["robots"][side]["tool_pose"] for side in SIDES}
    fractions = np.arange(1, APPROACH_STEPS + 1) / APPROACH_STEPS
    approach_q = {
        side: interpolate_orientation(current[side]["quaternion_wxyz"], mapped[side][1][0], fractions)
        for side in SIDES
    }
    waypoints = []
    for i, alpha in enumerate(fractions):
        waypoints.append({side: {
            "position_m": ((1 - alpha) * np.asarray(current[side]["position_m"]) + alpha * mapped[side][0][0]).tolist(),
            "quaternion_wxyz": approach_q[side][i].tolist(),
            "gripper_open_fraction": float((1 - alpha) + alpha * opening[side][0]),
        } for side in SIDES})
    # The approach ends at source frame 0. Each subsequent command is the
    # endpoint of the next 100 ms interval, so do not hold frame 0 for an
    # extra policy tick. This keeps rendered time aligned with source time.
    for frame in range(SOURCE_FPS // POLICY_FPS, len(POSES), SOURCE_FPS // POLICY_FPS):
        waypoints.append({side: {
            "position_m": mapped[side][0][frame].tolist(),
            "quaternion_wxyz": mapped[side][1][frame].tolist(),
            "gripper_open_fraction": float(opening[side][frame]),
        } for side in SIDES})
    return {"action_dt_s": 1 / POLICY_FPS, "waypoints": waypoints}
