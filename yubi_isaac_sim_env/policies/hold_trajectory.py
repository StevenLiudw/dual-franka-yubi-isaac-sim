"""Minimal trajectory-policy example: command both tools to hold their poses.

Run with ``--trajectory-policy-script yubi_isaac_sim_env/policies/hold_trajectory.py``.
Replace the body of ``predict`` with checkpoint inference after converting the
checkpoint's output into absolute world-frame YUBI tool poses at 10 Hz.
"""

from __future__ import annotations


def predict(observation: dict, step: int, episode: int) -> dict:
    waypoint = {}
    for side in ("left", "right"):
        robot = observation["robots"][side]
        pose = robot["tool_pose"]
        waypoint[side] = {
            "position_m": list(pose["position_m"]),
            "quaternion_wxyz": list(pose["quaternion_wxyz"]),
            "gripper_open_fraction": min(1.0, max(0.0, float(robot["gripper_open_fraction"]))),
        }
    return {"action_dt_s": 0.1, "waypoints": [waypoint]}
