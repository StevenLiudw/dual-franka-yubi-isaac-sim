"""Bounded dual-Franka push demonstration for the synthetic cup/plate scene.

Use with ``python -m yubi_isaac_sim_env.run --policy-script``. Episode 0 guides
the right open gripper toward the cup; episode 1 guides the left open gripper
toward the plate. Both targets are fixed relative to each episode's initial
sampled object pose. This demonstrates joint control and contact motion; it
does not grasp or place either object.

The observation supplies the live tool world position and 3x7 translation
Jacobian. A damped least-squares step produces seven arm joint targets. The
environment enforces its articulation limits after this policy's per-step
increment bound.
"""

from __future__ import annotations

import numpy as np


_ARM_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
_GRIPPER_OPEN_FRACTION = 1.0
_DAMPING = 0.04
_GAIN = 0.7
_MAX_JOINT_INCREMENT_RAD = 0.08
_WAYPOINT_TOLERANCE_M = 0.01
_STATE: dict | None = None


def _initial_state(observation: dict, episode: int) -> dict:
    if episode == 0:
        side, object_name, xy_key = "right", "cup", "cup_xy"
        lateral = 1.0
        # Same overhead, lowering, and side sweep used by the bounded GPU cup
        # probe. The final goal stays tied to the reset pose after contact.
        offsets = ((0.08, 0.25), (0.08, 0.11), (0.0, 0.09))
    elif episode == 1:
        side, object_name, xy_key = "left", "plate", "tray_xy"
        lateral = -1.0
        # The plate's wider rim calls for a slightly wider approach and a
        # shallower final height than the cup.
        offsets = ((0.10, 0.25), (0.10, 0.10), (0.006, 0.07))
    else:
        raise ValueError("push_demo supports episode 0 (cup) and episode 1 (plate) only")

    scenario = observation["scenario"]
    xy = np.asarray(scenario[xy_key], dtype=float)
    z = float(observation["objects"][object_name]["position_m"][2])
    if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(z):
        raise ValueError("Initial sampled object pose is missing or non-finite")
    waypoints = [np.array((xy[0], xy[1] + lateral * dy, z + dz), dtype=float) for dy, dz in offsets]
    return {
        "episode": episode,
        "side": side,
        "waypoints": waypoints,
        "stage": 0,
        "stage_steps": 0,
    }


def act(observation: dict, step: int, episode: int) -> dict:
    """Return an open-gripper joint action for one bounded push episode.

    The policy is stateful only across successive calls within an episode;
    ``step == 0`` rebuilds its targets from the new reset observation.
    """
    global _STATE
    if step == 0 or _STATE is None or _STATE["episode"] != episode:
        _STATE = _initial_state(observation, episode)

    side = _STATE["side"]
    robot = observation["robots"][side]
    names = tuple(robot["joint_names"])
    if not all(name in names for name in _ARM_NAMES):
        raise ValueError(f"Missing arm joints: {names}")
    arm_indices = [names.index(name) for name in _ARM_NAMES]
    tool = np.asarray(robot["tool_pose"]["position_m"], dtype=float)
    q = np.asarray([robot["joint_positions"][i] for i in arm_indices], dtype=float)
    jacobian = np.asarray(robot["tool_translation_jacobian"], dtype=float)
    if tool.shape != (3,) or q.shape != (7,) or jacobian.shape != (3, 7):
        raise ValueError("Expected tool position (3), arm joints (7), and Jacobian (3, 7)")
    if not (np.isfinite(tool).all() and np.isfinite(q).all() and np.isfinite(jacobian).all()):
        raise ValueError("Non-finite tool, joint, or Jacobian observation")

    limits = (40, 50, 50)
    while _STATE["stage"] < len(_STATE["waypoints"]):
        stage = _STATE["stage"]
        error = _STATE["waypoints"][stage] - tool
        if _STATE["stage_steps"] >= limits[stage] or (
            _STATE["stage_steps"] > 0 and np.linalg.norm(error) < _WAYPOINT_TOLERANCE_M
        ):
            _STATE["stage"] += 1
            _STATE["stage_steps"] = 0
            continue
        break

    if _STATE["stage"] >= len(_STATE["waypoints"]):
        return {side: {"gripper_open_fraction": _GRIPPER_OPEN_FRACTION}}

    error = _STATE["waypoints"][_STATE["stage"]] - tool
    normal = jacobian @ jacobian.T + (_DAMPING**2) * np.eye(3)
    increment = _GAIN * (jacobian.T @ np.linalg.solve(normal, error))
    increment = np.clip(increment, -_MAX_JOINT_INCREMENT_RAD, _MAX_JOINT_INCREMENT_RAD)
    targets = q + increment
    if not np.isfinite(targets).all():
        raise ValueError("Damped least-squares produced non-finite joint targets")
    _STATE["stage_steps"] += 1
    return {
        side: {
            "arm_joint_targets_rad": targets.tolist(),
            "gripper_open_fraction": _GRIPPER_OPEN_FRACTION,
        }
    }
