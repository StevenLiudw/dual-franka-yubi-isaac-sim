"""Replay a complete two-episode UMI cup task through both Panda arms.

The official recordings split the complete process into consecutive episodes:
the right hand places the cup on the plate, then the left hand returns it.  We
keep their original 30 Hz ordering and sample at the simulator's 10 Hz policy
rate. The inactive left arm stays clear until the right gripper releases the
cup, then joins its source path before its own first closing motion.

Required environment variable::

    UMI_REPLAY_PATH=/path/to/YUBI_Visualization/data/records/<uuid>.json

``UMI_REPLAY_EPISODES`` defaults to ``259632,259633``.  The first episode's
right-hand grasp is registered to the initial cup position.  The second
episode's left-hand grasp is registered to the plate.  A grasp is the first
sample whose normalized opening falls below 0.2; this is more faithful than
using the minimum-angle frame, which occurs after the object has been lifted.
Source orientations remain disabled until the glove-to-YUBI rotation is
calibrated.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from yubi_isaac_sim_env.umi_gripper_mapping import source_angles_to_open_fraction


SOURCE_FPS = 30
POLICY_FPS = 10
SIDES = ("left", "right")
POSITION_COLUMNS = {"left": slice(2, 5), "right": slice(9, 12)}
ANGLE_COLUMNS = {"left": 0, "right": 1}
ACTIVE_EPISODE_INDEX = {"right": 0, "left": 1}
_SIM_JAW = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())["yubi"]
SIM_JAW_CLOSED_RAD = float(_SIM_JAW["q_closed_rad"])
SIM_JAW_OPEN_RAD = float(_SIM_JAW["q_open_rad"])


def _load_recording() -> tuple[np.ndarray, np.ndarray, tuple[int, int], int]:
    source = os.environ.get("UMI_REPLAY_PATH")
    if not source:
        raise RuntimeError("UMI_REPLAY_PATH must point to a cached UMI record JSON")
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"UMI replay record does not exist: {path}")

    episode_text = os.environ.get("UMI_REPLAY_EPISODES", "259632,259633")
    try:
        episodes = tuple(int(value.strip()) for value in episode_text.split(","))
    except ValueError as exc:
        raise ValueError("UMI_REPLAY_EPISODES must contain two comma-separated integers") from exc
    if len(episodes) != 2 or episodes[0] == episodes[1]:
        raise ValueError("UMI_REPLAY_EPISODES must contain two distinct episode indices")

    approach_steps = int(os.environ.get("UMI_REPLAY_APPROACH_STEPS", "15"))
    if not 1 <= approach_steps <= 30:
        raise ValueError("UMI_REPLAY_APPROACH_STEPS must be in [1, 30]")

    raw = json.loads(path.read_text(encoding="utf-8"))
    all_poses = np.asarray(raw.get("poses"), dtype=np.float64)
    all_angles = np.asarray(raw.get("angles"), dtype=np.float64)
    if all_poses.ndim != 2 or all_poses.shape[1] != 16 or all_angles.shape != (len(all_poses), 2):
        raise ValueError("UMI replay JSON must contain poses[N,16] and angles[N,2]")

    pose_parts = []
    angle_parts = []
    for episode in episodes:
        mask = all_poses[:, 0].astype(np.int64) == episode
        poses, angles = all_poses[mask], all_angles[mask]
        if len(poses) < SOURCE_FPS or not np.isfinite(poses).all() or not np.isfinite(angles).all():
            raise ValueError(f"Episode {episode} is missing, too short, or contains non-finite values")
        frames = poses[:, 1].astype(np.int64)
        if not np.array_equal(frames, np.arange(len(frames))):
            raise ValueError(f"Episode {episode} frame indices are not contiguous from zero")
        pose_parts.append(poses)
        angle_parts.append(angles)

    return np.concatenate(pose_parts), np.concatenate(angle_parts), episodes, approach_steps


POSES, ANGLES, SOURCE_EPISODES, APPROACH_STEPS = _load_recording()


def _opening_progress(side: str) -> np.ndarray:
    # Relative progress is only an event detector for grasp/release anchors.
    # Actuator targets always use the independent, fixed aperture calibration.
    source = ANGLES[:, ANGLE_COLUMNS[side]]
    low, high = float(source.min()), float(source.max())
    if not np.isfinite([low, high]).all() or high - low < 0.05:
        raise ValueError(f"{side} gripper angle does not contain a usable open/close range")
    return np.clip((source - low) / (high - low), 0.0, 1.0)


def _physical_opening(side: str) -> np.ndarray:
    """Convert the already offset-corrected source angle with a fixed CAD map."""
    return source_angles_to_open_fraction(
        ANGLES[:, ANGLE_COLUMNS[side]], SIM_JAW_CLOSED_RAD, SIM_JAW_OPEN_RAD
    )


def _first_grasp_index(side: str, opening: np.ndarray) -> int:
    active_episode = SOURCE_EPISODES[ACTIVE_EPISODE_INDEX[side]]
    candidates = np.flatnonzero((POSES[:, 0].astype(np.int64) == active_episode) & (opening < 0.2))
    if not len(candidates):
        raise ValueError(f"{side} active episode never closes below 0.2")
    return int(candidates[0])


def _first_release_index(side: str, opening: np.ndarray, grasp_index: int) -> int:
    active_episode = SOURCE_EPISODES[ACTIVE_EPISODE_INDEX[side]]
    candidates = np.flatnonzero(
        (np.arange(len(opening)) > grasp_index)
        & (POSES[:, 0].astype(np.int64) == active_episode)
        & (opening > 0.8)
    )
    if not len(candidates):
        raise ValueError(f"{side} active episode never opens above 0.8 after grasp")
    return int(candidates[0])


def _command(positions: dict[str, np.ndarray], orientations: dict[str, list[float]],
             openings: dict[str, float]) -> dict:
    return {
        side: {
            "position_m": positions[side].tolist(),
            "quaternion_wxyz": orientations[side],
            "gripper_open_fraction": float(openings[side]),
        }
        for side in SIDES
    }


def predict(observation: dict, step: int, episode: int) -> dict:
    if step != 0:
        raise RuntimeError("The paired UMI replay is one preloaded chunk; run its declared step count")

    current = {
        side: np.asarray(observation["robots"][side]["tool_pose"]["position_m"], dtype=np.float64)
        for side in SIDES
    }
    orientations = {
        side: [float(value) for value in observation["robots"][side]["tool_pose"]["quaternion_wxyz"]]
        for side in SIDES
    }
    source_positions = {side: POSES[:, POSITION_COLUMNS[side]] for side in SIDES}
    progress = {side: _opening_progress(side) for side in SIDES}
    openings = {side: _physical_opening(side) for side in SIDES}

    cup = np.asarray(observation["objects"]["cup"]["position_m"], dtype=np.float64)
    plate = np.asarray(observation["objects"]["plate"]["position_m"], dtype=np.float64)
    grasp_height = np.array((0.0, 0.0, 0.0375), dtype=np.float64)
    anchors = {"right": cup + grasp_height, "left": plate + grasp_height}
    grasp_indices = {side: _first_grasp_index(side, progress[side]) for side in SIDES}
    mapped = {
        side: source_positions[side] - source_positions[side][grasp_indices[side]] + anchors[side]
        for side in SIDES
    }
    # The recorded left hand waits close to the plate during the first
    # episode. On two Panda arms that parked pose blocks the right arm's
    # post-release retreat. Keep the left arm at its safe reset position and
    # enter the original left path before its first source closure begins.
    right_release = _first_release_index("right", progress["right"], grasp_indices["right"])
    left_close = _first_grasp_index("left", progress["left"])
    left_join = min(left_close - 3, int(np.flatnonzero(POSES[:, 0] == SOURCE_EPISODES[1])[0]) + SOURCE_FPS)
    if left_join <= right_release:
        raise ValueError("Paired replay has no time for the left arm to clear the right retreat")
    for frame in range(left_join):
        alpha = np.clip((frame - right_release) / (left_join - right_release), 0.0, 1.0)
        mapped["left"][frame] = (1.0 - alpha) * current["left"] + alpha * mapped["left"][frame]

    first = {side: mapped[side][0] for side in SIDES}
    waypoints = []
    for index in range(1, APPROACH_STEPS + 1):
        alpha = index / APPROACH_STEPS
        waypoints.append(_command(
            {side: (1.0 - alpha) * current[side] + alpha * first[side] for side in SIDES},
            orientations,
            {side: (1.0 - alpha) + alpha * float(openings[side][0]) for side in SIDES},
        ))

    for frame in range(0, len(POSES), SOURCE_FPS // POLICY_FPS):
        waypoints.append(_command(
            {side: mapped[side][frame] for side in SIDES},
            orientations,
            {side: openings[side][frame] for side in SIDES},
        ))

    return {"action_dt_s": 1.0 / POLICY_FPS, "waypoints": waypoints}


__all__ = ["predict"]
