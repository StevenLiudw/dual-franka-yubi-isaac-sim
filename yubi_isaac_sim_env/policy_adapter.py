"""Model-independent 6D YUBI trajectory adapter for the 10 Hz environment.

Checkpoint-specific code supplies absolute **world-frame** YUBI tool poses.
This module validates those waypoints, buffers an action chunk, and converts
the next waypoint to the environment's dual-arm joint-position command. It
uses NumPy only and can be tested without starting Isaac Sim.

The controller previews every pose in a submitted chunk.  A natural cubic
spline supplies continuous Cartesian waypoint velocities to bounded
resolved-rate IK, while the selected robot reference governor supplies bounded
joint velocity, acceleration, and jerk.  It is still not a collision-aware
path planner.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .interarm_guard import limit_interarm_motion


ARM_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))
SIDES = frozenset(("left", "right"))
ARM_COMMAND_KEYS = frozenset(("position_m", "quaternion_wxyz", "gripper_open_fraction"))
CHUNK_KEYS = frozenset(("action_dt_s", "waypoints", "execute_steps"))

TRAJECTORY_EXECUTOR_PROFILES = {
    "default": {},
    "franka-transfer": {
        "damping": 0.08,
        "position_gain": 0.6,
        "orientation_gain": 0.4,
        "max_position_step_m": 0.05,
        "max_orientation_step_rad": 0.22,
        "max_joint_step_rad": 0.065,
    },
    "franka-transfer-smooth": {
        "damping": 0.12,
        "position_gain": 0.5,
        "orientation_gain": 0.25,
        "max_position_step_m": 0.04,
        "max_orientation_step_rad": 0.15,
        "max_joint_step_rad": 0.05,
    },
    "franka-lookahead": {
        # Time-faithful tracking: on the nominal path, 0.7 of the next 10 Hz
        # displacement comes from pose error and 0.3 from whole-chunk preview.
        # Their sum is 1.0, avoiding both intentional slowdown and the doubled
        # step that results from full feedback plus full feed-forward.
        "damping": 0.08,
        "position_gain": 0.7,
        "orientation_gain": 0.7,
        "max_position_step_m": 0.06,
        "max_orientation_step_rad": 0.3,
        "max_joint_step_rad": 0.08,
        "lookahead_feedforward_gain": 0.3,
    },
    "panda-pose-replay": {
        # Full 6D replay exposes the old 0.08 rad/100 ms implementation cap
        # during rapid wrist retreats. Use the slowest Panda joint's nominal
        # 2.174 rad/s bound at 10 Hz; the servo governor enforces each joint's
        # velocity, acceleration and jerk limits at its own period.
        "damping": 0.08,
        "position_gain": 0.7,
        "orientation_gain": 0.7,
        "max_position_step_m": 0.10,
        "max_orientation_step_rad": 0.3,
        "max_joint_step_rad": 0.2174,
        "lookahead_feedforward_gain": 0.3,
    },
}


def trajectory_executor_profile(name: str) -> dict[str, float]:
    """Return one shared simulation/hardware differential-IK configuration."""
    try:
        return dict(TRAJECTORY_EXECUTOR_PROFILES[name])
    except KeyError as exc:
        choices = ", ".join(TRAJECTORY_EXECUTOR_PROFILES)
        raise ValueError(f"Unknown trajectory executor profile {name!r}; choose {choices}") from exc


class TrajectoryFormatError(ValueError):
    """A model output or observation does not satisfy the trajectory contract."""


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TrajectoryFormatError(f"{name} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TrajectoryFormatError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise TrajectoryFormatError(f"{name} must be a positive finite number")
    return number


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TrajectoryFormatError(f"{name} must contain {size} finite numbers") from exc
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise TrajectoryFormatError(f"{name} must contain {size} finite numbers")
    return vector


def _matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TrajectoryFormatError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix") from exc
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise TrajectoryFormatError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix")
    return matrix


def _quaternion(value: Any, name: str, *, target: bool) -> np.ndarray:
    quaternion = _vector(value, 4, name)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8 or (target and abs(norm - 1.0) > 0.05):
        raise TrajectoryFormatError(f"{name} must be a nonzero unit quaternion in wxyz order")
    return quaternion / norm


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dtype=np.float64,
    )


def _world_rotation_error(goal: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Shortest SO(3) log of goal * inverse(current), in world axes."""
    error = _quat_multiply(goal, current * np.array((1.0, -1.0, -1.0, -1.0)))
    if error[0] < 0:
        error = -error  # q and -q encode the same physical orientation.
    sin_half = float(np.linalg.norm(error[1:]))
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(sin_half, max(0.0, float(error[0])))
    return error[1:] * (angle / sin_half)


def _cap_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector * (maximum / norm) if norm > maximum else vector


def _natural_cubic_tangents(values: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    """Return C2 natural-cubic derivatives at every time-stamped knot.

    Solving the coupled tridiagonal system means the derivative at an
    executable waypoint uses the entire submitted chunk, including waypoints
    beyond ``execute_steps``.  This is intentionally a preview operation, not
    a causal filter applied after commands reach the robot.
    """
    count = len(values)
    if count == 1:
        return np.zeros_like(values)
    intervals = np.diff(times_s)
    if times_s.shape != (count,) or np.any(intervals <= 0):
        raise TrajectoryFormatError("spline waypoint times must be strictly increasing")
    matrix = np.zeros((count, count), dtype=np.float64)
    rhs = np.zeros_like(values, dtype=np.float64)
    matrix[0, 0:2] = (2.0, 1.0)
    matrix[-1, -2:] = (1.0, 2.0)
    rhs[0] = 3.0 * (values[1] - values[0]) / intervals[0]
    rhs[-1] = 3.0 * (values[-1] - values[-2]) / intervals[-1]
    for index in range(1, count - 1):
        previous_h, next_h = intervals[index - 1], intervals[index]
        matrix[index, index - 1:index + 2] = (
            next_h,
            2.0 * (previous_h + next_h),
            previous_h,
        )
        rhs[index] = 3.0 * (
            next_h * (values[index] - values[index - 1]) / previous_h
            + previous_h * (values[index + 1] - values[index]) / next_h
        )
    return np.linalg.solve(matrix, rhs)


class TrajectoryChunkExecutor:
    """Convert buffered dual-arm tool trajectories to joint actions.

    The predictor is called only when the queue is empty::

        executor = TrajectoryChunkExecutor(policy_hz=10)
        executor.reset(episode=0)
        for step in range(steps):
            action = executor.act(observation, step, 0, predict)
            observation, reward, terminated, truncated, info = env.step(action)

    ``predict(observation, step, episode)`` returns::

        {
          "action_dt_s": 0.1,
          "waypoints": [
            {"left": {"position_m": [x, y, z],
                      "quaternion_wxyz": [w, x, y, z],
                      "gripper_open_fraction": 0.5},
             "right": {"position_m": [...], "quaternion_wxyz": [...]}},
            ...
          ],
          "execute_steps": 4  # optional: run first four, then predict again
        }

    Position is in meters; quaternion is scalar-first. Poses are absolute in
    the simulator world frame and refer to ``yubi_tool``. Omitted arms retain
    their previous targets. An arm may contain only a gripper command. Model
    wrappers must transform and resample their output to ``policy_hz`` first.
    """

    def __init__(
        self,
        *,
        policy_hz: float = 10.0,
        damping: float = 0.05,
        position_gain: float = 0.7,
        orientation_gain: float = 0.7,
        max_position_step_m: float = 0.06,
        max_orientation_step_rad: float = 0.3,
        max_joint_step_rad: float = 0.08,
        max_target_distance_m: float = 1.5,
        lookahead_feedforward_gain: float = 0.0,
    ) -> None:
        self.policy_hz = _positive_number(policy_hz, "policy_hz")
        self.damping = _positive_number(damping, "damping")
        self.position_gain = _positive_number(position_gain, "position_gain")
        self.orientation_gain = _positive_number(orientation_gain, "orientation_gain")
        self.max_position_step_m = _positive_number(max_position_step_m, "max_position_step_m")
        self.max_orientation_step_rad = _positive_number(max_orientation_step_rad, "max_orientation_step_rad")
        self.max_joint_step_rad = _positive_number(max_joint_step_rad, "max_joint_step_rad")
        self.max_target_distance_m = _positive_number(max_target_distance_m, "max_target_distance_m")
        if isinstance(lookahead_feedforward_gain, bool):
            raise TrajectoryFormatError("lookahead_feedforward_gain must be a finite nonnegative number")
        self.lookahead_feedforward_gain = float(lookahead_feedforward_gain)
        if not math.isfinite(self.lookahead_feedforward_gain) or self.lookahead_feedforward_gain < 0:
            raise TrajectoryFormatError("lookahead_feedforward_gain must be a finite nonnegative number")
        self._queue: deque[dict[str, dict[str, Any]]] = deque()
        self._episode: int | None = None
        self._last_step = -1
        self.last_interarm_guard: dict[str, Any] = {"active": False}

    @property
    def remaining_waypoints(self) -> int:
        return len(self._queue)

    def reset(self, episode: int | None = None) -> None:
        """Clear old model actions before a new simulator episode."""
        if episode is not None and (isinstance(episode, bool) or not isinstance(episode, int) or episode < 0):
            raise TrajectoryFormatError("episode must be a nonnegative integer")
        self._queue.clear()
        self._episode = episode
        self._last_step = -1
        self.last_interarm_guard = {"active": False}

    def _validate_waypoint(self, waypoint: Any, index: int) -> dict[str, dict[str, Any]]:
        if not isinstance(waypoint, Mapping) or not waypoint or set(waypoint) - SIDES:
            raise TrajectoryFormatError(f"waypoints[{index}] must contain left and/or right")
        normalized: dict[str, dict[str, Any]] = {}
        for side, command in waypoint.items():
            label = f"waypoints[{index}].{side}"
            if not isinstance(command, Mapping) or not command or set(command) - ARM_COMMAND_KEYS:
                raise TrajectoryFormatError(f"{label} has unknown or missing command fields")
            has_position = "position_m" in command
            has_quaternion = "quaternion_wxyz" in command
            if has_position != has_quaternion:
                raise TrajectoryFormatError(f"{label} needs both position_m and quaternion_wxyz")
            item: dict[str, Any] = {}
            if has_position:
                item["position_m"] = _vector(command["position_m"], 3, f"{label}.position_m").tolist()
                item["quaternion_wxyz"] = _quaternion(
                    command["quaternion_wxyz"], f"{label}.quaternion_wxyz", target=True
                ).tolist()
            if "gripper_open_fraction" in command:
                if isinstance(command["gripper_open_fraction"], bool):
                    raise TrajectoryFormatError(f"{label}.gripper_open_fraction must be in [0, 1]")
                try:
                    gripper = float(command["gripper_open_fraction"])
                except (TypeError, ValueError) as exc:
                    raise TrajectoryFormatError(f"{label}.gripper_open_fraction must be in [0, 1]") from exc
                if not math.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
                    raise TrajectoryFormatError(f"{label}.gripper_open_fraction must be in [0, 1]")
                item["gripper_open_fraction"] = gripper
            normalized[side] = item
        return normalized

    def submit_chunk(self, chunk: Mapping[str, Any]) -> int:
        """Replace the queue with validated waypoints; return queued count."""
        if not isinstance(chunk, Mapping) or set(chunk) - CHUNK_KEYS:
            raise TrajectoryFormatError("prediction must contain only action_dt_s, waypoints, execute_steps")
        if "action_dt_s" not in chunk or "waypoints" not in chunk:
            raise TrajectoryFormatError("prediction needs action_dt_s and waypoints")
        action_dt_s = _positive_number(chunk["action_dt_s"], "action_dt_s")
        if not math.isclose(action_dt_s, 1.0 / self.policy_hz, rel_tol=1e-6, abs_tol=1e-9):
            raise TrajectoryFormatError(
                f"action_dt_s={action_dt_s:g} does not match simulator period {1.0 / self.policy_hz:g}; "
                "resample in the checkpoint wrapper"
            )
        waypoints = chunk["waypoints"]
        if not isinstance(waypoints, (list, tuple)) or not waypoints:
            raise TrajectoryFormatError("waypoints must be a nonempty list")
        execute_steps = chunk.get("execute_steps", len(waypoints))
        if isinstance(execute_steps, bool) or not isinstance(execute_steps, int) or not 1 <= execute_steps <= len(waypoints):
            raise TrajectoryFormatError("execute_steps must be an integer in [1, len(waypoints)]")
        # Validate the entire prediction before replacing a live queue.
        normalized = [self._validate_waypoint(waypoint, i) for i, waypoint in enumerate(waypoints)]
        self._add_chunk_preview(normalized, action_dt_s)
        self._queue = deque(normalized[:execute_steps])
        return len(self._queue)

    @staticmethod
    def _add_chunk_preview(waypoints: list[dict[str, dict[str, Any]]], action_dt_s: float) -> None:
        """Annotate pose commands with whole-chunk Cartesian spline velocity."""
        for side in SIDES:
            indices = [
                index for index, waypoint in enumerate(waypoints)
                if side in waypoint and "position_m" in waypoint[side]
            ]
            if not indices:
                continue
            positions = np.asarray([waypoints[index][side]["position_m"] for index in indices])
            times = np.asarray(indices, dtype=np.float64) * action_dt_s
            if len(indices) == 1:
                linear_velocity = np.zeros((1, 3), dtype=np.float64)
                angular_velocity = np.zeros((1, 3), dtype=np.float64)
            else:
                linear_velocity = _natural_cubic_tangents(positions, times)
                quaternions = np.asarray(
                    [waypoints[index][side]["quaternion_wxyz"] for index in indices], dtype=np.float64
                )
                for index in range(1, len(quaternions)):
                    if float(np.dot(quaternions[index - 1], quaternions[index])) < 0:
                        quaternions[index] *= -1.0
                rotation_path = np.zeros((len(quaternions), 3), dtype=np.float64)
                for index in range(1, len(quaternions)):
                    rotation_path[index] = rotation_path[index - 1] + _world_rotation_error(
                        quaternions[index], quaternions[index - 1]
                    )
                angular_velocity = _natural_cubic_tangents(rotation_path, times)
            for row, waypoint_index in enumerate(indices):
                command = waypoints[waypoint_index][side]
                command["_linear_velocity_m_s"] = linear_velocity[row].tolist()
                command["_angular_velocity_rad_s"] = angular_velocity[row].tolist()

    def _arm_action(self, side: str, command: dict[str, Any], observation: Mapping[str, Any]) -> dict:
        action = {}
        if "gripper_open_fraction" in command:
            action["gripper_open_fraction"] = command["gripper_open_fraction"]
        if "position_m" not in command:
            return action
        try:
            robot = observation["robots"][side]
        except (KeyError, TypeError) as exc:
            raise TrajectoryFormatError(f"observation is missing robots.{side}") from exc
        if not isinstance(robot, Mapping):
            raise TrajectoryFormatError(f"robots.{side} must be an object")
        if robot.get("tool_body_name") != "yubi_tool":
            raise TrajectoryFormatError(f"robots.{side} must expose the yubi_tool body for 6D control")
        names = robot.get("joint_names")
        if not isinstance(names, (list, tuple)) or not all(isinstance(name, str) for name in names) or len(set(names)) != len(names):
            raise TrajectoryFormatError(f"robots.{side}.joint_names must be unique")
        try:
            indices = [names.index(name) for name in ARM_NAMES]
        except ValueError as exc:
            raise TrajectoryFormatError(f"robots.{side} is missing a Panda arm joint") from exc
        full_joint_positions = _vector(robot.get("joint_positions"), len(names), f"robots.{side}.joint_positions")
        q = full_joint_positions[indices]
        limits = _matrix(robot.get("arm_joint_limits_rad"), (7, 2), f"robots.{side}.arm_joint_limits_rad")
        jacobian = _matrix(robot.get("tool_jacobian"), (6, 7), f"robots.{side}.tool_jacobian")
        if np.any(limits[:, 0] >= limits[:, 1]):
            raise TrajectoryFormatError(f"robots.{side}.arm_joint_limits_rad must be finite 7x2 lower/upper bounds")
        if np.any(q < limits[:, 0] - 1e-3) or np.any(q > limits[:, 1] + 1e-3):
            raise TrajectoryFormatError(f"robots.{side} arm joints are outside their USD limits")
        pose = robot.get("tool_pose")
        if not isinstance(pose, Mapping):
            raise TrajectoryFormatError(f"robots.{side}.tool_pose is missing")
        position = _vector(pose.get("position_m"), 3, f"robots.{side}.tool_pose.position_m")
        orientation = _quaternion(
            pose.get("quaternion_wxyz"), f"robots.{side}.tool_pose.quaternion_wxyz", target=False
        )
        goal_position = np.asarray(command["position_m"], dtype=np.float64)
        distance = float(np.linalg.norm(goal_position - position))
        if distance > self.max_target_distance_m:
            raise TrajectoryFormatError(
                f"{side} tool target is {distance:.3f} m away; limit is {self.max_target_distance_m:.3f} m"
            )
        position_error = goal_position - position
        goal_orientation = np.asarray(command["quaternion_wxyz"], dtype=np.float64)
        orientation_error = _world_rotation_error(goal_orientation, orientation)
        preview_dt = 1.0 / self.policy_hz
        linear_preview = np.asarray(command.get("_linear_velocity_m_s", (0.0, 0.0, 0.0)))
        angular_preview = np.asarray(command.get("_angular_velocity_rad_s", (0.0, 0.0, 0.0)))
        translation = _cap_norm(
            self.position_gain * position_error
            + self.lookahead_feedforward_gain * preview_dt * linear_preview,
            self.max_position_step_m,
        )
        rotation = _cap_norm(
            self.orientation_gain * orientation_error
            + self.lookahead_feedforward_gain * preview_dt * angular_preview,
            self.max_orientation_step_rad,
        )
        twist = np.concatenate((translation, rotation))
        gram = jacobian @ jacobian.T + (self.damping**2) * np.eye(6)
        try:
            increment = jacobian.T @ np.linalg.solve(gram, twist)
        except np.linalg.LinAlgError as exc:
            raise TrajectoryFormatError(f"{side} damped IK solve failed") from exc
        if not np.isfinite(increment).all():
            raise TrajectoryFormatError(f"{side} damped IK returned non-finite increments")
        increment = np.clip(increment, -self.max_joint_step_rad, self.max_joint_step_rad)
        targets = np.clip(q + increment, limits[:, 0], limits[:, 1])
        action["arm_joint_targets_rad"] = targets.tolist()
        return action

    def act(
        self,
        observation: Mapping[str, Any],
        step: int,
        episode: int,
        predict: Callable[[Mapping[str, Any], int, int], Mapping[str, Any]],
    ) -> dict[str, dict]:
        """Return one joint action, requesting a new model chunk when needed."""
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise TrajectoryFormatError("step must be a nonnegative integer")
        if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
            raise TrajectoryFormatError("episode must be a nonnegative integer")
        if not callable(predict):
            raise TrajectoryFormatError("predict must be callable")
        if self._episode != episode:
            if step != 0:
                raise TrajectoryFormatError("a new episode must start at step 0")
            self.reset(episode)
        if step != self._last_step + 1:
            raise TrajectoryFormatError(f"expected step {self._last_step + 1} in episode {episode}, got {step}")
        if not self._queue:
            self.submit_chunk(predict(observation, step, episode))
        waypoint = self._queue[0]
        result = {side: self._arm_action(side, command, observation) for side, command in waypoint.items()}
        result, self.last_interarm_guard = limit_interarm_motion(observation, result)
        self._queue.popleft()
        self._last_step = step
        return result


__all__ = [
    "TRAJECTORY_EXECUTOR_PROFILES",
    "TrajectoryChunkExecutor",
    "TrajectoryFormatError",
    "trajectory_executor_profile",
]
