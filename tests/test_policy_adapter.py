"""Pure-NumPy checks for the checkpoint-independent trajectory boundary."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yubi_isaac_sim_env.policy_adapter import (
    TrajectoryChunkExecutor,
    TrajectoryFormatError,
    trajectory_executor_profile,
)


def observation(*, joint_zero: float = 0.0, jacobian: list | None = None) -> dict:
    jacobian = jacobian if jacobian is not None else np.eye(6, 7).tolist()
    arm = {
        "joint_names": [*(f"panda_joint{i}" for i in range(1, 8)), "yubi_finger_joint"],
        "joint_positions": [joint_zero] + [0.0] * 7,
        "arm_joint_limits_rad": [[-2.0, 2.0] for _ in range(7)],
        "tool_body_name": "yubi_tool",
        "tool_pose": {"position_m": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "tool_jacobian": jacobian,
    }
    return {"robots": {"left": arm, "right": arm}}


def pose_waypoint(*, quaternion=None, position=None) -> dict:
    return {
        "left": {
            "position_m": [0.1, 0.0, 0.0] if position is None else position,
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0] if quaternion is None else quaternion,
            "gripper_open_fraction": 0.5,
        }
    }


def chunk(*waypoints, execute_steps=None, action_dt_s=0.1) -> dict:
    result = {"action_dt_s": action_dt_s, "waypoints": list(waypoints)}
    if execute_steps is not None:
        result["execute_steps"] = execute_steps
    return result


class TrajectoryChunkExecutorTests(unittest.TestCase):
    def test_franka_transfer_profile_is_less_aggressive_than_default(self):
        profile = trajectory_executor_profile("franka-transfer")
        self.assertGreater(profile["damping"], 0.05)
        self.assertLess(profile["orientation_gain"], 0.7)
        self.assertLess(profile["max_joint_step_rad"], 0.08)

    def test_lookahead_profile_previews_full_chunk(self):
        profile = trajectory_executor_profile("franka-lookahead")
        executor = TrajectoryChunkExecutor(**profile)
        prediction = chunk(
            pose_waypoint(position=[0.0, 0.0, 0.0]),
            pose_waypoint(position=[0.1, 0.0, 0.0]),
            pose_waypoint(position=[0.4, 0.0, 0.0]),
            execute_steps=1,
        )
        executor.submit_chunk(prediction)
        preview_with_future = executor._queue[0]["left"]["_linear_velocity_m_s"][0]
        short = TrajectoryChunkExecutor(**profile)
        short.submit_chunk(chunk(pose_waypoint(position=[0.0, 0.0, 0.0])))
        preview_without_future = short._queue[0]["left"]["_linear_velocity_m_s"][0]
        self.assertGreater(preview_with_future, 0.0)
        self.assertEqual(preview_without_future, 0.0)

    def test_lookahead_feedforward_preserves_model_output_contract(self):
        executor = TrajectoryChunkExecutor(**trajectory_executor_profile("franka-lookahead"))
        action = executor.act(
            observation(), 0, 0,
            lambda *_: chunk(
                pose_waypoint(position=[0.0, 0.0, 0.0]),
                pose_waypoint(position=[0.1, 0.0, 0.0]),
            ),
        )
        self.assertEqual(set(action["left"]), {"arm_joint_targets_rad", "gripper_open_fraction"})
        self.assertGreater(action["left"]["arm_joint_targets_rad"][0], 0.0)

    def test_buffers_chunk_and_predicts_again_only_after_execution_horizon(self):
        calls = []

        def predict(obs, step, episode):
            calls.append((step, episode))
            return chunk(pose_waypoint(), pose_waypoint(), pose_waypoint(), execute_steps=2)

        executor = TrajectoryChunkExecutor()
        first = executor.act(observation(), 0, 0, predict)
        self.assertEqual(calls, [(0, 0)])
        self.assertEqual(executor.remaining_waypoints, 1)
        self.assertEqual(first["left"]["gripper_open_fraction"], 0.5)
        self.assertEqual(len(first["left"]["arm_joint_targets_rad"]), 7)
        executor.act(observation(), 1, 0, predict)
        self.assertEqual(calls, [(0, 0)])
        self.assertEqual(executor.remaining_waypoints, 0)
        executor.act(observation(), 2, 0, predict)
        self.assertEqual(calls, [(0, 0), (2, 0)])

    def test_episode_reset_discards_stale_actions(self):
        calls = []

        def predict(obs, step, episode):
            calls.append((step, episode))
            return chunk(pose_waypoint(), pose_waypoint())

        executor = TrajectoryChunkExecutor()
        executor.act(observation(), 0, 0, predict)
        self.assertEqual(executor.remaining_waypoints, 1)
        executor.reset(episode=1)
        self.assertEqual(executor.remaining_waypoints, 0)
        executor.act(observation(), 0, 1, predict)
        self.assertEqual(calls, [(0, 0), (0, 1)])

    def test_translation_and_world_rotation_sign_with_joint_cap(self):
        angle = math.pi / 2
        quaternion = [math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
        executor = TrajectoryChunkExecutor(max_joint_step_rad=0.08)
        action = executor.act(
            observation(), 0, 0,
            lambda obs, step, episode: chunk(pose_waypoint(quaternion=quaternion)),
        )
        targets = action["left"]["arm_joint_targets_rad"]
        self.assertGreater(targets[0], 0)
        self.assertGreater(targets[5], 0)
        self.assertLessEqual(max(abs(value) for value in targets), 0.08 + 1e-12)

    def test_negated_quaternion_is_same_orientation(self):
        executor = TrajectoryChunkExecutor()
        action = executor.act(
            observation(), 0, 0,
            lambda obs, step, episode: chunk(pose_waypoint(quaternion=[-1, 0, 0, 0], position=[0, 0, 0])),
        )
        self.assertEqual(action["left"]["arm_joint_targets_rad"], [0.0] * 7)

    def test_joint_limit_clipping_and_singular_jacobian(self):
        executor = TrajectoryChunkExecutor()
        near_limit = observation(joint_zero=1.99)
        action = executor.act(near_limit, 0, 0, lambda *_: chunk(pose_waypoint()))
        self.assertEqual(action["left"]["arm_joint_targets_rad"][0], 2.0)

        executor.reset(episode=1)
        singular = observation(jacobian=np.zeros((6, 7)).tolist())
        action = executor.act(singular, 0, 1, lambda *_: chunk(pose_waypoint()))
        self.assertEqual(action["left"]["arm_joint_targets_rad"], [0.0] * 7)

    def test_gripper_only_and_two_arms(self):
        executor = TrajectoryChunkExecutor()
        action = executor.act(
            observation(), 0, 0,
            lambda *_: chunk({"left": {"gripper_open_fraction": 0.0}, "right": {"gripper_open_fraction": 1.0}}),
        )
        self.assertEqual(action, {"left": {"gripper_open_fraction": 0.0}, "right": {"gripper_open_fraction": 1.0}})

    def test_rejects_rate_mismatch_invalid_pose_and_invalid_observation_without_consuming(self):
        executor = TrajectoryChunkExecutor()
        with self.assertRaisesRegex(TrajectoryFormatError, "resample"):
            executor.submit_chunk(chunk(pose_waypoint(), action_dt_s=0.05))
        with self.assertRaisesRegex(TrajectoryFormatError, "unit quaternion"):
            executor.submit_chunk(chunk(pose_waypoint(quaternion=[2, 0, 0, 0])))
        with self.assertRaisesRegex(TrajectoryFormatError, "finite numbers"):
            executor.submit_chunk(chunk(pose_waypoint(position=[math.nan, 0, 0])))
        with self.assertRaisesRegex(TrajectoryFormatError, "both position_m and quaternion_wxyz"):
            executor.submit_chunk(chunk({"left": {"position_m": [0, 0, 0]}}))
        executor.reset(episode=0)
        executor.submit_chunk(chunk(pose_waypoint()))
        invalid = observation()
        invalid["robots"]["left"] = dict(invalid["robots"]["left"], tool_jacobian=[[1]])
        with self.assertRaisesRegex(TrajectoryFormatError, "6x7"):
            executor.act(invalid, 0, 0, lambda *_: self.fail("unexpected prediction"))
        self.assertEqual(executor.remaining_waypoints, 1)

    def test_rejects_far_goal_and_nonsequential_step(self):
        executor = TrajectoryChunkExecutor()
        with self.assertRaisesRegex(TrajectoryFormatError, "limit is"):
            executor.act(
                observation(), 0, 0,
                lambda *_: chunk(pose_waypoint(position=[4.0, 0.0, 0.0])),
            )
        self.assertEqual(executor.remaining_waypoints, 1)
        executor.reset(episode=0)
        executor.act(observation(), 0, 0, lambda *_: chunk(pose_waypoint()))
        with self.assertRaisesRegex(TrajectoryFormatError, "expected step 1"):
            executor.act(observation(), 2, 0, lambda *_: self.fail("unexpected prediction"))


if __name__ == "__main__":
    unittest.main()
