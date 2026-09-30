"""Offline checks for the model-facing RGB observation contract."""

import unittest
from unittest.mock import patch

import numpy as np

from yubi_isaac_sim_env import run


class _Camera:
    def get_world_pose(self, camera_axes):
        assert camera_axes == "usd"
        return np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0])

    def get_current_frame(self):
        return {"rendering_time": 1.5}


class PolicyImagesTest(unittest.TestCase):
    def test_float_rgb_is_copied_as_uint8(self):
        source = np.full((2, 4, 3), 0.5, dtype=np.float32)
        spec = {"name": "head", "resolution": [4, 2]}
        result = run._policy_rgb(source, spec)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (2, 4, 3))
        self.assertEqual(int(result[0, 0, 0]), 127)
        self.assertTrue(result.flags.writeable)
        source[0, 0, 0] = 0.0
        self.assertEqual(int(result[0, 0, 0]), 127)

    def test_three_view_observation_does_not_mutate_state(self):
        names = ("head", "left_wrist", "right_wrist")
        cameras = {name: _Camera() for name in names}
        specs = {name: {"name": name, "resolution": [4, 2], "lens_model": "test"} for name in names}
        state = {"physics_time_s": 1.0, "robots": {}}
        with patch.object(run, "_rendered_rgb", return_value=np.full((2, 4, 3), 42, dtype=np.uint8)):
            observation = run._policy_observation(state, cameras, specs, object())
        self.assertEqual(set(observation["images"]), set(names))
        self.assertEqual(observation["image_metadata"]["head"]["physics_time_s"], 1.0)
        self.assertEqual(observation["image_metadata"]["head"]["rendering_time_s"], 1.5)
        self.assertNotIn("images", state)

    def test_images_require_custom_policy(self):
        with self.assertRaises(SystemExit):
            run._parse_args(["--policy-images", "all"])

    def test_target_interpolation_flag(self):
        args = run._parse_args(["--interpolate-targets"])
        self.assertTrue(args.interpolate_targets)

    def test_portable_joint_profile_does_not_change_policy_contract(self):
        args = run._parse_args([
            "--joint-command-profile", "franka-panda-interface",
            "--trajectory-controller-profile", "franka-transfer",
        ])
        self.assertEqual(args.joint_command_profile, "franka-panda-interface")
        self.assertEqual(args.trajectory_controller_profile, "franka-transfer")
        self.assertFalse(args.interpolate_targets)

    def test_joint_profile_and_linear_interpolation_are_exclusive(self):
        with self.assertRaises(SystemExit):
            run._parse_args([
                "--interpolate-targets", "--joint-command-profile", "franka-panda-interface"
            ])


if __name__ == "__main__":
    unittest.main()
