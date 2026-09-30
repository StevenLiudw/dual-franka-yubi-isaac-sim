import unittest

import numpy as np

from yubi_isaac_sim_env.interarm_guard import (
    limit_interarm_motion,
    observed_interarm_clearance_m,
)
from yubi_isaac_sim_env.umi_gripper_mapping import source_angles_to_open_fraction


def _robot(x):
    jacobian = np.zeros((3, 7))
    jacobian[0, 0] = 1.0
    return {
        "joint_names": [f"panda_joint{i}" for i in range(1, 8)],
        "joint_positions": [0.0] * 7,
        "gripper_open_fraction": 0.5,
        "collision_points": [{
            "name": "yubi_base", "position_m": [x, 0.0, 0.0],
            "radius_m": 0.02, "arm_translation_jacobian": jacobian.tolist(),
        }],
    }


class InterarmGuardTests(unittest.TestCase):
    def test_observed_clearance_accounts_for_both_link_radii(self):
        observation = {"robots": {"left": _robot(0.0), "right": _robot(0.2)}}
        self.assertAlmostEqual(observed_interarm_clearance_m(observation), 0.16)

    def test_swept_collision_is_blocked_even_when_endpoint_is_clear(self):
        observation = {"robots": {"left": _robot(0.0), "right": _robot(0.2)}}
        action = {"left": {"arm_joint_targets_rad": [0.4] + [0.0] * 6},
                  "right": {"arm_joint_targets_rad": [0.0] * 7}}
        guarded, status = limit_interarm_motion(observation, action)
        self.assertLess(guarded["left"]["arm_joint_targets_rad"][0], 0.2)
        self.assertGreaterEqual(status["predicted_swept_clearance_m"], 0.02 - 1e-9)

    def test_close_arms_can_only_move_apart(self):
        observation = {"robots": {"left": _robot(0.0), "right": _robot(0.045)}}
        action = {"left": {"arm_joint_targets_rad": [0.02] + [0.0] * 6},
                  "right": {"arm_joint_targets_rad": [0.02] + [0.0] * 6}}
        guarded, status = limit_interarm_motion(observation, action)
        self.assertGreaterEqual(status["predicted_swept_clearance_m"],
                                status["current_clearance_m"] - 1e-4)
        self.assertLessEqual(guarded["left"]["arm_joint_targets_rad"][0], 0.02)

    def test_gripper_mapping_does_not_close_on_a_held_object_minimum(self):
        fraction = source_angles_to_open_fraction([0.44638841], -0.1, 0.7)
        self.assertGreater(float(fraction[0]), 0.5)


if __name__ == "__main__":
    unittest.main()
