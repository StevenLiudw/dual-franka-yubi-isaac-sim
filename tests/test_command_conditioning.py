"""Offline tests for the simulator/hardware joint reference boundary."""

from __future__ import annotations

import unittest

import numpy as np

from yubi_isaac_sim_env.command_conditioning import (
    FRANKA_PANDA_INTERFACE_PROFILE,
    FRANKA_LIMIT_EPS,
    JointReferenceGovernor,
    get_joint_reference_profile,
)


class JointReferenceGovernorTests(unittest.TestCase):
    def test_panda_profile_matches_official_fci_limits(self):
        profile = FRANKA_PANDA_INTERFACE_PROFILE
        np.testing.assert_allclose(
            profile.max_velocity_rad_s,
            np.asarray([2.175] * 4 + [2.610] * 3) - FRANKA_LIMIT_EPS,
        )
        np.testing.assert_allclose(
            profile.max_acceleration_rad_s2,
            np.asarray([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]) - FRANKA_LIMIT_EPS,
        )
        np.testing.assert_allclose(
            profile.max_jerk_rad_s3,
            np.asarray([7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0])
            - FRANKA_LIMIT_EPS,
        )
        self.assertIs(get_joint_reference_profile("franka-panda-interface"), profile)
    def test_constant_target_converges_without_crossing(self):
        dt = FRANKA_PANDA_INTERFACE_PROFILE.servo_period_s
        governor = JointReferenceGovernor(FRANKA_PANDA_INTERFACE_PROFILE, dt_s=dt)
        initial = np.asarray([0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741])
        governor.reset(initial)
        samples = [initial]
        target = initial + 0.5
        for _ in range(2000):
            samples.append(governor.update(target))
        positions = np.asarray(samples)
        self.assertTrue(np.all(positions >= initial - 1e-12))
        self.assertTrue(np.all(positions <= target + 1e-9))
        np.testing.assert_allclose(positions[-1], target, atol=5e-5)

    def test_velocity_acceleration_and_jerk_are_bounded(self):
        dt = FRANKA_PANDA_INTERFACE_PROFILE.servo_period_s
        governor = JointReferenceGovernor(FRANKA_PANDA_INTERFACE_PROFILE, dt_s=dt)
        initial = np.asarray([0.0, -0.569, 0.0, -2.81, 0.0, 3.037, 0.741])
        governor.reset(initial)
        positions = [initial]
        for index in range(4000):
            target = initial + (0.05 if (index // 500) % 2 == 0 else -0.05)
            positions.append(governor.update(target))
        positions = np.asarray(positions)
        velocity = np.diff(positions, axis=0) / dt
        acceleration = np.diff(np.vstack((np.zeros((1, 7)), velocity)), axis=0) / dt
        jerk = np.diff(np.vstack((np.zeros((1, 7)), acceleration)), axis=0) / dt
        limits = np.asarray(FRANKA_PANDA_INTERFACE_PROFILE.max_velocity_rad_s)
        self.assertTrue(np.all(np.abs(velocity) <= limits + 1e-9))
        acceleration_limits = np.asarray(FRANKA_PANDA_INTERFACE_PROFILE.max_acceleration_rad_s2)
        jerk_limits = np.asarray(FRANKA_PANDA_INTERFACE_PROFILE.max_jerk_rad_s3)
        self.assertTrue(np.all(np.abs(acceleration) <= acceleration_limits + 1e-8))
        self.assertTrue(np.all(np.abs(jerk) <= jerk_limits + 1e-6))

    def test_profile_is_explicit_and_invalid_state_is_rejected(self):
        self.assertIs(
            get_joint_reference_profile("franka-panda-interface"),
            FRANKA_PANDA_INTERFACE_PROFILE,
        )
        self.assertIsNone(get_joint_reference_profile("direct"))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            get_joint_reference_profile("mystery")
        governor = JointReferenceGovernor(FRANKA_PANDA_INTERFACE_PROFILE, dt_s=0.001)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            governor.update(np.zeros(7))

if __name__ == "__main__":
    unittest.main()
