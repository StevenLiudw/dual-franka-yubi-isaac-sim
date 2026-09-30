import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from yubi_isaac_sim_env.umi_pose_mapping import source_to_tool, interpolate_orientation


class PoseMappingTests(unittest.TestCase):
    def test_identity_hand_points_to_operator_right_in_world(self):
        xyz, wxyz = source_to_tool(np.array([[0, 0, 0, 0, 0, 0, 1.]]))
        rotation = Rotation.from_quat(wxyz[:, [1, 2, 3, 0]])
        np.testing.assert_allclose(xyz[0], [0, -0.09343, 0], atol=1e-12)
        # Tool +Z points down the fingers; +Y is the camera side.
        np.testing.assert_allclose(rotation.apply([0, 0, 1])[0], [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(rotation.apply([0, 1, 0])[0], [0, 0, 1], atol=1e-12)

    def test_rotation_moves_tcp_and_preserves_angular_motion(self):
        source = Rotation.from_euler("z", [0, 90], degrees=True)
        xyz, wxyz = source_to_tool(np.column_stack([np.zeros((2, 3)), source.as_quat()]))
        np.testing.assert_allclose(xyz[1], [0.09343, 0, 0], atol=1e-12)
        mapped = Rotation.from_quat(wxyz[:, [1, 2, 3, 0]])
        self.assertAlmostEqual((mapped[0].inv() * mapped[1]).magnitude(), np.pi / 2)

    def test_antipodal_quaternions_do_not_create_turns(self):
        _, q = source_to_tool(np.array([[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, -1.]]))
        np.testing.assert_allclose(q[0], q[1], atol=1e-12)
        midpoint = interpolate_orientation([1, 0, 0, 0], [-1, 0, 0, 0], [0.5])
        self.assertAlmostEqual(abs(midpoint[0, 0]), 1.)

    def test_shared_frame_preserves_interhand_distance(self):
        values = np.array([[-0.3, 0.2, 0.1, 0, 0, 0, 1], [0.3, 0.2, 0.1, 0, 0, 0, 1.]])
        xyz, _ = source_to_tool(values)
        self.assertAlmostEqual(np.linalg.norm(xyz[1] - xyz[0]), 0.6)
        self.assertGreater(xyz[0, 1], xyz[1, 1])  # Left hand stays image-left.

    def test_rejects_missing_orientation(self):
        with self.assertRaises(ValueError):
            source_to_tool(np.zeros((2, 7)))


if __name__ == "__main__":
    unittest.main()
