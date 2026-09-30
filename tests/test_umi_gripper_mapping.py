import unittest
import warnings
import numpy as np

from yubi_isaac_sim_env.umi_gripper_mapping import (
    source_angles_to_open_fraction, source_aperture_m, motorized_aperture_m,
)


class FixedApertureMappingTests(unittest.TestCase):
    def test_identical_model_output_is_independent_of_recording_context(self):
        single = source_angles_to_open_fraction([.4464], -.1, .7)[0]
        batch = source_angles_to_open_fraction([.1, .4464, .77], -.1, .7)[1]
        cropped = source_angles_to_open_fraction([.4464, .6], -.1, .7)[0]
        self.assertEqual(single, batch)
        self.assertEqual(single, cropped)

    def test_preserves_distal_aperture_across_different_jaw_geometries(self):
        source = np.linspace(.1, .79, 200)
        q = -.1 + .8 * source_angles_to_open_fraction(source, -.1, .7)
        np.testing.assert_allclose(motorized_aperture_m(q), source_aperture_m(source), atol=1e-6)
        self.assertTrue(np.all(np.diff(q) > 0))

    def test_closed_empty_gripper_and_held_cup_are_distinct(self):
        fraction = source_angles_to_open_fraction([0., .44638841], -.1, .7)
        self.assertAlmostEqual(fraction[0], 0.)
        self.assertGreater(fraction[1], .5)
        self.assertLess(float(motorized_aperture_m([-.1])[0]), .0005)

    def test_out_of_stroke_is_reported_not_silently_rescaled(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fraction = source_angles_to_open_fraction([.94], -.1, .7)
        self.assertEqual(fraction[0], 1.)
        self.assertTrue(any("stroke" in str(w.message) for w in caught))

    def test_invalid_input_is_rejected(self):
        for angles in ([], [np.nan], [[.4]]):
            with self.assertRaises(ValueError):
                source_angles_to_open_fraction(angles, -.1, .7)


if __name__ == "__main__":
    unittest.main()
