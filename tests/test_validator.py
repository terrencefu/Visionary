"""Observed vs expected pose -> the correction the user must apply."""
import unittest

from perception.validator import PoseError, compare, correction_text

TARGET = {"x_mm": 125.0, "y_mm": 95.0, "theta_deg": 0.0}


class CompareTests(unittest.TestCase):
    def test_a_placement_within_tolerance_is_correct(self):
        observed = {"x_mm": 125.4, "y_mm": 94.7, "theta_deg": 1.2}

        error = compare(observed, TARGET, tol_mm=2.0, tol_deg=5.0)

        self.assertTrue(error.correct)
        self.assertEqual(error.status, "ok")

    def test_delta_is_the_correction_to_apply_not_the_mistake_made(self):
        """dx is what the user must still move, i.e. expected - observed."""
        observed = {"x_mm": 113.7, "y_mm": 101.2, "theta_deg": 0.0}

        error = compare(observed, TARGET, tol_mm=2.0, tol_deg=5.0)

        self.assertAlmostEqual(error.dx_mm, 11.3, places=6)
        self.assertAlmostEqual(error.dy_mm, -6.2, places=6)
        self.assertFalse(error.correct)

    def test_distance_is_reported(self):
        observed = {"x_mm": 122.0, "y_mm": 91.0, "theta_deg": 0.0}

        error = compare(observed, TARGET, tol_mm=1.0, tol_deg=5.0)

        self.assertAlmostEqual(error.distance_mm, 5.0, places=6)

    def test_angle_error_is_the_short_way_round(self):
        observed = {"x_mm": 125.0, "y_mm": 95.0, "theta_deg": 350.0}

        error = compare(observed, TARGET, tol_mm=2.0, tol_deg=5.0)

        self.assertAlmostEqual(error.dtheta_deg, 10.0, places=6)

    def test_angle_error_never_exceeds_180(self):
        for observed_theta in (0.0, 45.0, 179.0, 181.0, 275.0, 359.0):
            with self.subTest(theta=observed_theta):
                error = compare({"x_mm": 125.0, "y_mm": 95.0, "theta_deg": observed_theta},
                                TARGET, tol_mm=2.0, tol_deg=5.0)
                self.assertLessEqual(abs(error.dtheta_deg), 180.0)

    def test_position_alone_can_fail_the_check(self):
        error = compare({"x_mm": 140.0, "y_mm": 95.0, "theta_deg": 0.0},
                        TARGET, tol_mm=2.0, tol_deg=5.0)
        self.assertFalse(error.correct)
        self.assertEqual(error.status, "wrong_position")

    def test_angle_alone_can_fail_the_check(self):
        error = compare({"x_mm": 125.0, "y_mm": 95.0, "theta_deg": 30.0},
                        TARGET, tol_mm=2.0, tol_deg=5.0)
        self.assertFalse(error.correct)
        self.assertEqual(error.status, "wrong_angle")


class SymmetryTests(unittest.TestCase):
    def test_a_180_symmetric_part_placed_backwards_is_correct(self):
        """A 2x4 brick turned end for end is the same physical placement."""
        observed = {"x_mm": 125.0, "y_mm": 95.0, "theta_deg": 180.0}

        error = compare(observed, TARGET, tol_mm=2.0, tol_deg=5.0, symmetry_deg=180)

        self.assertTrue(error.correct)
        self.assertAlmostEqual(error.dtheta_deg, 0.0, places=6)

    def test_symmetry_folds_the_error_to_the_nearest_equivalent(self):
        observed = {"x_mm": 125.0, "y_mm": 95.0, "theta_deg": 179.0}

        error = compare(observed, TARGET, tol_mm=2.0, tol_deg=5.0, symmetry_deg=180)

        self.assertAlmostEqual(error.dtheta_deg, 1.0, places=6)

    def test_an_asymmetric_part_placed_backwards_is_wrong(self):
        observed = {"x_mm": 125.0, "y_mm": 95.0, "theta_deg": 180.0}

        error = compare(observed, TARGET, tol_mm=2.0, tol_deg=5.0, symmetry_deg=360)

        self.assertFalse(error.correct)
        self.assertAlmostEqual(abs(error.dtheta_deg), 180.0, places=6)


class CorrectionTextTests(unittest.TestCase):
    def test_success_reads_as_success(self):
        error = compare(TARGET, TARGET, tol_mm=2.0, tol_deg=5.0)
        self.assertIn("OK", correction_text(error))

    def test_names_the_board_axis_and_the_millimetres(self):
        error = compare({"x_mm": 113.7, "y_mm": 101.2, "theta_deg": 0.0},
                        TARGET, tol_mm=2.0, tol_deg=5.0)

        text = correction_text(error)

        self.assertIn("+X", text)
        self.assertIn("11.3", text)
        self.assertIn("-Y", text)
        self.assertIn("6.2", text)

    def test_names_the_rotation_direction(self):
        ccw = compare({"x_mm": 125.0, "y_mm": 95.0, "theta_deg": 350.0},
                      TARGET, tol_mm=2.0, tol_deg=5.0)
        cw = compare({"x_mm": 125.0, "y_mm": 95.0, "theta_deg": 10.0},
                     TARGET, tol_mm=2.0, tol_deg=5.0)

        self.assertIn("CCW", correction_text(ccw))
        self.assertIn("CW", correction_text(cw))
        self.assertNotIn("CCW", correction_text(cw))

    def test_an_in_tolerance_axis_is_not_mentioned(self):
        error = compare({"x_mm": 125.2, "y_mm": 80.0, "theta_deg": 0.0},
                        TARGET, tol_mm=2.0, tol_deg=5.0)

        text = correction_text(error)

        self.assertNotIn("X", text)
        self.assertIn("+Y", text)


class ExpectedPoseShapeTests(unittest.TestCase):
    def test_accepts_a_pose_object_as_well_as_a_dict(self):
        class Pose:
            x_mm, y_mm, theta_deg = 125.0, 95.0, 0.0

        error = compare(Pose(), TARGET, tol_mm=2.0, tol_deg=5.0)
        self.assertTrue(error.correct)

    def test_missing_target_field_is_rejected_clearly(self):
        with self.assertRaises(KeyError):
            compare(TARGET, {"x_mm": 1.0, "y_mm": 2.0}, tol_mm=2.0, tol_deg=5.0)


if __name__ == "__main__":
    unittest.main()
