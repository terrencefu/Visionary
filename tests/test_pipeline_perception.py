"""The whole perception slice: two frames in, a correction out.

End to end through a real lens model and an upside-down camera, with no
hardware: before frame, after frame, expected part, board pose.
"""
import unittest

import cv2
import numpy as np

from perception import part_catalog
from perception.part_matcher import MatchResult
from perception.pipeline import detect_and_validate
from tests.synthetic import blank_scene, board_camera, render_part_on_board

K = np.array([[1500.0, 0, 960.0], [0, 1490.0, 540.0], [0, 0, 1.0]])
DIST = np.array([0.05, -0.02, 0.001, -0.001, 0.0])
SHAPE = (1080, 1920)
BGR = {"red": (40, 40, 210), "blue": (200, 70, 30),
       "yellow": (40, 210, 225), "green": (60, 180, 60)}


class FakeBoardPose:
    """Stands in for perception.aruco.BoardPose, which needs a real image."""

    def __init__(self, rvec, tvec, image_points=None):
        self.rvec, self.tvec = rvec, tvec
        self.image_points = image_points
        self.reprojection_error = 0.4
        self.visible_ids = [0, 1, 2, 3]


def overhead_camera(distance_mm=600.0, look_at=(125.0, 95.0), elev_deg=35.0):
    """The real rig: mounted upside down and looking at the board from an angle."""
    return FakeBoardPose(*board_camera(elev_deg=elev_deg, distance_mm=distance_mm,
                                       look_at=look_at))


def scene(board_pose, placements=()):
    frame = blank_scene(SHAPE)
    for part_id, x_mm, y_mm, theta_deg, colour in placements:
        render_part_on_board(frame, part_catalog.get(part_id).outline_mm,
                             x_mm, y_mm, theta_deg, board_pose.rvec, board_pose.tvec,
                             K, DIST, BGR[colour])
    return frame


class DetectAndValidateTests(unittest.TestCase):
    def setUp(self):
        self.board = overhead_camera()
        self.before = scene(self.board)

    def test_reports_the_placement_in_board_millimetres(self):
        after = scene(self.board, [("red_l_plate", 118.0, 92.0, 25.0, "red")])

        result = detect_and_validate(self.before, after, "red_l_plate", self.board, K, DIST)

        self.assertTrue(result.detected)
        self.assertTrue(result.correct_part)
        self.assertEqual(result.part_id, "red_l_plate")
        self.assertAlmostEqual(result.observed_pose["x_mm"], 118.0, delta=2.0)
        self.assertAlmostEqual(result.observed_pose["y_mm"], 92.0, delta=2.0)
        self.assertAlmostEqual(result.observed_pose["theta_deg"], 25.0, delta=4.0)
        self.assertGreater(result.confidence, 0.85)

    def test_result_matches_the_agreed_integration_shape(self):
        after = scene(self.board, [("red_l_plate", 125.0, 95.0, 0.0, "red")])

        payload = detect_and_validate(self.before, after, "red_l_plate",
                                      self.board, K, DIST).as_dict()

        self.assertEqual(set(payload), {"detected", "correct_part", "part_id",
                                        "observed_pose", "confidence"})
        self.assertEqual(set(payload["observed_pose"]), {"x_mm", "y_mm", "z_mm", "theta_deg"})

    def test_nothing_placed_is_reported_as_no_change(self):
        result = detect_and_validate(self.before, self.before.copy(), "red_l_plate",
                                     self.board, K, DIST)

        self.assertFalse(result.detected)
        self.assertEqual(result.status, "no_change")

    def test_the_wrong_part_is_named(self):
        after = scene(self.board, [("blue_2x4", 125.0, 95.0, 0.0, "blue")])

        result = detect_and_validate(self.before, after, "red_l_plate", self.board, K, DIST)

        self.assertTrue(result.detected)
        self.assertFalse(result.correct_part)
        self.assertEqual(result.status, "wrong_part")
        self.assertEqual(result.part_id, "blue_2x4")

    def test_a_hand_in_the_frame_is_refused_rather_than_measured(self):
        after = self.before.copy()
        cv2.rectangle(after, (200, 100), (1700, 950), (150, 170, 200), -1)

        result = detect_and_validate(self.before, after, "red_l_plate", self.board, K, DIST)

        self.assertFalse(result.detected)
        self.assertEqual(result.status, "no_change")

    def test_a_part_from_an_earlier_step_does_not_retrigger(self):
        """The earlier part is in BOTH frames, so only the new one is judged."""
        self.before = scene(self.board, [("blue_2x4", 60.0, 40.0, 0.0, "blue")])
        after = scene(self.board, [("blue_2x4", 60.0, 40.0, 0.0, "blue"),
                                   ("red_l_plate", 150.0, 120.0, 10.0, "red")])

        result = detect_and_validate(self.before, after, "red_l_plate", self.board, K, DIST)

        self.assertTrue(result.correct_part)
        self.assertAlmostEqual(result.observed_pose["x_mm"], 150.0, delta=2.0)


class ValidationAgainstTargetTests(unittest.TestCase):
    def setUp(self):
        self.board = overhead_camera()
        self.before = scene(self.board)

    def test_a_good_placement_validates(self):
        after = scene(self.board, [("red_l_plate", 125.0, 95.0, 0.0, "red")])
        target = {"x_mm": 125.0, "y_mm": 95.0, "theta_deg": 0.0}

        result = detect_and_validate(self.before, after, "red_l_plate", self.board, K, DIST,
                                     expected_pose=target, tol_mm=3.0, tol_deg=8.0)

        self.assertTrue(result.error.correct)
        self.assertEqual(result.status, "ok")

    def test_a_misplacement_returns_the_correction(self):
        after = scene(self.board, [("red_l_plate", 113.0, 101.0, 12.0, "red")])
        target = {"x_mm": 125.0, "y_mm": 95.0, "theta_deg": 0.0}

        result = detect_and_validate(self.before, after, "red_l_plate", self.board, K, DIST,
                                     expected_pose=target, tol_mm=3.0, tol_deg=8.0)

        self.assertFalse(result.error.correct)
        self.assertAlmostEqual(result.error.dx_mm, 12.0, delta=2.0)
        self.assertAlmostEqual(result.error.dy_mm, -6.0, delta=2.0)
        self.assertAlmostEqual(result.error.dtheta_deg, -12.0, delta=4.0)

    def test_symmetry_comes_from_the_catalogue(self):
        """blue_2x4 is 180-symmetric, so end-for-end is not an error."""
        after = scene(self.board, [("blue_2x4", 125.0, 95.0, 180.0, "blue")])
        target = {"x_mm": 125.0, "y_mm": 95.0, "theta_deg": 0.0}

        result = detect_and_validate(self.before, after, "blue_2x4", self.board, K, DIST,
                                     expected_pose=target, tol_mm=3.0, tol_deg=8.0)

        self.assertTrue(result.error.correct)


class MatcherSwapTests(unittest.TestCase):
    def test_the_pipeline_accepts_any_matcher(self):
        """Proves the CAD-render matcher can replace HSV without touching this
        module: a stub that reports a fixed pose flows straight through."""
        board = overhead_camera()
        before = scene(board)
        after = scene(board, [("red_l_plate", 125.0, 95.0, 0.0, "red")])

        class StubMatcher:
            def __init__(self):
                self.calls = []

            def verify(self, frame, region, expected_part_id, px_per_mm):
                self.calls.append((expected_part_id, px_per_mm))
                from perception.pose_estimator import PlanarPose
                return MatchResult("ok", expected_part_id, None, 0.99,
                                   PlanarPose(960.0, 540.0, 0.0, 0.99), "stub")

        stub = StubMatcher()
        result = detect_and_validate(before, after, "red_l_plate", board, K, DIST,
                                     matcher=stub)

        self.assertTrue(result.correct_part)
        self.assertAlmostEqual(result.confidence, 0.99)
        self.assertEqual(stub.calls[0][0], "red_l_plate")
        self.assertGreater(stub.calls[0][1], 0.0, "matcher must be told the local px/mm")


class PreUndistortedInputTests(unittest.TestCase):
    """The state machine undistorts the baseline once and reuses it, instead of
    paying ~17 ms per frame to undistort the same unchanged image again."""

    def setUp(self):
        self.board = overhead_camera()
        self.before = scene(self.board)
        self.after = scene(self.board, [("red_l_plate", 118.0, 92.0, 25.0, "red")])

    def test_pre_undistorted_frames_give_the_same_answer(self):
        done_inside = detect_and_validate(self.before, self.after, "red_l_plate",
                                          self.board, K, DIST)
        done_outside = detect_and_validate(cv2.undistort(self.before, K, DIST),
                                           cv2.undistort(self.after, K, DIST),
                                           "red_l_plate", self.board, K, DIST,
                                           already_undistorted=True)

        self.assertTrue(done_outside.correct_part)
        for field in ("x_mm", "y_mm", "theta_deg"):
            self.assertAlmostEqual(done_outside.observed_pose[field],
                                   done_inside.observed_pose[field], delta=0.5)


class InputGuardTests(unittest.TestCase):
    def test_mismatched_frame_sizes_are_rejected(self):
        board = overhead_camera()
        with self.assertRaises(ValueError):
            detect_and_validate(blank_scene(SHAPE), blank_scene((480, 640)),
                                "red_l_plate", board, K, DIST)

    def test_a_missing_board_pose_is_rejected(self):
        with self.assertRaises(ValueError):
            detect_and_validate(blank_scene(SHAPE), blank_scene(SHAPE),
                                "red_l_plate", None, K, DIST)


if __name__ == "__main__":
    unittest.main()
