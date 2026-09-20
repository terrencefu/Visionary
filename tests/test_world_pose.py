"""Part pose in board millimetres, through a real lens and an angled camera.

These tests pin the conventions the README leaves open:

    theta_deg is the angle of the part's +x axis, measured in the BOARD frame,
    CCW from board +x toward board +y, in [0, 360).

    x_mm / y_mm are the AREA CENTROID of the part silhouette, not the outline
    origin. A CAD target pose must use the same reference point.

Pose is recovered the way the pipeline does it: rectify the board plane to a
metric top-down view, then align the catalogue outline there. Matching in raw
image space is wrong at any camera angle -- see test_camera_fixture.py.
"""
import unittest

import cv2
import numpy as np

from perception.geometry import board_point_from_undistorted_pixel, camera_pixel_to_board
from perception.pose_estimator import BoardPlanarPose, align_outline, polygon_centroid
from perception.rectify import BoardRectifier, local_scale_px_per_mm
from tests.synthetic import (blank_scene, board_camera, largest_contour,
                             render_part_on_board_at_height)

STUD = 8.0
L_CHIRAL = np.array([(0, 0), (6, 0), (6, 2), (2, 2), (2, 4), (0, 4)], float) * STUD

K = np.array([[1110.0, 0, 952.7], [0, 1110.0, 529.7], [0, 0, 1.0]])
DIST = np.array([0.05, -0.02, 0.001, -0.001, 0.0])
SHAPE = (1080, 1920)
BOARD_CENTRE = np.array([246.5, 152.5])
RED_BGR = (40, 40, 210)
RED_HSV = (np.array([0, 120, 70], np.uint8), np.array([8, 255, 255], np.uint8))


def recover_pose(x_mm, y_mm, theta_deg, rvec, tvec, outline=L_CHIRAL, z_mm=0.0):
    """Render a part, then measure it back the way the pipeline does."""
    frame = blank_scene(SHAPE)
    render_part_on_board_at_height(frame, outline, x_mm, y_mm, theta_deg, 0.0,
                                   rvec, tvec, K, DIST, RED_BGR, base_mm=z_mm)
    undistorted = cv2.undistort(frame, K, DIST)

    rect = BoardRectifier.around(rvec, tvec, K, centre_mm=(x_mm, y_mm),
                                 radius_mm=70.0, px_per_mm=3.0, z_mm=z_mm)
    patch = rect.warp(undistorted)
    contour = largest_contour(cv2.inRange(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV), *RED_HSV))
    if contour is None:
        return None
    pose = align_outline(contour, outline, rect.px_per_mm)
    x, y = rect.to_board(pose.x_px, pose.y_px)
    return BoardPlanarPose(float(x), float(y), rect.theta_to_board(pose.theta_deg),
                           z_mm, pose.score)


class UndistortedPixelSiblingTests(unittest.TestCase):
    def test_matches_raw_version_when_there_is_no_distortion(self):
        rvec, tvec = board_camera(elev_deg=35.0)

        raw = camera_pixel_to_board(1000.0, 600.0, rvec, tvec, K, np.zeros(5))
        undistorted = board_point_from_undistorted_pixel(1000.0, 600.0, rvec, tvec, K)

        np.testing.assert_allclose(undistorted, raw, atol=1e-9)

    def test_agrees_with_raw_version_after_undistorting_the_pixel(self):
        rvec, tvec = board_camera(elev_deg=35.0)
        truth = np.array([180.0, 40.0, 0.0])
        raw_px = cv2.projectPoints(truth.reshape(1, 3), rvec, tvec, K, DIST)[0].ravel()
        undistorted_px = cv2.undistortPoints(raw_px.reshape(1, 1, 2), K, DIST, P=K).ravel()

        via_raw = camera_pixel_to_board(*raw_px, rvec, tvec, K, DIST)
        via_undistorted = board_point_from_undistorted_pixel(*undistorted_px, rvec, tvec, K)

        np.testing.assert_allclose(via_raw, truth, atol=1e-6)
        np.testing.assert_allclose(via_undistorted, truth, atol=1e-3)

    def test_rejects_a_ray_that_never_meets_the_board(self):
        rvec = np.array([0.0, np.pi / 2, 0.0])
        with self.assertRaises(ValueError):
            board_point_from_undistorted_pixel(K[0, 2], K[1, 2], rvec,
                                               np.array([0.0, 0.0, 500.0]), K)


class BoardScaleTests(unittest.TestCase):
    def test_scale_matches_a_measured_millimetre(self):
        rvec, tvec = board_camera(elev_deg=35.0)
        a, b = np.array([246.5, 152.5, 0.0]), np.array([247.5, 152.5, 0.0])
        projected = cv2.projectPoints(np.array([a, b]), rvec, tvec, K, np.zeros(5))[0].reshape(-1, 2)
        expected = float(np.linalg.norm(projected[1] - projected[0]))

        scale = local_scale_px_per_mm(rvec, tvec, K, BOARD_CENTRE)

        self.assertAlmostEqual(scale, expected, delta=0.15)


class PartPoseInBoardTests(unittest.TestCase):
    def setUp(self):
        self.rvec, self.tvec = board_camera(elev_deg=35.0)

    def test_recovers_position_in_millimetres(self):
        world = recover_pose(246.5, 152.5, 0.0, self.rvec, self.tvec)

        self.assertAlmostEqual(world.x_mm, 246.5, delta=1.5)
        self.assertAlmostEqual(world.y_mm, 152.5, delta=1.5)
        self.assertEqual(world.z_mm, 0.0)

    def test_recovers_position_away_from_the_optical_axis(self):
        world = recover_pose(120.0, 230.0, 0.0, self.rvec, self.tvec)

        self.assertAlmostEqual(world.x_mm, 120.0, delta=2.0)
        self.assertAlmostEqual(world.y_mm, 230.0, delta=2.0)

    def test_recovers_angle_across_full_circle(self):
        for truth in (0.0, 35.0, 90.0, 168.0, 244.0, 300.0):
            with self.subTest(theta=truth):
                world = recover_pose(246.5, 152.5, truth, self.rvec, self.tvec)

                error = abs((world.theta_deg - truth + 180.0) % 360.0 - 180.0)
                self.assertLess(error, 4.0, f"got {world.theta_deg:.1f} want {truth}")

    def test_positive_theta_turns_board_x_toward_board_y(self):
        """Sign check: the part's +x axis at theta=+40 must have a positive
        board-y component. This is the convention the projector side needs."""
        world = recover_pose(246.5, 152.5, 40.0, self.rvec, self.tvec)

        axis = world.axis_board
        self.assertGreater(axis[0], 0.0)
        self.assertGreater(axis[1], 0.0)

    def test_angle_survives_a_change_of_camera_angle(self):
        """The same physical placement must read the same from any viewpoint --
        the mount must never leak into the world frame."""
        for elev in (5.0, 20.0, 35.0, 45.0):
            with self.subTest(elev=elev):
                rvec, tvec = board_camera(elev_deg=elev, azim_deg=200.0)
                world = recover_pose(246.5, 152.5, 55.0, rvec, tvec)

                self.assertAlmostEqual(world.theta_deg, 55.0, delta=4.0)

    def test_angle_survives_a_change_of_camera_azimuth(self):
        for azim in (0.0, 90.0, 200.0, 315.0):
            with self.subTest(azim=azim):
                rvec, tvec = board_camera(elev_deg=35.0, azim_deg=azim)
                world = recover_pose(246.5, 152.5, 55.0, rvec, tvec)

                self.assertAlmostEqual(world.theta_deg, 55.0, delta=4.0)

    def test_a_chiral_part_is_recovered_not_mirrored(self):
        """The regression that raw-image matching could not pass at all."""
        world = recover_pose(246.5, 152.5, 20.0, self.rvec, self.tvec)

        self.assertGreater(world.score, 0.85)


if __name__ == "__main__":
    unittest.main()
