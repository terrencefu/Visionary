"""Part pose in board millimetres, through a real lens and an upside-down camera.

These tests pin the yaw convention the README leaves open:

    theta_deg is the angle of the part's +x axis, measured in the BOARD frame,
    CCW from board +x toward board +y, in [0, 360).

and the position convention:

    x_mm / y_mm are the AREA CENTROID of the part silhouette, not its outline
    origin. A CAD target pose must use the same reference point.
"""
import unittest

import cv2
import numpy as np

from perception.geometry import board_point_from_undistorted_pixel, camera_pixel_to_board
from perception.pose_estimator import (align_outline, board_scale_px_per_mm,
                                       planar_pose_to_board, polygon_centroid)
from tests.synthetic import fill_polygon, largest_contour

STUD = 8.0
L_CHIRAL = np.array([(0, 0), (6, 0), (6, 2), (2, 2), (2, 4), (0, 4)], float) * STUD

K = np.array([[1500.0, 0, 960.0], [0, 1490.0, 540.0], [0, 0, 1.0]])
DIST = np.array([0.05, -0.02, 0.001, -0.001, 0.0])
SHAPE = (1080, 1920)
BOARD_CENTRE = np.array([125.0, 95.0])


def upside_down_camera(tilt=(0.02, -0.03), height_mm=600.0, look_at=BOARD_CENTRE):
    """Camera ~600 mm above the board, rolled 180 deg (mounted upside down),
    with a little tilt. `look_at` lands on the optical axis."""
    rvec = np.array([tilt[0], tilt[1], np.pi])
    R = cv2.Rodrigues(rvec)[0]
    tvec = np.array([0.0, 0.0, height_mm]) - R @ np.array([look_at[0], look_at[1], 0.0])
    return rvec, tvec


def rotate_in_board(points, theta_deg):
    """Rotate (N,2) board-frame points CCW from +x toward +y."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.asarray(points, float) @ np.array([[c, -s], [s, c]]).T


def render_part_on_board(outline_mm, x_mm, y_mm, theta_deg, rvec, tvec, shape=SHAPE):
    """Project a part lying flat on the board into the distorted camera image,
    and return its observed contour. The outline's AREA CENTROID is placed at
    (x_mm, y_mm)."""
    centred = np.asarray(outline_mm, float) - polygon_centroid(outline_mm)
    board_xy = rotate_in_board(centred, theta_deg) + np.array([x_mm, y_mm])
    board_xyz = np.c_[board_xy, np.zeros(len(board_xy))]
    image_pts = cv2.projectPoints(board_xyz, rvec, tvec, K, DIST)[0].reshape(-1, 2)
    return largest_contour(fill_polygon(shape, image_pts))


class UndistortedPixelSiblingTests(unittest.TestCase):
    def test_matches_raw_version_when_there_is_no_distortion(self):
        rvec, tvec = upside_down_camera()
        zero = np.zeros(5)

        raw = camera_pixel_to_board(1000.0, 600.0, rvec, tvec, K, zero)
        undistorted = board_point_from_undistorted_pixel(1000.0, 600.0, rvec, tvec, K)

        np.testing.assert_allclose(undistorted, raw, atol=1e-9)

    def test_agrees_with_raw_version_after_undistorting_the_pixel(self):
        rvec, tvec = upside_down_camera()
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
            board_point_from_undistorted_pixel(960.0, 540.0, rvec, np.array([0.0, 0.0, 500.0]), K)


class BoardScaleTests(unittest.TestCase):
    def test_scale_matches_a_measured_millimetre(self):
        rvec, tvec = upside_down_camera()
        a, b = np.array([125.0, 95.0, 0.0]), np.array([126.0, 95.0, 0.0])
        projected = cv2.projectPoints(np.array([a, b]), rvec, tvec, K, DIST)[0].reshape(-1, 2)
        expected = float(np.linalg.norm(projected[1] - projected[0]))

        scale = board_scale_px_per_mm(BOARD_CENTRE, rvec, tvec, K, DIST)

        self.assertAlmostEqual(scale, expected, delta=0.05)

    def test_scale_is_about_focal_length_over_height(self):
        rvec, tvec = upside_down_camera(height_mm=600.0)
        self.assertAlmostEqual(board_scale_px_per_mm(BOARD_CENTRE, rvec, tvec, K, DIST),
                               1495.0 / 600.0, delta=0.1)


class PartPoseInBoardTests(unittest.TestCase):
    def setUp(self):
        self.rvec, self.tvec = upside_down_camera()

    def recover(self, x_mm, y_mm, theta_deg, outline=L_CHIRAL):
        contour = render_part_on_board(outline, x_mm, y_mm, theta_deg, self.rvec, self.tvec)
        self.assertIsNotNone(contour, "part did not render inside the frame")
        scale = board_scale_px_per_mm((x_mm, y_mm), self.rvec, self.tvec, K, DIST)
        pose_px = align_outline(contour, outline, scale)
        return planar_pose_to_board(pose_px, self.rvec, self.tvec, K, DIST)

    def test_recovers_position_in_millimetres(self):
        world = self.recover(125.0, 95.0, 0.0)

        self.assertAlmostEqual(world.x_mm, 125.0, delta=1.0)
        self.assertAlmostEqual(world.y_mm, 95.0, delta=1.0)
        self.assertEqual(world.z_mm, 0.0)

    def test_recovers_position_away_from_the_optical_axis(self):
        world = self.recover(60.0, 150.0, 0.0)

        self.assertAlmostEqual(world.x_mm, 60.0, delta=1.5)
        self.assertAlmostEqual(world.y_mm, 150.0, delta=1.5)

    def test_recovers_angle_across_full_circle(self):
        for truth in (0.0, 35.0, 90.0, 168.0, 244.0, 300.0):
            with self.subTest(theta=truth):
                world = self.recover(125.0, 95.0, truth)

                error = abs((world.theta_deg - truth + 180.0) % 360.0 - 180.0)
                self.assertLess(error, 3.0, f"got {world.theta_deg:.1f} want {truth}")

    def test_positive_theta_turns_board_x_toward_board_y(self):
        """Sign check: the part's +x axis at theta=+40 must have a positive
        board-y component. This is the convention the projector side needs."""
        world = self.recover(125.0, 95.0, 40.0)

        axis = world.axis_board
        self.assertGreater(axis[0], 0.0)
        self.assertGreater(axis[1], 0.0)

    def test_angle_survives_the_upside_down_mount(self):
        """Same physical placement, camera rolled by 180 deg, must report the
        same board angle -- the mount must not leak into the world frame."""
        upright = np.array([0.02, -0.03, 0.0])
        R = cv2.Rodrigues(upright)[0]
        tvec = np.array([0.0, 0.0, 600.0]) - R @ np.array([125.0, 95.0, 0.0])

        contour = render_part_on_board(L_CHIRAL, 125.0, 95.0, 55.0, upright, tvec)
        scale = board_scale_px_per_mm(BOARD_CENTRE, upright, tvec, K, DIST)
        upright_world = planar_pose_to_board(align_outline(contour, L_CHIRAL, scale),
                                             upright, tvec, K, DIST)

        flipped_world = self.recover(125.0, 95.0, 55.0)

        self.assertAlmostEqual(upright_world.theta_deg, 55.0, delta=3.0)
        self.assertAlmostEqual(flipped_world.theta_deg, 55.0, delta=3.0)


if __name__ == "__main__":
    unittest.main()
