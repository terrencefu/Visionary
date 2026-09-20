"""Guard against the bug that every other perception test missed.

A camera placed on the WRONG side of the board plane still satisfies the
positive-depth check for flat parts, because a z=0 point is in front of it
either way. But it mirrors every silhouette, and no in-plane rotation can undo
a mirror -- so chiral parts silently stop matching while symmetric ones keep
working. These tests make the mistake impossible to reintroduce unnoticed.
"""
import unittest

import cv2
import numpy as np

from perception.geometry import camera_height_above_board
from perception.pose_estimator import polygon_centroid
from tests.synthetic import board_camera, rotate_in_board

K = np.array([[1110.0, 0, 952.7], [0, 1110.0, 529.7], [0, 0, 1.0]])
DIST = np.zeros(5)
CENTRE = np.array([246.5, 152.5])
L_CHIRAL = np.array([(0, 0), (6, 0), (6, 2), (2, 2), (2, 4), (0, 4)], float) * 8.0


def signed_area(points):
    x, y = np.asarray(points, float)[:, 0], np.asarray(points, float)[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def project_flat_part(rvec, tvec, theta_deg=0.0):
    centred = L_CHIRAL - polygon_centroid(L_CHIRAL)
    board_xy = rotate_in_board(centred, theta_deg) + CENTRE
    return cv2.projectPoints(np.c_[board_xy, np.zeros(len(board_xy))],
                             rvec, tvec, K, DIST)[0].reshape(-1, 2)


def wrong_side_camera(height_mm=700.0):
    """The broken fixture this repo used to have: board +z points AWAY from the
    camera, which puts the camera underneath the table."""
    rvec = np.array([0.0, 0.0, np.pi])
    R = cv2.Rodrigues(rvec)[0]
    return rvec, np.array([0.0, 0.0, height_mm]) - R @ np.array([*CENTRE, 0.0])


class CameraSideTests(unittest.TestCase):
    def test_a_valid_camera_is_above_the_board(self):
        rvec, tvec = board_camera(elev_deg=35.0)
        self.assertGreater(camera_height_above_board(rvec, tvec), 0.0)

    def test_the_wrong_side_camera_is_detected(self):
        rvec, tvec = wrong_side_camera()
        self.assertLess(camera_height_above_board(rvec, tvec), 0.0)

    def test_a_point_above_the_board_is_closer_to_a_valid_camera(self):
        rvec, tvec = board_camera(elev_deg=35.0)
        R = cv2.Rodrigues(rvec)[0]
        on_board = float((R @ np.array([*CENTRE, 0.0]) + tvec)[2])
        raised = float((R @ np.array([*CENTRE, 20.0]) + tvec)[2])

        self.assertLess(raised, on_board, "a raised point must be nearer the camera")

    def test_a_point_above_the_board_is_farther_from_the_wrong_side_camera(self):
        """The signature of the old bug, stated so it reads as a defect."""
        rvec, tvec = wrong_side_camera()
        R = cv2.Rodrigues(rvec)[0]
        on_board = float((R @ np.array([*CENTRE, 0.0]) + tvec)[2])
        raised = float((R @ np.array([*CENTRE, 20.0]) + tvec)[2])

        self.assertGreater(raised, on_board)

class HandednessOfTheRawImageTests(unittest.TestCase):
    """Why matching in raw image space was wrong in the first place.

    A valid overhead camera shows board +X to the right but board +Y UPWARD,
    while part_catalog draws outlines in board axes with y DOWN. Rasterising
    that outline straight into image pixels therefore compares a shape against
    its own mirror, and no in-plane rotation can reconcile the two. Matching
    happens in a rectified board-plane view instead -- see test_rectify.py.
    """

    def test_board_plus_y_appears_upward_in_the_raw_image(self):
        rvec, tvec = board_camera(elev_deg=0.0)
        origin, along_y = np.array([[*CENTRE, 0.0]]), np.array([[CENTRE[0], CENTRE[1] + 50.0, 0.0]])
        pixels = cv2.projectPoints(np.vstack([origin, along_y]), rvec, tvec, K, DIST)[0].reshape(-1, 2)

        self.assertLess(pixels[1][1], pixels[0][1], "board +Y should appear upward on screen")

    def test_raw_image_handedness_is_opposite_to_the_catalogue_convention(self):
        """Stated as a fact to be corrected, not a property to rely on."""
        centred = L_CHIRAL - polygon_centroid(L_CHIRAL)
        board_area = signed_area(rotate_in_board(centred, 0.0) + CENTRE)

        for elev in (0.0, 15.0, 30.0, 45.0):
            with self.subTest(elev=elev):
                rvec, tvec = board_camera(elev_deg=elev)
                image_area = signed_area(project_flat_part(rvec, tvec))
                self.assertLess(board_area * image_area, 0.0)


if __name__ == "__main__":
    unittest.main()
