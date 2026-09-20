"""Rectify the board plane to a metric top-down view before matching anything.

This is what makes a 30-45 degree camera workable. Three problems collapse into
one fix:

  * perspective foreshortening -- a tilted plane compresses by ~cos(elevation),
  * handedness -- the raw image shows board +Y upward, the catalogue draws it
    downward, so raw-space template matching compares a shape to its mirror,
  * scale -- px/mm varies across a tilted view.

In the rectified view board +X is right, board +Y is DOWN, and the scale is an
exact constant we chose, all matching part_catalog's drawing convention.
"""
import unittest

import cv2
import numpy as np

from perception.rectify import BoardRectifier, board_to_image_homography
from perception.pose_estimator import polygon_centroid
from tests.synthetic import (blank_scene, board_camera, render_part_on_board_at_height,
                             rotate_in_board)

K = np.array([[1110.0, 0, 952.7], [0, 1110.0, 529.7], [0, 0, 1.0]])
DIST = np.zeros(5)
SHAPE = (1080, 1920)
CENTRE = np.array([246.5, 152.5])
L_CHIRAL = np.array([(0, 0), (6, 0), (6, 2), (2, 2), (2, 4), (0, 4)], float) * 8.0
RED = (40, 40, 210)


def signed_area(points):
    x, y = np.asarray(points, float)[:, 0], np.asarray(points, float)[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


class HomographyTests(unittest.TestCase):
    def test_matches_project_points_on_the_board_plane(self):
        rvec, tvec = board_camera(elev_deg=35.0)
        H = board_to_image_homography(rvec, tvec, K)
        board = np.array([[100.0, 60.0], [300.0, 200.0], [246.5, 152.5]])

        through_h = cv2.perspectiveTransform(board.reshape(-1, 1, 2), H).reshape(-1, 2)
        expected = cv2.projectPoints(np.c_[board, np.zeros(len(board))],
                                     rvec, tvec, K, DIST)[0].reshape(-1, 2)

        np.testing.assert_allclose(through_h, expected, atol=1e-6)

    def test_an_elevated_plane_projects_differently(self):
        """The z=h homography is what keeps stacked parts from being measured
        on the wrong plane."""
        rvec, tvec = board_camera(elev_deg=35.0)
        point = np.array([[300.0, 200.0]])

        on_board = cv2.perspectiveTransform(point.reshape(-1, 1, 2),
                                            board_to_image_homography(rvec, tvec, K)).ravel()
        raised = cv2.perspectiveTransform(point.reshape(-1, 1, 2),
                                          board_to_image_homography(rvec, tvec, K, z_mm=20.0)).ravel()

        self.assertGreater(np.linalg.norm(raised - on_board), 5.0)


class RectifiedViewTests(unittest.TestCase):
    def setUp(self):
        self.rvec, self.tvec = board_camera(elev_deg=35.0)

    def rectifier(self, px_per_mm=3.0, origin=(150.0, 80.0), size_mm=(200.0, 150.0)):
        return BoardRectifier(self.rvec, self.tvec, K, origin_mm=origin,
                              size_mm=size_mm, px_per_mm=px_per_mm)

    def test_rectified_pixels_map_back_to_the_board_metrically(self):
        rect = self.rectifier(px_per_mm=3.0, origin=(150.0, 80.0))

        np.testing.assert_allclose(rect.to_board(0.0, 0.0), [150.0, 80.0], atol=1e-6)
        np.testing.assert_allclose(rect.to_board(30.0, 60.0), [160.0, 100.0], atol=1e-6)

    def test_board_plus_y_is_downward_in_the_rectified_view(self):
        """The catalogue's convention, restored."""
        rect = self.rectifier()
        origin = rect.to_rect(200.0, 100.0)
        along_y = rect.to_rect(200.0, 140.0)

        self.assertGreater(along_y[1], origin[1])
        self.assertAlmostEqual(along_y[0], origin[0], delta=1e-6)

    def test_rectified_view_has_the_catalogue_handedness(self):
        """The mirror is gone: board and rectified signed areas now agree."""
        centred = L_CHIRAL - polygon_centroid(L_CHIRAL)
        board_xy = rotate_in_board(centred, 0.0) + CENTRE
        rect = self.rectifier(origin=(150.0, 80.0), size_mm=(200.0, 150.0))

        in_rect = np.array([rect.to_rect(x, y) for x, y in board_xy])

        self.assertGreater(signed_area(board_xy) * signed_area(in_rect), 0.0)

    def test_a_flat_part_keeps_its_true_proportions_under_a_tilted_camera(self):
        """Foreshortening removed: a 48x32 mm part measures 48x32 mm again."""
        frame = blank_scene(SHAPE)
        render_part_on_board_at_height(frame, L_CHIRAL, 246.5, 152.5, 0.0, 0.0,
                                       self.rvec, self.tvec, K, DIST, RED)
        rect = BoardRectifier(self.rvec, self.tvec, K, origin_mm=(180.0, 100.0),
                              size_mm=(140.0, 110.0), px_per_mm=3.0)

        patch = rect.warp(frame)
        mask = cv2.inRange(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV),
                           np.array([0, 120, 70], np.uint8), np.array([8, 255, 255], np.uint8))
        xs, ys = np.nonzero(mask)[1], np.nonzero(mask)[0]
        width_mm = (xs.max() - xs.min()) / 3.0
        height_mm = (ys.max() - ys.min()) / 3.0

        self.assertAlmostEqual(width_mm, 48.0, delta=2.0)
        self.assertAlmostEqual(height_mm, 32.0, delta=2.0)

    def test_the_patch_has_the_requested_pixel_size(self):
        rect = self.rectifier(px_per_mm=2.0, size_mm=(200.0, 150.0))
        patch = rect.warp(blank_scene(SHAPE))
        self.assertEqual(patch.shape[:2], (300, 400))

    def test_a_window_can_be_built_around_a_board_point(self):
        rect = BoardRectifier.around(self.rvec, self.tvec, K, centre_mm=(246.5, 152.5),
                                     radius_mm=60.0, px_per_mm=3.0)

        np.testing.assert_allclose(rect.to_board(*rect.centre_px), [246.5, 152.5], atol=0.5)

    def test_scale_is_chosen_not_to_upsample_when_asked_to_auto_pick(self):
        """Rectifying far above the source resolution just invents pixels."""
        rect = BoardRectifier.around(self.rvec, self.tvec, K, centre_mm=(246.5, 152.5),
                                     radius_mm=60.0, px_per_mm=None)
        self.assertGreater(rect.px_per_mm, 0.3)
        self.assertLess(rect.px_per_mm, 4.0)


class ElevatedPlaneTests(unittest.TestCase):
    def test_a_thin_part_on_a_raised_layer_needs_its_own_plane(self):
        """A part resting on an earlier layer, measured at z=0, lands in the
        wrong place; measured on its own plane it does not. Kept thin so this
        isolates the plane offset from the separate 3D side-face problem."""
        rvec, tvec = board_camera(elev_deg=35.0)
        height = 20.0
        frame = blank_scene(SHAPE)
        render_part_on_board_at_height(frame, L_CHIRAL, 246.5, 152.5, 0.0, 2.0,
                                       rvec, tvec, K, DIST, RED, base_mm=height)

        def measured_centre(z_mm):
            rect = BoardRectifier.around(rvec, tvec, K, centre_mm=(246.5, 152.5),
                                         radius_mm=80.0, px_per_mm=3.0, z_mm=z_mm)
            patch = rect.warp(frame)
            mask = cv2.inRange(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV),
                               np.array([0, 120, 70], np.uint8), np.array([8, 255, 255], np.uint8))
            ys, xs = np.nonzero(mask)
            return rect.to_board(xs.mean(), ys.mean())

        on_wrong_plane = measured_centre(0.0)
        on_right_plane = measured_centre(height)
        truth = np.array([246.5, 152.5])

        self.assertGreater(np.linalg.norm(on_wrong_plane - truth), 5.0)
        self.assertLess(np.linalg.norm(on_right_plane - truth), 3.0)


if __name__ == "__main__":
    unittest.main()
