"""Silhouette alignment: recover in-plane position and angle of a known outline."""
import unittest

import numpy as np

from perception.pose_estimator import align_outline
from tests.synthetic import fill_polygon, largest_contour, place_outline

STUD = 8.0
# Corner plate: arms run right and down. Deliberately not symmetric under any
# rotation, and not its own mirror image.
L_PLATE = np.array([(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)], float) * STUD
# Unequal arms, so this one is genuinely chiral: no rotation maps it onto its
# mirror image. The equal-arm L_PLATE above IS its own mirror (reflect about
# y = x), which is why it cannot be used for the mirror test.
L_CHIRAL = np.array([(0, 0), (6, 0), (6, 2), (2, 2), (2, 4), (0, 4)], float) * STUD
# A part that IS its own mirror image and 180-symmetric, for contrast.
RECTANGLE = np.array([(0, 0), (4, 0), (4, 2), (0, 2)], float) * STUD


def rendered_contour(outline_mm, center_px, theta_deg, px_per_mm, shape=(480, 640)):
    poly = place_outline(outline_mm, center_px, theta_deg, px_per_mm)
    return largest_contour(fill_polygon(shape, poly))


class AlignOutlineTests(unittest.TestCase):
    px_per_mm = 2.8

    def test_recovers_zero_angle_and_centroid(self):
        contour = rendered_contour(L_PLATE, (320, 240), 0.0, self.px_per_mm)

        pose = align_outline(contour, L_PLATE, self.px_per_mm)

        self.assertAlmostEqual(pose.x_px, 320, delta=1.0)
        self.assertAlmostEqual(pose.y_px, 240, delta=1.0)
        self.assertAlmostEqual(pose.theta_deg, 0.0, delta=2.0)
        self.assertGreater(pose.score, 0.9)

    def test_recovers_angle_across_full_circle(self):
        for truth in (17.0, 90.0, 154.0, 200.0, 271.0, 335.0):
            with self.subTest(theta=truth):
                contour = rendered_contour(L_PLATE, (300, 250), truth, self.px_per_mm)

                pose = align_outline(contour, L_PLATE, self.px_per_mm)

                error = abs((pose.theta_deg - truth + 180.0) % 360.0 - 180.0)
                self.assertLess(error, 2.5, f"got {pose.theta_deg:.1f} want {truth}")
                self.assertGreater(pose.score, 0.9)

    def test_positive_theta_is_counter_clockwise_on_screen(self):
        """A +30 deg part must have its long arm swung CCW on screen, i.e. the
        tip that pointed right (+x) now sits ABOVE the centroid (smaller y)."""
        contour = rendered_contour(L_PLATE, (320, 240), 30.0, self.px_per_mm)

        pose = align_outline(contour, L_PLATE, self.px_per_mm)

        self.assertAlmostEqual(pose.theta_deg, 30.0, delta=2.5)
        tip = pose.part_axis_px
        self.assertGreater(tip[0], 0.0, "part +x should still point right-ish")
        self.assertLess(tip[1], 0.0, "part +x should have swung UP on screen")

    def test_returns_angle_in_zero_to_360(self):
        pose = align_outline(rendered_contour(L_PLATE, (320, 240), 350.0, self.px_per_mm),
                             L_PLATE, self.px_per_mm)
        self.assertGreaterEqual(pose.theta_deg, 0.0)
        self.assertLess(pose.theta_deg, 360.0)

    def test_wrong_shape_scores_lower_than_right_shape(self):
        contour = rendered_contour(L_PLATE, (320, 240), 40.0, self.px_per_mm)

        right = align_outline(contour, L_PLATE, self.px_per_mm)
        wrong = align_outline(contour, RECTANGLE, self.px_per_mm)

        self.assertGreater(right.score, 0.9)
        self.assertLess(wrong.score, 0.8)

    def test_mirror_image_is_distinguished(self):
        """The IoU alignment must tell a chiral part from its mirror, which Hu
        moments and minAreaRect cannot."""
        mirrored = L_CHIRAL * np.array([-1.0, 1.0])
        contour = rendered_contour(L_CHIRAL, (320, 240), 25.0, self.px_per_mm)

        right = align_outline(contour, L_CHIRAL, self.px_per_mm)
        flipped = align_outline(contour, mirrored, self.px_per_mm)

        self.assertGreater(right.score, 0.9)
        self.assertGreater(right.score - flipped.score, 0.15)

    def test_self_mirroring_part_cannot_be_flip_detected(self):
        """Known limitation, asserted so it stays known: an equal-arm L is its
        own mirror image, so a flipped placement is invisible. Catalogue parts
        should be chosen chiral."""
        mirrored = L_PLATE * np.array([-1.0, 1.0])
        contour = rendered_contour(L_PLATE, (320, 240), 25.0, self.px_per_mm)

        right = align_outline(contour, L_PLATE, self.px_per_mm)
        flipped = align_outline(contour, mirrored, self.px_per_mm)

        self.assertAlmostEqual(right.score, flipped.score, delta=0.02)

    def test_scale_invariance_across_camera_heights(self):
        for px_per_mm in (1.6, 2.8, 4.5):
            with self.subTest(px_per_mm=px_per_mm):
                contour = rendered_contour(L_PLATE, (320, 240), 63.0, px_per_mm)

                pose = align_outline(contour, L_PLATE, px_per_mm)

                self.assertAlmostEqual(pose.theta_deg, 63.0, delta=3.0)
                self.assertGreater(pose.score, 0.88)


if __name__ == "__main__":
    unittest.main()
