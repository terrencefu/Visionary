"""Expected-part verification: 'we expect part X -- did X appear, and where?'

The matcher never does open-set object recognition. It answers a yes/no
question about one expected part, and only falls back to a catalogue search to
explain a 'no'.
"""
import unittest

import numpy as np

from perception import part_catalog
from perception.part_matcher import HsvOutlineMatcher, MatchResult
from tests.synthetic import blank_scene, render_part

PX_PER_MM = 2.8
CENTRE = (320, 240)
# Saturated BGR paints that land inside the catalogue's HSV ranges.
BGR = {"red": (40, 40, 210), "blue": (200, 70, 30),
       "yellow": (40, 210, 225), "green": (60, 180, 60)}


def scene_with(part_id, colour, centre=CENTRE, theta_deg=0.0, shape=(480, 640)):
    frame = blank_scene(shape)
    outline = part_catalog.get(part_id).outline_mm
    return render_part(frame, outline, centre, theta_deg, PX_PER_MM, BGR[colour])


class CatalogueTests(unittest.TestCase):
    def test_unknown_part_lists_what_is_known(self):
        with self.assertRaises(KeyError) as caught:
            part_catalog.get("no_such_part")
        self.assertIn("red_l_plate", str(caught.exception))

    def test_outline_area_is_computed_from_the_polygon(self):
        signature = part_catalog.get("blue_2x4")
        self.assertAlmostEqual(signature.area_mm2, 4 * 8.0 * 2 * 8.0, delta=1e-6)

    def test_every_catalogue_part_declares_its_rotational_symmetry(self):
        for part_id, signature in part_catalog.CATALOG.items():
            with self.subTest(part=part_id):
                self.assertIn(signature.symmetry_deg, (90, 180, 360))


class VerifyExpectedPartTests(unittest.TestCase):
    def setUp(self):
        self.matcher = HsvOutlineMatcher()

    def test_finds_the_expected_part(self):
        frame = scene_with("red_l_plate", "red", theta_deg=20.0)

        result = self.matcher.verify(frame, None, "red_l_plate", PX_PER_MM)

        self.assertEqual(result.status, "ok")
        self.assertTrue(result.ok)
        self.assertEqual(result.part_id, "red_l_plate")
        self.assertGreater(result.score, 0.85)

    def test_reports_the_pose_it_already_aligned(self):
        frame = scene_with("red_l_plate", "red", centre=(300, 260), theta_deg=20.0)

        result = self.matcher.verify(frame, None, "red_l_plate", PX_PER_MM)

        self.assertAlmostEqual(result.pose_px.x_px, 300, delta=2.0)
        self.assertAlmostEqual(result.pose_px.y_px, 260, delta=2.0)
        self.assertAlmostEqual(result.pose_px.theta_deg, 20.0, delta=3.0)

    def test_empty_scene_is_not_found(self):
        result = self.matcher.verify(blank_scene(), None, "red_l_plate", PX_PER_MM)

        self.assertEqual(result.status, "not_found")
        self.assertFalse(result.ok)
        self.assertIsNone(result.pose_px)

    def test_a_different_catalogue_part_is_named_as_the_wrong_part(self):
        frame = scene_with("blue_2x4", "blue")

        result = self.matcher.verify(frame, None, "red_l_plate", PX_PER_MM)

        self.assertEqual(result.status, "wrong_part")
        self.assertEqual(result.part_id, "blue_2x4")
        self.assertIn("blue_2x4", result.message)

    def test_right_colour_wrong_shape_is_low_confidence_not_wrong_part(self):
        """A red blob that is not the red L must not be confidently renamed."""
        frame = blank_scene()
        blob = np.array([(0, 0), (5, 0), (5, 5), (0, 5)], float) * 8.0
        render_part(frame, blob, CENTRE, 0.0, PX_PER_MM, BGR["red"])

        result = self.matcher.verify(frame, None, "red_l_plate", PX_PER_MM)

        self.assertEqual(result.status, "low_confidence")
        self.assertLess(result.score, 0.85)

    def test_search_is_restricted_to_the_region_it_is_given(self):
        """A part outside the changed region must be ignored, so that parts
        placed in earlier steps do not re-trigger the current step."""
        frame = scene_with("red_l_plate", "red", centre=(120, 120))

        far_away = (400, 300, 200, 150)   # x, y, w, h
        result = self.matcher.verify(frame, far_away, "red_l_plate", PX_PER_MM)

        self.assertEqual(result.status, "not_found")

    def test_finds_the_part_inside_the_region_it_is_given(self):
        frame = scene_with("red_l_plate", "red", centre=(420, 300))

        region = (330, 220, 180, 170)
        result = self.matcher.verify(frame, region, "red_l_plate", PX_PER_MM)

        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.pose_px.x_px, 420, delta=3.0)


class IdentifyTests(unittest.TestCase):
    def test_identify_names_the_best_matching_catalogue_part(self):
        matcher = HsvOutlineMatcher()
        frame = scene_with("yellow_l_plate", "yellow", theta_deg=70.0)

        result = matcher.identify(frame, None, PX_PER_MM)

        self.assertEqual(result.part_id, "yellow_l_plate")
        self.assertGreater(result.score, 0.85)


class SwappableMatcherTests(unittest.TestCase):
    def test_a_stub_matcher_satisfies_the_interface(self):
        """The CAD-render matcher must be able to replace this one without the
        pipeline knowing. Only `verify` is required."""
        class StubMatcher:
            def verify(self, frame, region, expected_part_id, px_per_mm):
                return MatchResult(status="ok", part_id=expected_part_id,
                                   contour=None, score=1.0, pose_px=None, message="stub")

        result = StubMatcher().verify(None, None, "anything", 1.0)
        self.assertTrue(result.ok)


if __name__ == "__main__":
    unittest.main()
