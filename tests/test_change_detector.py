"""Frame differencing: WHERE did something appear? Never WHAT appeared.

Identity is the matcher's job. The diff blob is contaminated by shadow and
projector light, so its shape must never be trusted -- only its location.
"""
import unittest

import cv2
import numpy as np

from perception.change_detector import detect_change, frames_are_still
from tests.synthetic import blank_scene, render_part

BRICK = np.array([(0, 0), (4, 0), (4, 2), (0, 2)], float) * 8.0
PX_PER_MM = 2.8


def scene_with_part(centre, shape=(480, 640), bgr=(200, 70, 30), outline=BRICK):
    return render_part(blank_scene(shape), outline, centre, 0.0, PX_PER_MM, bgr)


class DetectChangeTests(unittest.TestCase):
    def test_workspace_ignores_large_background_change(self):
        before = blank_scene()
        after = scene_with_part((320, 240))
        after[:160] = 0  # More than 25% of the full image changes outside ROI.
        roi = np.zeros(before.shape[:2], np.uint8)
        roi[180:400, 180:500] = 255
        region = detect_change(before, after, search_mask=roi)
        self.assertIsNotNone(region)
        self.assertLess(abs(region.centroid_px[0] - 320), 5)
        self.assertLess(abs(region.centroid_px[1] - 240), 5)
        background_only = before.copy()
        background_only[:160] = 0
        self.assertIsNone(detect_change(before, background_only, search_mask=roi))
        self.assertTrue(frames_are_still(before, background_only, search_mask=roi))
        self.assertFalse(frames_are_still(before, after, search_mask=roi))

    def test_empty_workspace_cannot_detect_or_settle(self):
        before = blank_scene()
        roi = np.zeros(before.shape[:2], np.uint8)
        self.assertIsNone(detect_change(before, before, search_mask=roi))
        self.assertFalse(frames_are_still(before, before, search_mask=roi))

    def test_finds_a_newly_placed_part(self):
        before = blank_scene()
        after = scene_with_part((320, 240))

        region = detect_change(before, after)

        self.assertIsNotNone(region)
        x, y, w, h = region.bbox
        self.assertLess(abs((x + w / 2) - 320), 5)
        self.assertLess(abs((y + h / 2) - 240), 5)

    def test_identical_frames_report_no_change(self):
        frame = scene_with_part((320, 240))
        self.assertIsNone(detect_change(frame, frame.copy()))

    def test_sensor_noise_alone_is_not_a_placement(self):
        before = blank_scene(noise=6)
        after = blank_scene(noise=6)
        self.assertIsNone(detect_change(before, after))

    def test_a_hand_across_the_scene_is_rejected_as_too_large(self):
        """A hand or forearm dominates the diff. Validating then would measure
        the hand, so an oversized change must be refused outright."""
        before = blank_scene()
        after = before.copy()
        cv2.rectangle(after, (40, 40), (600, 430), (150, 170, 200), -1)

        self.assertIsNone(detect_change(before, after))

    def test_a_speck_is_rejected_as_too_small(self):
        before = blank_scene()
        after = before.copy()
        cv2.circle(after, (300, 200), 2, (10, 10, 10), -1)

        self.assertIsNone(detect_change(before, after))

    def test_change_over_a_marker_is_ignored(self):
        """Markers must never become candidate parts: they change appearance
        under projector light and would hijack every step."""
        before = blank_scene()
        after = scene_with_part((150, 150))
        marker_quad = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], float)

        self.assertIsNone(detect_change(before, after, exclude_quads=[marker_quad]))

    def test_a_part_beside_a_marker_is_still_found(self):
        before = blank_scene()
        after = scene_with_part((450, 300))
        marker_quad = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], float)

        region = detect_change(before, after, exclude_quads=[marker_quad])

        self.assertIsNotNone(region)
        self.assertLess(abs((region.bbox[0] + region.bbox[2] / 2) - 450), 6)

    def test_the_largest_change_wins_when_two_things_moved(self):
        before = blank_scene()
        after = render_part(scene_with_part((150, 150), outline=BRICK * 0.4),
                            BRICK, (450, 300), 0.0, PX_PER_MM, (60, 180, 60))

        region = detect_change(before, after)

        self.assertLess(abs((region.bbox[0] + region.bbox[2] / 2) - 450), 6)

    def test_region_reports_its_area(self):
        region = detect_change(blank_scene(), scene_with_part((320, 240)))
        expected = (4 * 8.0 * PX_PER_MM) * (2 * 8.0 * PX_PER_MM)
        self.assertAlmostEqual(region.area_px, expected, delta=0.3 * expected)


class StillnessTests(unittest.TestCase):
    def test_identical_frames_are_still(self):
        frame = scene_with_part((320, 240))
        self.assertTrue(frames_are_still(frame, frame.copy()))

    def test_noise_alone_still_counts_as_still(self):
        self.assertTrue(frames_are_still(blank_scene(noise=6), blank_scene(noise=6)))

    def test_a_moving_part_is_not_still(self):
        self.assertFalse(frames_are_still(scene_with_part((300, 240)),
                                          scene_with_part((340, 240))))


if __name__ == "__main__":
    unittest.main()


class FragmentMergingTests(unittest.TestCase):
    """A part laid over a same-coloured one changes few pixels where they meet,
    splitting one placement into several blobs. Keeping only the largest throws
    away half the silhouette, so nearby fragments are merged into one region."""

    def scene(self, mask):
        before = np.full((480, 640, 3), 200, np.uint8)
        after = before.copy()
        after[mask != 0] = 30
        return before, after

    def test_fragments_split_by_a_thin_gap_are_one_region(self):
        mask = np.zeros((480, 640), np.uint8)
        cv2.rectangle(mask, (200, 220), (300, 260), 255, -1)
        cv2.rectangle(mask, (312, 220), (420, 260), 255, -1)   # 12 px gap
        region = detect_change(*self.scene(mask))
        self.assertIsNotNone(region)
        x, y, w, h = region.bbox
        self.assertLess(x, 205)
        self.assertGreater(x+w, 415, 'Region stops short: the far fragment was dropped')
        self.assertGreater(np.count_nonzero(region.mask), 0.9*np.count_nonzero(mask))

    def test_distant_blobs_remain_separate(self):
        """Merging must not swallow an unrelated object elsewhere on the board."""
        mask = np.zeros((480, 640), np.uint8)
        cv2.rectangle(mask, (40, 40), (140, 90), 255, -1)
        cv2.rectangle(mask, (400, 330), (560, 430), 255, -1)   # far larger blob
        region = detect_change(*self.scene(mask))
        x, y, w, h = region.bbox
        self.assertGreater(x, 300, 'Unrelated distant blob was merged in')
        self.assertEqual(np.count_nonzero(region.mask[40:90, 40:140]), 0,
                         'The far-away smaller blob leaked into the region')
