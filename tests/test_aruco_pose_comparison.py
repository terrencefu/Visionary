import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
import config
from calibration.aruco_debug import audit_marker_geometry, save_debug
from calibration.aruco_pose_diagnostics import (annotate_comparison, compare_poses,
                                               comparison_lines, local_corners,
                                               parse_rotations, physical_corners)
from hardware.camera import preview_frame
from perception.aruco import BoardTracker, make_detector, marker_corners


class PoseComparisonTests(unittest.TestCase):
    def setUp(self):
        self.settings = patch.multiple(config, MARKER_CENTERS_MM={0:(0.,0.), 1:(493.,0.), 2:(0.,305.), 3:(493.,305.)},
                                       MARKER_SIZE_MM=30., MAX_ARUCO_RMS_PX=2.,
                                       MIN_VISIBLE_MARKERS=3, BOARD_BOUNDS_MM=None)
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.K = np.array([[1200., 0., 960.], [0., 1210., 540.], [0., 0., 1.]])
        self.dist = np.array([-.1, .02, .001, -.001, .001])
        self.r = np.array([.2, -.1, .1])
        self.t = np.array([-220., -120., 750.])
        self.raw = np.zeros((1080, 1920, 3), np.uint8)
        self.raw[0, 0] = [15, 25, 35]

    def scene(self, heights=None, actual_rotations=None, ids=(0,1,2,3), declared=None):
        corners = []
        for i in ids:
            obj = physical_corners(i, (actual_rotations or {}).get(i, 0.)).copy()
            obj[:, 2] += (heights or {}).get(i, 0.)
            corners.append(cv2.projectPoints(obj, self.r, self.t, self.K, self.dist)[0].reshape(1,4,2))
        tracker = BoardTracker(self.K, self.dist)
        tracker.detector = Mock()
        tracker.detector.detectMarkers.return_value = (corners, np.array(ids).reshape(-1,1), [])
        tracker.estimate(self.raw)
        debug = tracker.last_diagnostics
        audit, note = audit_marker_geometry(debug, self.K, self.dist)
        report = compare_poses(debug, self.K, self.dist, parse_rotations(declared or []), audit)
        return report, debug, audit, note

    def test_flat_fixture_fits_each_marker_and_both_boards(self):
        report, debug, _, _ = self.scene()
        self.assertTrue(report['individual_solvers_succeeded'])
        self.assertTrue(report['individuals_pass_checks'])
        self.assertFalse(report['individuals_pass_combined_fails'])
        for item in report['individuals']:
            self.assertLess(item['rms_px'], 1e-4)
        self.assertLess(report['full_board']['rms_px'], 1e-4)
        self.assertLess(report['subset_013']['rms_px'], 1e-4)
        self.assertAlmostEqual(report['full_board']['rms_px'], debug.rms_error_px, places=9)

    def test_row_fits_evaluate_but_do_not_fit_opposite_row(self):
        flat, _, _, _ = self.scene()
        moved, _, _, _ = self.scene(heights={2:100., 3:100.})
        for name, ids in [('top_01', [0, 1]), ('bottom_23', [2, 3])]:
            fit = flat['row_fits'][name]
            self.assertEqual(fit['ids'], ids)
            self.assertLess(fit['rms_px'], 1e-4)
            for i, error in fit['per_marker_residuals'].items():
                self.assertEqual(error['in_fit'], i in ids)
                self.assertLess(error['rms_px'], 1e-4)
        top = moved['row_fits']['top_01']
        self.assertLess(top['rms_px'], 1e-4)
        self.assertGreater(top['per_marker_residuals'][2]['rms_px'], 5.)
        self.assertGreater(top['per_marker_residuals'][3]['rms_px'], 5.)

    def test_lifted_marker_two_is_held_out_of_subset_not_of_evaluation(self):
        # Deliberately large synthetic separation exercises the failure branch;
        # it is not an estimate of the physical fixture's lift.
        report, _, _, _ = self.scene(heights={2:100.})
        self.assertTrue(report['individuals_pass_combined_fails'])
        self.assertTrue(report['subset_013']['passes_checks'])
        self.assertEqual(report['subset_013']['ids'], [0,1,3])
        residual = report['subset_013']['per_marker_residuals'][2]
        self.assertFalse(residual['in_fit'])
        self.assertGreater(residual['rms_px'], 5.)
        self.assertIn('marker-2-specific', ' '.join(report['interpretation']))

    def test_other_lifted_markers_can_keep_subset_failing(self):
        report, _, _, _ = self.scene(heights={0:300., 2:100.})
        self.assertTrue(report['individuals_pass_combined_fails'])
        self.assertFalse(report['subset_013']['passes_checks'])
        self.assertIn('removing marker 2 alone does not resolve', ' '.join(report['interpretation']))

    def test_rotation_override_only_changes_diagnostic_mapping(self):
        before = copy.deepcopy(config.MARKER_CENTERS_MM)
        K, dist = self.K.copy(), self.dist.copy()
        wrong, debug, audit, _ = self.scene(actual_rotations={2:90.})
        self.assertFalse(wrong['full_board']['passes_checks'])
        self.assertAlmostEqual(wrong['corner_mapping'][2]['estimated_rotation_deg'], 90., delta=.1)
        original_rms = debug.rms_error_px
        right = compare_poses(debug, self.K, self.dist, parse_rotations(['2=90']), audit)
        self.assertTrue(right['full_board']['passes_checks'])
        self.assertLess(right['subset_013']['per_marker_residuals'][2]['rms_px'], 1e-4)
        self.assertEqual(right['corner_mapping'][2]['physical_corner_labels'], ['TR','BR','BL','TL'])
        self.assertEqual(debug.rms_error_px, original_rms)
        self.assertEqual(config.MARKER_CENTERS_MM, before)
        self.assertEqual(config.MARKER_SIZE_MM, 30.)
        self.assertEqual(config.MAX_ARUCO_RMS_PX, 2.)
        np.testing.assert_array_equal(self.K, K)
        np.testing.assert_array_equal(self.dist, dist)

    def test_missing_subset_marker_does_not_silently_change_subset(self):
        report, _, _, _ = self.scene(ids=(0,2,3))
        self.assertFalse(report['subset_013']['available'])
        self.assertEqual(report['subset_013']['missing_ids'], [1])
        self.assertIsNone(report['subset_013']['solvepnp_succeeded'])
        self.assertIsNone(report['individuals_pass_combined_fails'])

    def test_duplicate_ids_do_not_make_a_valid_subset(self):
        report, _, _, _ = self.scene(ids=(0,1,1,2,3))
        self.assertEqual(report['duplicate_ids'], [1])
        self.assertFalse(report['subset_013']['available'])
        self.assertEqual(len(report['individuals']), 5)

    def test_unknown_rotation_inputs_fail_clearly(self):
        for value in ['oops', '9=90', '2=nan', 'x=0']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_rotations([value])

    def test_numbered_raw_frame_and_saved_report(self):
        report, debug, audit, note = self.scene(heights={2:100.})
        before = self.raw.copy()
        with patch('calibration.aruco_pose_diagnostics.cv2.putText', wraps=cv2.putText) as put:
            view = annotate_comparison(self.raw, debug, report)
        labels = [call.args[1] for call in put.call_args_list]
        for i in range(4):
            self.assertIn(f'D{i}', labels)
            self.assertIn(f'P{i}', labels)
        np.testing.assert_array_equal(self.raw, before)
        np.testing.assert_array_equal(view[0,0], self.raw[0,0])
        with patch.object(config, 'PREVIEW_ROTATE_180', True):
            np.testing.assert_array_equal(preview_frame(view, rotate=False), view)
        with tempfile.TemporaryDirectory() as folder:
            save_debug(folder, self.raw, view, debug, audit, note, report)
            saved = json.loads((Path(folder)/'report.json').read_text())
            self.assertTrue(saved['pose_comparison']['individuals_pass_combined_fails'])
            np.testing.assert_array_equal(cv2.imread(str(Path(folder)/'correspondence_raw.png')), view)
        text = '\n'.join(comparison_lines(report, detailed=True))
        self.assertIn('ID 2 (HELD OUT)', text)
        self.assertIn('corner 0 (TL): board', text)

    def test_detection_order_does_not_change_full_or_subset_correspondences(self):
        a, _, _, _ = self.scene(heights={2:40.}, ids=(0,1,2,3))
        b, _, _, _ = self.scene(heights={2:40.}, ids=(2,3,0,1))
        self.assertAlmostEqual(a['full_board']['rms_px'], b['full_board']['rms_px'], places=9)
        self.assertAlmostEqual(a['subset_013']['rms_px'], b['subset_013']['rms_px'], places=9)

    def test_real_detector_canonical_corner_order_for_each_quarter_turn(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 0, 160)
        original = np.array([[80.,80.], [239.,80.], [239.,239.], [80.,239.]])
        detector = make_detector()
        for turn in range(4):
            with self.subTest(turn=turn):
                image = np.full((320,320), 255, np.uint8)
                image[80:240,80:240] = np.rot90(marker, -turn)
                corners, ids, _ = detector.detectMarkers(image)
                self.assertEqual(ids.ravel().tolist(), [0])
                expected = np.roll(original, -turn, axis=0)
                np.testing.assert_allclose(corners[0].reshape(4,2), expected, atol=1.)
                # No screen-based reordering: canonical corner 0 follows the print.
                offset = physical_corners(0, turn*90)[:, :2]
                angle = np.radians(turn*90)
                rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                np.testing.assert_allclose(offset, local_corners()[:,:2] @ rot.T, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
