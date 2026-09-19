import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import cv2
import numpy as np
import config
from calibration.aruco_debug import (annotate, audit_marker_geometry, diagnostic_lines,
                                     fixture_lines, save_debug)
from hardware.camera import preview_frame
from perception.aruco import BoardTracker, marker_corners


class ArucoDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.centers = {0: (0., 0.), 1: (493., 0.), 2: (.36, 310.), 3: (493.33, 305.)}
        self.settings = patch.multiple(config, MARKER_CENTERS_MM=self.centers,
                                       MARKER_SIZE_MM=30., MIN_VISIBLE_MARKERS=3,
                                       MAX_ARUCO_RMS_PX=2., BOARD_BOUNDS_MM=None)
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.K = np.array([[1110., 0., 953.], [0., 1111., 530.], [0., 0., 1.]])
        self.dist = np.array([-.13, -.34, .001, .00005, .44])
        self.rvec = np.array([.2, -.1, .1])
        self.tvec = np.array([-240., -150., 1200.])
        self.raw = np.zeros((1080, 1920, 3), np.uint8)
        self.tracker = BoardTracker(self.K, self.dist)
        self.tracker.detector = Mock()
        self.corners = [cv2.projectPoints(marker_corners(i), self.rvec, self.tvec,
                                         self.K, self.dist)[0].reshape(1, 4, 2) for i in range(4)]
        self.detect([0, 1, 2, 3])

    def detect(self, ids):
        corners = [self.corners[i if i in self.centers else 0] for i in ids]
        self.tracker.detector.detectMarkers.return_value = (corners, np.array(ids).reshape(-1, 1) if ids else None, [])

    def test_acceptance_retains_exact_solver_and_raw_input(self):
        before = self.raw.copy()
        with patch('perception.aruco.cv2.solvePnP', wraps=cv2.solvePnP) as solve:
            pose = self.tracker.estimate(self.raw)
        self.assertIsNotNone(pose)
        self.assertEqual(solve.call_args.kwargs['flags'], cv2.SOLVEPNP_ITERATIVE)
        self.tracker.detector.detectMarkers.assert_called_once_with(self.raw)
        debug = self.tracker.last_diagnostics
        self.assertEqual(debug.detected_ids, [0, 1, 2, 3])
        self.assertEqual(debug.known_ids, [0, 1, 2, 3])
        self.assertEqual(debug.used_marker_count, 4)
        self.assertTrue(debug.solvepnp_succeeded)
        self.assertIsNone(debug.rejection_reason)
        self.assertEqual(debug.rms_error_px, pose.reprojection_error)
        np.testing.assert_array_equal(self.raw, before)

    def test_unknown_ids_and_too_few_known_skip_solver(self):
        self.detect([0, 1, 40, 49])
        with patch('perception.aruco.cv2.solvePnP') as solve:
            self.assertIsNone(self.tracker.estimate(self.raw))
            solve.assert_not_called()
        d = self.tracker.last_diagnostics
        self.assertEqual(d.detected_ids, [0, 1, 40, 49])
        self.assertEqual(d.known_ids, [0, 1])
        self.assertEqual(d.unknown_ids, [40, 49])
        self.assertEqual(d.known_marker_count, 2)
        self.assertEqual(d.used_marker_count, 0)
        self.assertEqual(d.rejection_code, 'insufficient_known_markers')
        self.assertIn('Only 2 known markers; need 3', d.rejection_reason)
        self.assertIsNone(d.mean_error_px)

    def test_no_detection_resets_previous_success(self):
        self.tracker.estimate(self.raw)
        self.detect([])
        self.assertIsNone(self.tracker.estimate(self.raw))
        d = self.tracker.last_diagnostics
        self.assertEqual(d.rejection_code, 'no_markers')
        self.assertEqual(d.used_marker_count, 0)
        self.assertIsNone(d.rms_error_px)
        self.assertIsNone(d.solvepnp_succeeded)

    def test_duplicate_known_id_rejected_before_solver(self):
        self.detect([0, 1, 1, 2])
        self.assertIsNone(self.tracker.estimate(self.raw))
        self.assertEqual(self.tracker.last_diagnostics.rejection_code, 'duplicate_ids')
        self.assertIn('[1]', self.tracker.last_diagnostics.rejection_reason)
        self.assertFalse(self.tracker.last_diagnostics.solvepnp_attempted)

    def test_failed_solver_has_no_error_metrics(self):
        with patch('perception.aruco.cv2.solvePnP', return_value=(False, None, None)):
            self.assertIsNone(self.tracker.estimate(self.raw))
        d = self.tracker.last_diagnostics
        self.assertTrue(d.solvepnp_attempted)
        self.assertFalse(d.solvepnp_succeeded)
        self.assertEqual(d.rejection_code, 'solvepnp_failed')
        self.assertIsNone(d.max_error_px)

    def test_collector_accepts_three_markers_at_five_px_only_with_valid_pose(self):
        self.detect([0, 1, 3])
        tracker = BoardTracker(self.K, self.dist, max_rms_px=config.COLLECTOR_MAX_ARUCO_RMS_PX)
        tracker.detector = self.tracker.detector
        pixels = np.concatenate([self.corners[i] for i in [0, 1, 3]], axis=1).reshape(-1, 1, 2)
        cases = [(True, 5., 1000., True), (True, 5.01, 1000., False),
                 (False, 0., 1000., False), (True, 0., -1000., False),
                 (True, 0., float('nan'), False)]
        for success, error, depth, accepted in cases:
            projected = pixels + np.array([error, 0.])
            with self.subTest(success=success, error=error, depth=depth), \
                    patch('perception.aruco.cv2.solvePnP', return_value=(success, np.zeros(3), np.array([0.,0.,depth]))), \
                    patch('perception.aruco.cv2.projectPoints', return_value=(projected, None)):
                self.assertEqual(tracker.estimate(self.raw) is not None, accepted)
                self.assertEqual(tracker.last_diagnostics.used_marker_count, 3)
                self.assertEqual(tracker.last_diagnostics.max_acceptable_rms_px, 5.)

    def test_collector_override_does_not_relax_default_tracker(self):
        self.detect([0, 1, 3])
        projected = np.concatenate([self.corners[i] for i in [0,1,3]], axis=1).reshape(-1,1,2) + [4., 0.]
        with patch('perception.aruco.cv2.projectPoints', return_value=(projected, None)):
            self.assertIsNone(self.tracker.estimate(self.raw))
        self.assertEqual(self.tracker.last_diagnostics.max_acceptable_rms_px, 2.)

    def test_solver_exception_still_propagates_but_retains_reason(self):
        with patch('perception.aruco.cv2.solvePnP', side_effect=cv2.error('test solve failure')):
            with self.assertRaises(cv2.error):
                self.tracker.estimate(self.raw)
        self.assertEqual(self.tracker.last_diagnostics.rejection_code, 'solvepnp_exception')
        self.assertIn('test solve failure', self.tracker.last_diagnostics.rejection_reason)

    def test_mean_max_and_rms_are_distinct_and_gate_remains_rms(self):
        projected = np.concatenate(self.corners, axis=1).reshape(-1, 1, 2).copy()
        projected[0, 0, 0] += 10
        with patch('perception.aruco.cv2.projectPoints', return_value=(projected, None)):
            self.assertIsNone(self.tracker.estimate(self.raw))
        d = self.tracker.last_diagnostics
        self.assertAlmostEqual(d.mean_error_px, .625)
        self.assertAlmostEqual(d.max_error_px, 10.)
        self.assertAlmostEqual(d.rms_error_px, 2.5)
        self.assertEqual(d.max_acceptable_rms_px, 2.)
        self.assertEqual(d.rejection_code, 'reprojection_error')
        self.assertIn('2.500 px exceeds limit 2.000 px', d.rejection_reason)
        self.assertAlmostEqual(d.per_marker_errors[0]['max'], 10.)
        self.assertAlmostEqual(d.per_marker_errors[3]['max'], 0.)
        text = '\n'.join(diagnostic_lines(d))
        self.assertIn('Used for solvePnP: 4 markers / 16 corners', text)
        self.assertIn('solvePnP success: True', text)

    def test_projection_exception_preserves_solver_success_and_error_reason(self):
        with patch('perception.aruco.cv2.projectPoints', side_effect=cv2.error('test projection failure')):
            with self.assertRaises(cv2.error):
                self.tracker.estimate(self.raw)
        d = self.tracker.last_diagnostics
        self.assertTrue(d.solvepnp_succeeded)
        self.assertEqual(d.rejection_code, 'projection_exception')
        self.assertIn('test projection failure', d.rejection_reason)
        self.assertIsNone(d.mean_error_px)

    def test_nonpositive_depth_is_reported_even_when_reprojection_passes(self):
        projected = np.concatenate(self.corners, axis=1).reshape(-1, 1, 2)
        with patch('perception.aruco.cv2.solvePnP', return_value=(True, np.zeros(3), np.array([0., 0., -1000.]))), \
                patch('perception.aruco.cv2.projectPoints', return_value=(projected, None)):
            self.assertIsNone(self.tracker.estimate(self.raw))
        d = self.tracker.last_diagnostics
        self.assertEqual(d.rejection_code, 'nonpositive_depth')
        self.assertIn('16/16', d.rejection_reason)
        self.assertAlmostEqual(d.rms_error_px, 0.)

    def test_nonfinite_residuals_are_reported_and_json_is_valid(self):
        with patch('perception.aruco.cv2.projectPoints', return_value=(np.full((16, 1, 2), np.nan), None)):
            self.assertIsNone(self.tracker.estimate(self.raw))
        d = self.tracker.last_diagnostics
        self.assertEqual(d.rejection_code, 'nonfinite_reprojection')
        with tempfile.TemporaryDirectory() as directory:
            save_debug(directory, self.raw, annotate(self.raw, d), d, {}, '')
            report = json.loads((Path(directory)/'report.json').read_text())
            self.assertEqual(report['rms_error_px'], 'nan')
            saved = cv2.imread(str(Path(directory)/'raw.png'))
            np.testing.assert_array_equal(saved, self.raw)

    def test_preview_rotation_does_not_modify_input(self):
        self.raw[0, 0] = [10, 20, 30]
        with patch.object(config, 'PREVIEW_ROTATE_180', True):
            view = preview_frame(self.raw, debug_lines=['Diagnostics'])
        np.testing.assert_array_equal(view[1079, 1919], [10, 20, 30])
        np.testing.assert_array_equal(self.raw[0, 0], [10, 20, 30])
        self.assertGreater(view.shape[1], self.raw.shape[1])

    def test_center_audit_detects_physical_quarter_turn_without_changing_model(self):
        center = np.r_[self.centers[2], 0.]
        rotated = (marker_corners(2)-center) @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]) + center
        self.corners[2] = cv2.projectPoints(rotated, self.rvec, self.tvec, self.K, self.dist)[0].reshape(1, 4, 2)
        self.detect([0, 1, 2, 3])
        self.assertIsNone(self.tracker.estimate(self.raw))
        d = self.tracker.last_diagnostics
        audit, _ = audit_marker_geometry(d, self.K, self.dist)
        self.assertAlmostEqual(audit[2]['rotation_deg'], 90., delta=.05)
        self.assertAlmostEqual(audit[0]['rotation_deg'], 0., delta=.05)
        self.assertAlmostEqual(audit[2]['width_mm'], 30., delta=.05)
        self.assertEqual(d.rejection_code, 'reprojection_error')
        self.assertEqual(config.MAX_ARUCO_RMS_PX, 2.)

    def test_audit_preserves_skewed_fixture_with_upside_down_camera(self):
        self.corners = [cv2.projectPoints(marker_corners(i), np.array([.2, -.1, 3.0]),
                                         np.array([250., 160., 1200.]), self.K, self.dist)[0].reshape(1, 4, 2) for i in range(4)]
        self.detect([0, 1, 2, 3])
        self.tracker.estimate(self.raw)
        audit, _ = audit_marker_geometry(self.tracker.last_diagnostics, self.K, self.dist)
        for entry in audit.values():
            self.assertAlmostEqual(entry['rotation_deg'], 0., delta=.05)
            self.assertAlmostEqual(entry['width_mm'], 30., delta=.05)
            self.assertAlmostEqual(entry['height_mm'], 30., delta=.05)
        summary = '\n'.join(fixture_lines())
        self.assertIn('0->2: 310.000 mm', summary)
        self.assertIn('1->3: 305.000 mm', summary)
        self.assertEqual(config.MARKER_CENTERS_MM[2][1], 310.)
        self.assertEqual(config.MARKER_CENTERS_MM[3][1], 305.)


if __name__ == '__main__':
    unittest.main()
