import unittest
from unittest.mock import patch
import numpy as np
from types import SimpleNamespace
from calibration.planar import check_stationary, validate
from projection.planar import fit_mapping, transform, board_to_pixel, held_out_targets, warp_overlay


class PlanarTests(unittest.TestCase):
    def setUp(self):
        self.settings = patch.multiple('config', BOARD_BOUNDS_MM=(0.,0.,100.,100.),
                                      PROJECTOR_SIZE=(1920,1080), PLANAR_RANSAC_PX=3., PLANAR_MIN_COVERAGE=.2)
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.xy = np.array([[x,y] for y in [10,35,65,90] for x in [10,35,65,90]], float)
        self.H = np.array([[4.,.1,500.],[.1,2.,400.],[.0002,.0001,1.]])

    def test_robust_outlier_rejection_and_held_out_accuracy(self):
        uv = transform(self.H, self.xy)
        uv[5] += [100,-100]
        mapping = fit_mapping(self.xy, uv)
        self.assertFalse(mapping['inliers'][5])
        targets = held_out_targets(mapping)
        for p in targets:
            self.assertGreaterEqual(np.min(np.linalg.norm(self.xy-p, axis=1)), 5.)
            np.testing.assert_allclose(board_to_pixel(mapping,p), transform(self.H,[p])[0], atol=.001)

    def test_four_distributed_points_are_sufficient(self):
        xy = self.xy[[0,3,12,15]]
        mapping = fit_mapping(xy, transform(self.H, xy))
        self.assertEqual(sum(mapping['inliers']), 4)

    def test_cluster_and_collinear_and_insufficient_rejected(self):
        for xy in [self.xy*.1, np.array([[i,i] for i in range(10)], float), self.xy[:3]]:
            with self.assertRaises(ValueError):
                fit_mapping(xy, transform(self.H, xy))

    def test_no_extrapolation_and_overlay_orientation(self):
        mapping = fit_mapping(self.xy, transform(self.H,self.xy))
        with self.assertRaises(ValueError):
            board_to_pixel(mapping,[0,0])
        source = np.zeros((101,101,3), np.uint8)
        source[30:41,30:41] = [0,255,0]
        output = warp_overlay(mapping,source)
        u,v = np.rint(board_to_pixel(mapping,[35,35])).astype(int)
        self.assertGreater(output[v,u,1],200)
        self.assertEqual(output.shape,(1080,1920,3))

    def test_movement_invalidates_saved_mapping(self):
        record = {'status': 'UNVALIDATED'}
        tracker = SimpleNamespace(K=None, dist=None)
        with patch('calibration.planar.board_drift_px', return_value=10.), patch('calibration.planar.write_record') as save:
            with self.assertRaises(ValueError):
                check_stationary(record, object(), object(), tracker)
            self.assertTrue(record['status'].startswith('STALE'))
            save.assert_called_once()

    def test_physical_validation_does_not_refit_and_records_errors(self):
        mapping = fit_mapping(self.xy, transform(self.H,self.xy))
        original = np.array(mapping['H']).copy()
        record = dict(mapping=mapping, status='UNVALIDATED')
        targets = held_out_targets(mapping)
        rows = [(dict(board_xyz=[*(p+[1.,0.]),0.]), None, None) for p in targets]
        from unittest.mock import Mock
        with patch('calibration.planar.check_stationary'), patch('calibration.planar.measure_dot', side_effect=rows), patch('calibration.planar.show_preview'), patch('calibration.planar.write_record'):
            self.assertTrue(validate(record, None, Mock(), Mock(), Mock()))
        np.testing.assert_array_equal(original, mapping['H'])
        self.assertAlmostEqual(record['validation_history'][0]['rms_mm'], 1.)


if __name__ == '__main__':
    unittest.main()
