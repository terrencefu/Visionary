import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from calibration.planar_verify import stable_reference,landing_checks,acquire_reference


class VerifyTests(unittest.TestCase):
    def test_dropout_restarts_window_and_logs_missing_id(self):
        tracker=Mock()
        sequence=[[0,1,2,3]]*4+[[0,1,3]]+[[0,1,2,3]]*9
        def estimate(_):
            ids=sequence.pop(0)
            tracker.last_diagnostics=SimpleNamespace(detected_ids=ids,known_ids=ids,
                rms_error_px=1.2,rejection_reason=None)
            return SimpleNamespace(visible_ids=ids)
        tracker.estimate.side_effect=estimate
        report={}
        with patch('calibration.planar_verify.stable_reference',return_value=('reference',{})) as stable:
            self.assertEqual(acquire_reference(Mock(),Mock(),tracker,report)[0],'reference')
        self.assertEqual(report['reference_acquisition'][4]['missing_ids'],[2])
        self.assertEqual(report['reference_acquisition'][4]['consecutive_valid_frames'],0)
        self.assertEqual(report['reference_acquisition'][-1]['consecutive_valid_frames'],9)
        stable.assert_called_once()

    def test_timeout_keeps_diagnostics_and_projector_blank(self):
        tracker=Mock()
        tracker.estimate.return_value=None
        tracker.last_diagnostics=SimpleNamespace(detected_ids=[0,1],known_ids=[0,1],
                                                 rms_error_px=None,rejection_reason='insufficient markers')
        report={}; projector=Mock()
        with patch('calibration.planar_verify.time.monotonic',side_effect=[0.,0.,.5,31.]):
            with self.assertRaisesRegex(ValueError,'timed out'):
                acquire_reference(Mock(),projector,tracker,report)
        self.assertEqual(report['reference_acquisition'][0]['missing_ids'],[2,3])
        projector.black.assert_called_once()
        projector.dot.assert_not_called()

    def test_stable_window_selects_observed_pose_without_saved_reference_acceptance(self):
        obj=np.array([[0.,0.,0.],[30.,0.,0.],[30.,30.,0.],[0.,30.,0.]])
        poses=[SimpleNamespace(rvec=np.zeros(3),tvec=np.array([1.8+i*.01,0.,1000.]),
                               object_points=obj,visible_ids=[0,1,2,3]) for i in range(9)]
        ref,info=stable_reference(poses,np.diag([1000.,1000.,1.]),np.zeros(5))
        self.assertTrue(any(ref is p for p in poses))
        self.assertLess(info['max_deviation_px'],.1)
        poses[-1].tvec[0]=20.
        with self.assertRaises(ValueError):
            stable_reference(poses,np.diag([1000.,1000.,1.]),np.zeros(5))

    def test_checks_do_not_mutate_mapping_and_blank_on_failure(self):
        mapping={'H':np.eye(3).tolist()}
        before=copy.deepcopy(mapping)
        projector=Mock()
        with patch('calibration.planar_verify.held_out_targets',return_value=np.array([[1.,2.]]*5)), patch('calibration.planar_verify.board_to_pixel',return_value=[100,100]), patch('calibration.planar_verify.measure_dot',return_value=({'board_xyz':[1.,2.,0.]},None,None)):
            self.assertTrue(landing_checks(mapping,None,Mock(),projector,Mock())['passed'])
        self.assertEqual(mapping,before)
        projector.black.assert_called()
        projector.reset_mock()
        with patch('calibration.planar_verify.held_out_targets',return_value=np.array([[1.,2.]]*5)), patch('calibration.planar_verify.board_to_pixel',return_value=[100,100]), patch('calibration.planar_verify.measure_dot',side_effect=ValueError('movement')) as measure:
            self.assertFalse(landing_checks(mapping,None,Mock(),projector,Mock())['passed'])
            self.assertEqual(measure.call_count,1)
        projector.black.assert_called_once()
