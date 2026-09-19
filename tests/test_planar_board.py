import copy
import unittest
from unittest.mock import patch, Mock
import numpy as np
from calibration.planar_board import alignment_canvas, RecoveryGuard, validate_board


class BoardAlignmentTests(unittest.TestCase):
    def test_configured_vertices_and_crosshairs_use_saved_H(self):
        mapping = {'H': [[2.,0.,100.],[0.,2.,100.],[0.,0.,1.]]}
        before = copy.deepcopy(mapping)
        with patch.multiple('config', PROJECTOR_SIZE=(1920,1080),
                            MARKER_CENTERS_MM={0:(0,0),1:(303,0),2:(0,203),3:(303,204)},
                            BOARD_BOUNDS_MM=(30,30,273,173)):
            canvas = alignment_canvas(mapping)
        self.assertEqual(canvas.shape,(1080,1920,3))
        for u,v in [(100,100),(706,100),(100,506),(706,508)]:
            self.assertEqual(canvas[v,u,1],0)
            self.assertGreater(canvas[v,u,2],200)
        self.assertGreater(canvas[160,160,1],200)
        self.assertEqual(mapping,before)

    def test_tracking_recovers_and_movement_requires_three_poses(self):
        guard = RecoveryGuard()
        self.assertFalse(guard.update(None,0.))
        self.assertFalse(guard.update(None,1.))
        self.assertTrue(guard.update(0.,1.5))
        self.assertFalse(guard.update(100.,2.))
        self.assertFalse(guard.update(100.,2.1))
        with self.assertRaisesRegex(ValueError,'Movement confirmed'):
            guard.update(100.,2.2)

    def test_tracking_timeout(self):
        guard = RecoveryGuard()
        guard.update(None,0.)
        with self.assertRaisesRegex(ValueError,'did not recover'):
            guard.update(None,2.)

    def test_visual_loop_never_writes_or_refits(self):
        record = {'mapping': {'H': np.eye(3).tolist()}, 'status':'UNVALIDATED'}
        before = copy.deepcopy(record)
        with patch('calibration.planar.write_record') as write, patch('cv2.findHomography') as fit, patch('calibration.planar_board.board_drift_px',return_value=0.), patch('calibration.planar_board.show_preview'), patch('cv2.imshow'), patch('cv2.waitKey',return_value=27):
            validate_board(record,Mock(),Mock(),Mock(),Mock())
        write.assert_not_called()
        fit.assert_not_called()
        self.assertEqual(record,before)
