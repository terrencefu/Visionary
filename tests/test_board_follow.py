import unittest
import subprocess
import sys
from unittest.mock import Mock, patch, MagicMock
from types import SimpleNamespace
import cv2
import numpy as np
import config
from hardware.servo import ServoLink
from perception.aruco import BoardTracker, marker_corners
from tracking.controller import FollowController, board_center_pixel
from tracking.follow_board import selected_centers, parse_args


class ControlTests(unittest.TestCase):
    def test_deadband_and_loss(self):
        c = FollowController(acquire_frames=1)
        self.assertEqual(c.update([971,550], [960,540], 0), [])
        self.assertEqual(c.status, 'CENTERED')
        self.assertTrue(c.update([1200,800], [960,540], 1))
        previous = c.pan, c.tilt
        self.assertEqual(c.update(None, [960,540], 2), [])
        self.assertEqual((c.pan,c.tilt), previous)
        self.assertEqual(c.status, 'BOARD LOST - HOLD')

    def test_directions_rate_limits_and_pan_only(self):
        c = FollowController(pan_sign=-1, tilt_sign=1, acquire_frames=1, max_step=5)
        self.assertEqual(c.update([1960,-540],[960,540],0), [('P',1495),('T',995)])
        self.assertEqual(c.update([1960,-540],[960,540],.01), [])
        c.pan_only=True
        self.assertEqual(c.update([1960,-540],[960,540],1), [('P',1490)])

    def test_limits_do_not_accumulate_and_reverse_immediately(self):
        c = FollowController(pan=2700,tilt=700,acquire_frames=1)
        self.assertEqual(c.update([2000,0],[960,540],0), [])
        self.assertEqual(c.status,'LIMIT REACHED')
        commands=c.update([0,1080],[960,540],1)
        self.assertLess(dict(commands)['P'],2700)
        self.assertGreater(dict(commands)['T'],700)

    def test_reacquire_and_nan_never_command(self):
        c=FollowController()
        for t in range(2):
            self.assertEqual(c.update([1800,540],[960,540],t),[])
        self.assertTrue(c.update([1800,540],[960,540],2))
        self.assertEqual(c.update([np.nan,540],[960,540],3),[])
        self.assertEqual(c.update([1800,540],[960,540],4),[])
        c.reset()
        self.assertEqual(c.good_frames,0)

    def test_synthetic_closed_loop_recovers_rig_disturbance(self):
        for sign in (-1,1):
            c=FollowController(pan_sign=sign,tilt_sign=-sign,acquire_frames=1)
            error=np.array([350.,-250.])
            for i in range(120):
                old=np.array([c.pan,c.tilt])
                c.update(np.array([960,540])+error,[960,540],i*.2)
                delta=np.array([c.pan,c.tilt])-old
                # Synthetic mechanism: 3 pixels of feedback per microsecond.
                error -= delta*np.array([sign,-sign])*3
            self.assertTrue(np.all(np.abs(error)<=c.deadband),error)
            self.assertEqual(c.status,'CENTERED')

    def test_invalid_settings(self):
        for kwargs in [dict(gain=float('nan')),dict(interval=0),dict(pan_sign=0),
                       dict(pan_limits=(300,2700)),dict(tilt_limits=(1100,1500))]:
            with self.assertRaises(ValueError):FollowController(**kwargs)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.K=np.array([[1000.,0,960],[0,1000.,540],[0,0,1.]])
        self.dist=np.zeros(5)
        self.rvec=np.array([.05,.1,3.1])
        self.tvec=np.array([160.,100.,900.])

    def test_center_uses_board_geometry_not_visible_marker_average(self):
        pose=SimpleNamespace(rvec=self.rvec,tvec=self.tvec,R=cv2.Rodrigues(self.rvec)[0])
        actual=board_center_pixel(pose,self.K,self.dist,config.BOARD_BOUNDS_MM)
        x0,y0,x1,y1=config.BOARD_BOUNDS_MM
        expected=cv2.projectPoints(np.array([[(x0+x1)/2,(y0+y1)/2,0.]]),self.rvec,self.tvec,self.K,self.dist)[0].reshape(2)
        np.testing.assert_allclose(actual,expected)
        pose.tvec=np.array([0.,0.,-900.])
        self.assertIsNone(board_center_pixel(pose,self.K,self.dist,config.BOARD_BOUNDS_MM))

    def test_select_second_board_without_changing_global_fixture(self):
        before=dict(config.MARKER_CENTERS_MM)
        centers=selected_centers([4,5,6,7])
        tracker=BoardTracker(self.K,self.dist,marker_centers=centers)
        ids=[0,1,2,3,4,5,6,7]
        corners=[cv2.projectPoints(marker_corners(i if i<4 else i-4),self.rvec,
                                  self.tvec+(np.array([300.,0,0]) if i<4 else 0),
                                  self.K,self.dist)[0].reshape(1,4,2) for i in ids]
        tracker.detector=Mock()
        tracker.detector.detectMarkers.return_value=corners,np.array(ids).reshape(-1,1),[]
        pose=tracker.estimate(np.zeros((1080,1920,3),np.uint8))
        self.assertEqual(pose.visible_ids,[4,5,6,7])
        np.testing.assert_allclose(pose.tvec.reshape(3),self.tvec,atol=1e-5)
        self.assertEqual(config.MARKER_CENTERS_MM,before)
        # Duplicate selected IDs are ambiguous and must not move the rig.
        tracker.detector.detectMarkers.return_value=corners[4:]+[corners[4]],np.array([4,5,6,7,4]).reshape(-1,1),[]
        self.assertIsNone(tracker.estimate(np.zeros((1080,1920,3),np.uint8)))

    def test_missing_marker_keeps_same_target_center(self):
        centers=selected_centers([4,5,6,7])
        tracker=BoardTracker(self.K,self.dist,marker_centers=centers)
        ids=[4,5,7]
        corners=[cv2.projectPoints(marker_corners(i,centers),self.rvec,self.tvec,self.K,self.dist)[0].reshape(1,4,2) for i in ids]
        tracker.detector=Mock()
        tracker.detector.detectMarkers.return_value=corners,np.array(ids).reshape(-1,1),[]
        pose=tracker.estimate(np.zeros((1080,1920,3),np.uint8))
        np.testing.assert_allclose(pose.tvec.reshape(3),self.tvec,atol=1e-5)


class FakeSerial:
    def __init__(self,*args,**kwargs):
        self.lines=[];self.writes=[];self.closed=False
    def reset_input_buffer(self):self.lines=[]
    def write(self,data):
        self.writes.append(data)
        if data==b'STATUS\n':
            self.lines=[b'Ready. Outputs OFF.\n',b'Pan: commanded 1500 us; OFF\n',b'Tilt: commanded 1000 us; OFF\n']
        elif data==b'OFF\n':self.lines=[b'Signals OFF. Support the assembly.\n']
        else:
            axis,pulse=data.decode().split()
            self.lines=[f'{"Pan" if axis=="P" else "Tilt"} -> {pulse} us.\n'.encode()]
        return len(data)
    def readline(self):return self.lines.pop(0) if self.lines else b''
    def close(self):self.closed=True


class SerialTests(unittest.TestCase):
    def test_protocol_handshake_ack_and_hold_on_close(self):
        with ServoLink('fake',serial_factory=FakeSerial) as link:
            self.assertEqual(link.positions,(1500,1000))
            self.assertEqual(link.serial.writes,[b'STATUS\n'])
            link.move('P',1510);link.move('T',990)
            self.assertEqual(link.serial.writes[-2:],[b'P 1510\n',b'T 990\n'])
            with self.assertRaises(ValueError):link.move('T',1600)
        self.assertTrue(link.serial.closed)
        self.assertNotIn(b'OFF\n',link.serial.writes)

    def test_off_and_rejection(self):
        with ServoLink('fake',serial_factory=FakeSerial) as link:
            link.off()
            link.serial.write=lambda data: link.serial.lines.append(b'Rejected. Allowed range: 700-1500 us.\n') or len(data)
            with self.assertRaises(RuntimeError):link.move('T',1000)

    def test_timeout(self):
        link=ServoLink('fake',timeout=.001,serial_factory=FakeSerial)
        link.serial.write=lambda data:len(data)
        with self.assertRaises(TimeoutError):link.move('P',1500)
        link.serial.close()

    def test_launcher_forwards_tool_help(self):
        result = subprocess.run([sys.executable, 'main.py', 'follow-board', '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('--pan-sign', result.stdout)
        self.assertIn('--marker-ids', result.stdout)

    def test_preview_default_and_live_requires_explicit_signs(self):
        self.assertFalse(parse_args([]).live)
        with self.assertRaises(SystemExit):parse_args(['--live','--port','fake'])
        self.assertTrue(parse_args(['--live','--port','fake','--pan-sign','1','--pan-only']).live)


class LiveLoopTests(unittest.TestCase):
    def test_preview_never_opens_serial(self):
        self.run_loop(False)

    def test_live_waits_for_start_and_stops_corrections_on_loss(self):
        self.run_loop(True)

    def run_loop(self, live):
        from tracking.follow_board import main
        args=parse_args(['--live','--port','fake','--pan-sign','1','--tilt-sign','1'] if live else [])
        camera=MagicMock()
        camera.__enter__.return_value=camera
        camera.read.return_value=np.zeros((1080,1920,3),np.uint8)
        tracker=Mock()
        tracker.estimate.side_effect=[object()]*5+[None]*2
        tracker.last_diagnostics.rejection_reason='no markers'
        link=MagicMock()
        link.__enter__.return_value=link
        link.positions=(1500,1000)
        # First frame paused, second arms, three acquired frames, then loss.
        keys=iter([-1,32,-1,-1,-1,-1,ord('q')])
        commands_at_keys=[]
        def key(_):
            commands_at_keys.append(link.move.call_count)
            return next(keys)
        with patch('tracking.follow_board.parse_args',return_value=args), \
             patch('tracking.follow_board.load_camera',return_value=(np.eye(3),np.zeros(5))), \
             patch('tracking.follow_board.Camera',return_value=camera), \
             patch('tracking.follow_board.BoardTracker',return_value=tracker), \
             patch('tracking.follow_board.ServoLink',return_value=link) as factory, \
             patch('tracking.follow_board.board_center_pixel',return_value=np.array([1400.,800.])), \
             patch('tracking.follow_board.show_preview'), \
             patch('tracking.follow_board.cv2.waitKey',side_effect=key):
            main()
        if live:
            self.assertEqual(commands_at_keys[:2],[0,0])
            self.assertEqual(commands_at_keys[-3:],[4,4,4])
            self.assertEqual(link.move.call_args_list[-2].args,('P',1512))
            self.assertEqual(link.move.call_args_list[-1].args,('T',1009))
        else:
            factory.assert_not_called()
            link.move.assert_not_called()


if __name__=='__main__':unittest.main()
