import argparse
import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import config
from tracking.assembly_servo import AssemblyServo, add_servo_arguments, servo_options


class ServoIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.link = Mock(positions=(1500,1000))
        self.now = 0.
        self.servo = AssemblyServo(self.link,np.eye(3),np.zeros(5),clock=lambda:self.now,acquire_frames=1)
        self.blank = Mock()
        self.pose = object()

    def test_connect_does_not_move_and_arming_requires_pose_and_phase(self):
        self.servo.toggle(False,self.pose,self.blank)
        self.servo.toggle(True,None,self.blank)
        self.link.move.assert_not_called()
        self.servo.toggle(True,self.pose,self.blank)
        self.assertEqual(self.link.move.call_count,2)
        self.blank.assert_called_once()

    def test_placement_phase_freezes_commands_even_with_large_error(self):
        self.servo.toggle(True,self.pose,self.blank)
        self.link.reset_mock()
        with patch('tracking.assembly_servo.board_center_pixel',return_value=np.array([1600.,800.])) as project:
            self.assertTrue(self.servo.update(self.pose,False,self.blank))
            project.assert_not_called()
            self.link.move.assert_not_called()

    def test_movement_and_loss_block_baseline_until_fresh_centered_window(self):
        self.servo.toggle(True,self.pose,self.blank)
        with patch('tracking.assembly_servo.board_center_pixel',return_value=np.array([1600.,800.])):
            self.assertFalse(self.servo.update(self.pose,True,self.blank))
        count=self.link.move.call_count
        self.now=1.
        self.assertFalse(self.servo.update(None,True,self.blank))
        self.assertEqual(self.link.move.call_count,count)
        with patch('tracking.assembly_servo.board_center_pixel',return_value=np.array(config.CAMERA_SIZE)/2):
            self.assertFalse(self.servo.update(self.pose,True,self.blank))
            self.now+=config.SETTLE_SECONDS+.01
            self.assertTrue(self.servo.update(self.pose,True,self.blank))

    def test_pause_does_not_allow_immediate_capture_after_movement(self):
        self.servo.toggle(True,self.pose,self.blank)
        self.servo.toggle(True,self.pose,self.blank)
        self.assertFalse(self.servo.update(self.pose,True,self.blank))
        self.now=config.SETTLE_SECONDS+.01
        self.assertTrue(self.servo.update(self.pose,True,self.blank))

    def test_raw_out_of_view_target_holds(self):
        self.servo.toggle(True,self.pose,self.blank)
        self.link.reset_mock()
        with patch('tracking.assembly_servo.board_center_pixel',return_value=np.array([-1.,500.])):
            self.assertFalse(self.servo.update(self.pose,True,self.blank))
        self.link.move.assert_not_called()

    def test_cli_default_is_no_serial_and_live_requires_signs(self):
        parser=argparse.ArgumentParser()
        add_servo_arguments(parser)
        self.assertIsNone(servo_options(parser,parser.parse_args([])))
        with self.assertRaises(SystemExit):servo_options(parser,parser.parse_args(['--servo-port','COM5']))
        options=servo_options(parser,parser.parse_args(['--servo-port','COM5','--servo-pan-only','--servo-pan-sign','-1']))
        self.assertEqual(options['pan_sign'],-1)
        self.assertEqual(options['max_step'],2)

    def test_real_cli_routes_options_into_existing_mvp(self):
        from assembly.demo_perception import main
        with patch('sys.argv',['main.py','--change-only','--servo-port','COM5','--servo-pan-sign','1','--servo-pan-only']), \
             patch('perception.demo_change.run') as run:
            main()
        self.assertEqual(run.call_args.kwargs['servo_options']['port'],'COM5')

    def test_camera_loop_blocks_space_and_detection_while_correcting(self):
        from perception.demo_change import run
        camera=MagicMock()
        camera.__enter__.return_value=camera
        camera.read.return_value=np.zeros((1080,1920,3),np.uint8)
        tracker=Mock()
        tracker.estimate.return_value=self.pose
        assist=Mock()
        assist.update.return_value=False
        assist.status='CORRECTING'
        link=MagicMock()
        with patch('perception.demo_change.load_camera',return_value=(np.eye(3),np.zeros(5))), \
             patch('perception.demo_change.BoardTracker',return_value=tracker), \
             patch('perception.demo_change.Camera',return_value=camera), \
             patch('tracking.assembly_servo.AssemblyServo',return_value=assist), \
             patch('hardware.servo.ServoLink',return_value=link), \
             patch('perception.demo_change.show_preview'), \
             patch('perception.demo_change.cv2.waitKey',side_effect=[32,ord('q')]), \
             patch('perception.demo_change.workspace_polygon') as polygon, \
             patch('perception.demo_change.detect_change') as detect, \
             patch('perception.demo_change.cv2.destroyAllWindows'):
            run(servo_options=dict(port='COM5'))
        polygon.assert_not_called()  # Space was ignored; no stale baseline captured.
        detect.assert_not_called()
        self.assertEqual(assist.update.call_count,2)


if __name__=='__main__':unittest.main()
