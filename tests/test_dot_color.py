import unittest
from unittest.mock import patch
import cv2
import numpy as np
from perception.green_dot import detect_green_dot
from hardware.projector import Projector


class DotColorTests(unittest.TestCase):
    def test_selected_hue_and_ambiguity(self):
        background=np.zeros((200,300,3),np.uint8)
        for name,bgr in [('green',(0,255,0)),('magenta',(255,0,255))]:
            with patch('config.DOT_COLOR',name):
                frame=background.copy()
                cv2.circle(frame,(80,100),15,bgr,-1)
                self.assertIsNotNone(detect_green_dot(background,frame)[0])
                self.assertIsNone(detect_green_dot(frame,frame)[0])
                cv2.circle(frame,(220,100),15,bgr,-1)
                self.assertIsNone(detect_green_dot(background,frame)[0])
        with patch('config.DOT_COLOR','magenta'):
            for color in [(0,255,0),(0,0,255),(255,0,0),(255,255,255)]:
                frame=background.copy()
                cv2.circle(frame,(80,100),15,color,-1)
                self.assertIsNone(detect_green_dot(background,frame)[0])

    def test_projector_and_detector_share_magenta_setting(self):
        projector=Projector(); projector.width=300; projector.height=200
        with patch('config.DOT_COLOR','magenta'),patch('cv2.imshow') as show,patch('cv2.waitKey'):
            projector.dot(150,100)
            frame=show.call_args.args[1]
            np.testing.assert_array_equal(frame[100,150],[255,0,255])
            self.assertIsNotNone(detect_green_dot(np.zeros_like(frame),frame)[0])
