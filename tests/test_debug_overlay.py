import unittest
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
from perception.debug_overlay import placement_overlay


class DebugOverlayTests(unittest.TestCase):
    def test_pixel_coordinates_and_metric_labels_do_not_modify_inputs(self):
        frame = np.zeros((200,300,3),np.uint8)
        mask = np.zeros((200,300),np.uint8)
        mask[80:100,100:140] = 255
        original = mask.copy()
        region = SimpleNamespace(mask=mask,centroid_px=(119.5,89.5))
        with patch('perception.debug_overlay.silhouette',return_value=mask), \
             patch('perception.debug_overlay.cv2.putText',wraps=cv2.putText) as text:
            output = placement_overlay(frame,region,expected_xyz=[150,100,-19.2],
                                       fitted={'xy_mm':[151,98],'yaw_deg':12},base_z=-19.2)
        labels = [c.args[1] for c in text.call_args_list]
        self.assertTrue(any('TL=(100,80) BR=(139,99)' in s for s in labels))
        self.assertTrue(any('assumed Z=-19.20' in s for s in labels))
        self.assertTrue(any('dX=-1.00 dY=+2.00' in s for s in labels))
        np.testing.assert_array_equal(mask,original)
        self.assertFalse(frame.any())
        self.assertGreater(output.shape[0],frame.shape[0])
        self.assertTrue(output[80,100].any())  # box remains in original pixel coordinates

    def test_empty_region_is_labelled_without_inventing_coordinates(self):
        with patch('perception.debug_overlay.cv2.putText',wraps=cv2.putText) as text:
            placement_overlay(np.zeros((100,100,3),np.uint8),None)
        labels = [c.args[1] for c in text.call_args_list]
        self.assertIn('No accepted region',labels)
        self.assertFalse(any('board center' in s for s in labels))
