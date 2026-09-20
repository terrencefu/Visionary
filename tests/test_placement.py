import copy
import unittest
from unittest.mock import patch
import numpy as np
from projection.placement import footprint_corners, placement_canvas


class PlacementTests(unittest.TestCase):
    def test_requested_brick_and_rotation(self):
        np.testing.assert_allclose(footprint_corners(150,100,32,16,0),
                                   [[134,92],[166,92],[166,108],[134,108]])
        rotated = footprint_corners(150,100,32,16,90)
        np.testing.assert_allclose(rotated,[[158,84],[158,116],[142,116],[142,84]])

    def test_invalid_dimensions(self):
        for w,h,a in [(0,16,0),(32,-1,0),(32,16,float('nan'))]:
            with self.assertRaises(ValueError):
                footprint_corners(150,100,w,h,a)

    def test_canvas_mapping_and_workspace_rejection(self):
        mapping = dict(H=[[2.,0.,400.],[0.,2.,200.],[0.,0.,1.]],
                       hull=[[30.,30.],[273.,30.],[273.,173.],[30.,173.]])
        saved = copy.deepcopy(mapping)
        with patch.multiple('config',BOARD_BOUNDS_MM=(30,30,273,173),PROJECTOR_SIZE=(1920,1080)):
            frame,xy,uv = placement_canvas(mapping,'LEGO-2x4',150,100,32,16,0)
            np.testing.assert_allclose(uv,xy*2+[400,200])
            self.assertEqual(frame.shape,(1080,1920,3))
            self.assertGreater(frame[400,700,1],200)
            self.assertEqual(mapping,saved)
            with self.assertRaises(ValueError):
                placement_canvas(mapping,'brick',35,100,32,16,45)
