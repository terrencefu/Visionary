import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from projection.guidance import footprint_scene,render_scene
from projection.placement import footprint_corners


class GuidanceTests(unittest.TestCase):
    def test_live_pose_changes_projection_without_planar_file(self):
        scene=footprint_scene(footprint_corners(150,100,32,16,0),'LEGO-2x4',(150,100))
        calibration=(np.array([[1000.,0.,960.],[0.,1000.,540.],[0.,0.,1.]]),np.zeros(5),np.eye(4))
        pose=SimpleNamespace(rvec=np.zeros(3),tvec=np.array([-150.,-100.,1000.]))
        with patch('pathlib.Path.read_text',side_effect=AssertionError('Must not load planar data')):
            first=render_scene(scene,pose,calibration)
            pose.tvec[0]+=50.
            second=render_scene(scene,pose,calibration)
        self.assertFalse(np.array_equal(first,second))
        self.assertGreater(first[540,960,1],200)
        self.assertGreater(second[540,1010,1],200)
        pose.tvec[2]=-1000.
        with self.assertRaises(ValueError):render_scene(scene,pose,calibration)

    def test_workspace_rejection(self):
        with self.assertRaises(ValueError):
            footprint_scene(footprint_corners(0,0,32,16,0),'brick',(0,0))
