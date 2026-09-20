import unittest
from types import SimpleNamespace
import cv2
import numpy as np
from perception.geometry import pose_matrix
from perception.moving_base import solve_marker, attach_anchor, tracked_outline


class MovingBaseTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[1000.,0,640],[0,1000.,360],[0,0,1]])
        self.dist = np.array([-.1,.03,0.,0.,0.])
        self.pose = SimpleNamespace(rvec=np.array([.35,0.,0.]),tvec=np.array([-80.,-60.,450.]))
        self.cb = pose_matrix(self.pose.rvec,self.pose.tvec)

    def test_raw_distorted_marker_pose(self):
        objects = np.array([[-15.,-15,0],[15,-15,0],[15,15,0],[-15,15,0]])
        corners = cv2.projectPoints(objects,self.pose.rvec,self.pose.tvec,self.K,self.dist)[0].reshape(4,2)
        result = solve_marker(corners,30,self.K,self.dist,3.)
        np.testing.assert_allclose(result.matrix,self.cb,atol=1e-5)
        self.assertLess(result.rms,1e-5)
        bad = corners.copy(); bad[0] += [15.,-12.]
        with self.assertRaises(ValueError):
            solve_marker(bad,30,self.K,self.dist,.1)

    def test_binding_follows_rigid_base_translation_and_rotation(self):
        mb = pose_matrix(np.array([0.,0.,.4]),np.array([50.,50.,0.]))
        estimate = {'xy_mm':np.array([100.,100.]),'yaw_deg':25.}
        bound = attach_anchor(self.cb@mb,self.cb,estimate,self.pose)
        saved = bound.copy()
        outline = np.array([[-4.,-40,3.33],[4,-40,3.33],[4,40,3.33],[-4,40,3.33]])
        _, initial = tracked_outline(self.cb@mb,bound,outline,self.K,self.dist)
        delta = pose_matrix(np.array([0.,0.,.8]),np.array([35.,-15.,0.]))
        pixels, current = tracked_outline(self.cb@delta@mb,bound,outline,self.K,self.dist)
        expected = self.cb@delta@np.linalg.inv(self.cb)@initial
        np.testing.assert_allclose(current,expected,atol=1e-8)
        expected_pixels = cv2.projectPoints(outline,cv2.Rodrigues(expected[:3,:3])[0],expected[:3,3],self.K,self.dist)[0].reshape(4,2)
        np.testing.assert_allclose(pixels,expected_pixels,atol=1e-8)
        np.testing.assert_array_equal(saved,bound)

    def test_initial_base_must_be_on_board(self):
        raised = np.eye(4); raised[2,3] = 20
        with self.assertRaises(ValueError):
            attach_anchor(self.cb@raised,self.cb,{'xy_mm':[100,100],'yaw_deg':0},self.pose)


if __name__ == '__main__':
    unittest.main()
