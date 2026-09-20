import unittest
from types import SimpleNamespace
import numpy as np
from assembly.base_registration import MovingRegistration,planar_marker_board
from perception.geometry import pose_matrix
from perception.moving_base import MarkerPose


class MovingRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.pose = SimpleNamespace(rvec=np.array([.3,0.,0.]),tvec=np.array([0.,0.,450.]))
        self.cb = pose_matrix(self.pose.rvec,self.pose.tvec)
        self.K = np.array([[1000.,0,640],[0,1000.,360],[0,0,1]])

    def test_registration_moves_with_marker_without_refitting_anchor(self):
        controller = MovingRegistration(5,30,self.K,np.zeros(5))
        mb = pose_matrix(np.array([0.,0.,.4]),np.array([50.,60.,0.]))
        cad = pose_matrix(np.array([np.pi,0.,0.]),np.array([110.,100.,0.]))
        controller.current = MarkerPose(self.cb@mb,np.zeros((4,2)),0.)
        controller.bind(self.pose,cad)
        binding = controller.marker_from_cad.copy()
        delta = pose_matrix(np.array([0.,0.,.7]),np.array([10.,-20.,0.]))
        controller.current = MarkerPose(self.cb@delta@mb,np.zeros((4,2)),0.)
        np.testing.assert_allclose(controller.transform(self.pose),delta@cad,atol=1e-8)
        np.testing.assert_array_equal(controller.marker_from_cad,binding)

    def test_tilted_or_lifted_base_cannot_arm(self):
        for mb in (pose_matrix(np.zeros(3),np.array([0.,0.,20.])),
                   pose_matrix(np.array([.5,0.,0.]),np.zeros(3))):
            with self.assertRaises(ValueError):
                planar_marker_board(self.cb@mb,self.pose)

    def test_movement_invalidates_until_explicit_rearm(self):
        controller = MovingRegistration(5,30,self.K,np.zeros(5))
        import cv2
        corners = cv2.projectPoints(np.array([[-15.,-15,0],[15,-15,0],[15,15,0],[-15,15,0]]),
                                   self.pose.rvec,self.pose.tvec,self.K,np.zeros(5))[0].reshape(1,4,2).astype(np.float32)
        controller.detector = SimpleNamespace(detectMarkers=lambda raw:([corners],np.array([[5]]),[]))
        controller.observe(None)
        controller.arm()
        corners += np.array([5.,0.],np.float32)
        controller.observe(None)
        self.assertTrue(controller.invalidated)
        corners -= np.array([5.,0.],np.float32)
        controller.observe(None)
        self.assertTrue(controller.invalidated)
        controller.begin_move()
        controller.arm()
        self.assertFalse(controller.invalidated)

    def test_marker_loss_preserves_binding_but_prevents_transform(self):
        controller = MovingRegistration(5,30,self.K,np.zeros(5))
        controller.current = MarkerPose(self.cb,np.zeros((4,2)),0.)
        controller.bind(self.pose,np.eye(4))
        binding = controller.marker_from_cad.copy()
        controller.detector = SimpleNamespace(detectMarkers=lambda raw:([],None,[]))
        controller.observe(None)
        with self.assertRaises(ValueError):
            controller.transform(self.pose)
        np.testing.assert_array_equal(controller.marker_from_cad,binding)
