import unittest
from types import SimpleNamespace
import cv2
import numpy as np
from perception.geometry import pose_matrix
from perception.motion_compensation import aligned_colour_history, plane_projection
from perception.colour_change import detect_colour_change


class MotionCompensationTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[700.,0,320],[0,700.,240],[0,0,1]])
        self.old = SimpleNamespace(rvec=np.array([.2,.1,0.]), tvec=np.array([0.,0.,400.]))
        self.new = SimpleNamespace(rvec=np.array([.3,-.1,.1]), tvec=np.array([30.,-15.,430.]))
        self.T = np.eye(4)
        self.moved = pose_matrix(np.array([0.,0.,.2]), np.array([10.,5.,0.]))
        self.meshes = [np.array([[[-100.,-80.,0.],[100.,-80.,25.],[100.,80.,25.]]])]

    def rectangle(self, pose, T, z, x, colour=(20,20,230)):
        out = np.full((480,640,3),180,np.uint8)
        points = np.array([[x,-15,z],[x+40,-15,z],[x+40,0,z],[x,0,z]],float)
        camera = pose_matrix(pose.rvec,pose.tvec) @ T
        pts = points @ camera[:3,:3].T + camera[:3,3]
        pix = pts @ self.K.T
        cv2.fillConvexPoly(out,np.rint(pix[:,:2]/pix[:,2:]).astype(np.int32),colour)
        return out

    def test_existing_red_at_multiple_heights_is_not_new_after_camera_and_base_motion(self):
        for z in (0.,12.,25.):
            before = self.rectangle(self.old,self.T,z,-60)
            after = self.rectangle(self.new,self.moved,z,-60)
            history,valid = aligned_colour_history(before,'red',self.old,self.new,self.K,self.T,self.moved,self.meshes)
            region,_ = detect_colour_change(before,after,'red',previous_mask=history,search_mask=valid)
            self.assertIsNone(region, f'False new red at height {z}')

    def test_new_red_survives_motion_while_old_red_is_suppressed(self):
        before=self.rectangle(self.old,self.T,25,-60)
        after=self.rectangle(self.new,self.moved,25,-60)
        addition=self.rectangle(self.new,self.moved,25,20)
        selected=addition[:,:,2]>200
        after[selected]=addition[selected]
        history,valid=aligned_colour_history(before,'red',self.old,self.new,self.K,self.T,self.moved,self.meshes)
        region,_=detect_colour_change(before,after,'red',previous_mask=history,search_mask=valid)
        self.assertIsNotNone(region)
        self.assertGreater(np.count_nonzero(region.mask & selected.astype(np.uint8)),400)
        self.assertLess(np.count_nonzero(region.mask & (~selected).astype(np.uint8)),50)

    def test_newly_revealed_border_is_excluded(self):
        before=np.full((480,640,3),180,np.uint8)
        new=SimpleNamespace(rvec=self.old.rvec,tvec=self.old.tvec+[120.,0,0])
        history,valid=aligned_colour_history(before,'red',self.old,new,self.K,self.T,self.T,self.meshes)
        after=before.copy();after[150:250,10:70]=(20,20,230)
        self.assertIsNone(detect_colour_change(before,after,'red',previous_mask=history,search_mask=valid)[0])

    def test_missing_pose_cannot_be_compensated(self):
        before=np.full((480,640,3),180,np.uint8)
        behind=SimpleNamespace(rvec=self.old.rvec,tvec=np.array([0.,0.,-400.]))
        with self.assertRaises(ValueError):
            aligned_colour_history(before,'red',self.old,behind,self.K,self.T,self.T,self.meshes)
