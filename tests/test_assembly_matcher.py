import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import cv2
import numpy as np
from perception.assembly_matcher import depth_image, check_placement
from perception.change_detector import region_from_mask
from tests.test_stl_matcher import box


class AssemblyMatcherTests(unittest.TestCase):
    def setUp(self):
        self.pose = SimpleNamespace(rvec=np.zeros(3),tvec=np.array([0.,0.,400.]))
        self.K = np.array([[900.,0,320],[0,900.,240],[0,0,1]])
        T = np.diag([1.,-1.,-1.,1.])
        self.assembly = SimpleNamespace(index=1,transform=T,
            meshes=[box((40,24,8)),box((32,16,8))+[0,0,8]],
            names=['support','389423 Bright Blue Technic Brick 1 x 6 with Holes'],
            operations=[{'part':'support'},{'part':'blue'}])

    def render(self,shift=0,yaw=0,occlusion=False):
        a = self.assembly
        target = a.meshes[1]@a.transform[:3,:3].T
        angle=np.radians(yaw);R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        target=target@R.T+[shift,0,0]
        z = depth_image(target,self.pose,self.K,(480,640))
        installed = depth_image(a.meshes[0]@a.transform[:3,:3].T,self.pose,self.K,(480,640))
        visible=np.isfinite(z)&(z<installed-.05)
        frame=np.full((480,640,3),180,np.uint8)
        frame[visible]=(230,50,20)
        region=region_from_mask(visible.astype(np.uint8)*255)
        return frame,region

    def test_correct_and_displaced_and_rotated(self):
        for shift,yaw,wanted in ((0,0,'PLACEMENT OK'),(6,0,'ADJUST PLACEMENT'),(0,18,'ADJUST PLACEMENT')):
            frame,region=self.render(shift,yaw)
            result=check_placement(self.assembly,frame,region,self.pose,self.K,'blue')
            self.assertTrue(result[1].startswith(wanted),result[1])
            self.assertEqual(result[0],wanted=='PLACEMENT OK')

    def test_shared_margin_for_brick_and_plate_still_rejects_one_stud_offset(self):
        for name,size,colour in (
                ('389423 Bright Blue Technic Brick 1 x 6 with Holes',(48,8,11.2),'blue'),
                ('Plate 1x10 Silver',(80,8,3.2),'red')):
            self.assembly.names[1]=name
            self.assembly.meshes[1]=box(size)+[0,0,8]
            for shift in (0,8):
                frame,region=self.render(shift=shift)
                if colour=='red':
                    frame[region.mask>0]=(20,20,230)
                result=check_placement(self.assembly,frame,region,self.pose,self.K,colour)
                self.assertEqual(result[3]['ambiguity_margin'],.005)
                self.assertEqual(result[0],shift==0,result[1])
            # The setting, rather than the component name, controls the gate.
            frame,region=self.render()
            if colour=='red':
                frame[region.mask>0]=(20,20,230)
            with patch('config.PLACEMENT_AMBIGUITY_MARGIN',1.):
                strict=check_placement(self.assembly,frame,region,self.pose,self.K,colour)
            self.assertFalse(strict[0])
            self.assertEqual(strict[3]['ambiguity_margin'],1.)

    def test_depth_buffer_uses_nearest_surface_not_triangle_order(self):
        near=np.array([[[-10.,-10,0],[10,-10,0],[0,10,0]]])
        far=near+[0,0,100]
        for mesh in (np.concatenate([near,far]),np.concatenate([far,near])):
            depth=depth_image(mesh,self.pose,self.K,(480,640))
            self.assertAlmostEqual(depth[240,320],400.)

    def test_half_hidden_target_is_compared_only_where_visible(self):
        # A tall installed surface hides the left half of the future brick.
        self.assembly.meshes[0]=box((18,30,30))+[-10,0,0]
        frame,region=self.render()
        self.assertIsNotNone(region)
        result=check_placement(self.assembly,frame,region,self.pose,self.K,'blue')
        self.assertTrue(result[0],result[1])
        self.assertLess(result[3]['expected_hypothesis']['visible_fraction'],.7)

    def test_missing_target_cannot_pass_on_old_colour(self):
        frame,region=self.render()
        frame[:]=(180,180,180)
        result=check_placement(self.assembly,frame,region,self.pose,self.K,'blue')
        self.assertFalse(result[0])
        self.assertTrue(result[1].startswith('INSUFFICIENT EVIDENCE'))

    def test_mostly_hidden_target_does_not_pass(self):
        frame,region=self.render()
        self.assembly.meshes[0]=box((80,80,35))
        result=check_placement(self.assembly,frame,region,self.pose,self.K,'blue')
        self.assertFalse(result[0])
        self.assertNotIn('measured_xy',result[3])

    def test_evidence_replays_frozen_assembly_geometry(self):
        from perception.placement_evidence import save_check, replay
        frame,region=self.render(shift=6)
        result=check_placement(self.assembly,frame,region,self.pose,self.K,'blue')
        self.assembly.last_check=result[3]
        self.assembly.models={self.assembly.names[1]:box((32,16,8))}
        with tempfile.TemporaryDirectory() as directory,patch('config.DATA_DIR',Path(directory)):
            saved=save_check(self.assembly,np.full_like(frame,180),frame,region,self.pose,self.K,result[1])
            with patch('perception.stl_matcher.load_models',side_effect=AssertionError('Replay must not load current CAD')):
                repeated=replay(saved)
            self.assertEqual(repeated,result[:2])
