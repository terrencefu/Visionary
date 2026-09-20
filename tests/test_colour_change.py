import unittest
from unittest.mock import patch
from types import SimpleNamespace
import cv2
import numpy as np
import config
from perception.colour_change import colour_mask, detect_colour_change, part_colour
from perception.stl_matcher import load_models, silhouette
from perception.anchor import anchor_component_name, register_cad, plate_surface_heights
from assembly.manual_guidance import ManualAssembly


class ColourMaskTests(unittest.TestCase):
    def setUp(self):
        self.before = np.full((400,600,3),180,np.uint8)

    def test_physical_colour_mapping_keeps_cad_names(self):
        self.assertEqual(part_colour('Plate 1x10 Silver'),'red')
        self.assertEqual(part_colour('389423 Bright Blue Technic Brick 1 x 6 with Holes'),'blue')
        self.assertIsNone(part_colour('2850 Medium Stone Grey Technic Engine Cylinder Head'))

    def test_blue_mask_excludes_grey_engine_and_shadow(self):
        after=self.before.copy()
        after[100:140,100:200]=(230,50,20)
        after[140:160,70:240]=70
        region,mask=detect_colour_change(self.before,after,'blue')
        self.assertIsNotNone(region)
        self.assertEqual(region.bbox,(100,100,100,40))
        self.assertFalse(np.any(mask[140:160]))

    def test_wrong_colour_and_grey_are_rejected(self):
        for bgr in [(20,30,230),(60,60,60)]:
            after=self.before.copy();after[100:140,100:200]=bgr
            self.assertIsNone(detect_colour_change(self.before,after,'blue')[0])

    def test_red_hue_wrap(self):
        hsv=np.array([[[0,220,200],[179,220,200],[110,220,200]]],np.uint8)
        mask=colour_mask(cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR),'red')
        np.testing.assert_array_equal(mask,[[255,255,0]])

    def test_old_red_plate_does_not_satisfy_next_red_step(self):
        self.before[80:110,80:210]=(20,20,220)
        after=self.before.copy()
        self.assertIsNone(detect_colour_change(self.before,after,'red')[0])
        after[220:250,80:210]=(20,20,220)
        region,_=detect_colour_change(self.before,after,'red')
        self.assertEqual(region.bbox,(80,220,130,30))

    def test_removal_is_not_addition(self):
        self.before[100:140,100:200]=(230,50,20)
        self.assertIsNone(detect_colour_change(self.before,np.full_like(self.before,180),'blue')[0])

    def test_workspace_and_marker_exclusions(self):
        after=self.before.copy();after[100:140,100:200]=(230,50,20)
        mask=np.zeros((400,600),np.uint8);mask[:80]=255
        self.assertIsNone(detect_colour_change(self.before,after,'blue',search_mask=mask)[0])
        quad=np.array([[95,95],[205,95],[205,145],[95,145]])
        self.assertIsNone(detect_colour_change(self.before,after,'blue',exclude_quads=[quad])[0])

    def test_large_scene_change_is_rejected(self):
        after=self.before.copy();after[:300]=(230,50,20)
        self.assertIsNone(detect_colour_change(self.before,after,'blue')[0])


class ColourPlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pose=SimpleNamespace(rvec=np.zeros(3),tvec=np.array([-150.,-100.,500.]))
        cls.K=np.array([[1100.,0,640],[0,1100.,360],[0,0,1]])
        name=anchor_component_name('Fusion_output')
        data,T=register_cad('Fusion_output',name,{'xy_mm':np.array([150.,100.]),'yaw_deg':0.},cls.pose)
        cls.machine=ManualAssembly(data,T,'Fusion_output',plate_surface_heights(load_models('Fusion_output')[name]))

    def test_colour_still_requires_correct_position_and_angle(self):
        machine=self.machine
        for index,shift,angle,expected_pass in [(1,0.,0.,True),(1,8.,0.,False),(1,0.,24.,False),(2,0.,0.,True)]:
            with self.subTest(index=index,shift=shift,angle=angle):
                machine.index=index
                lo,hi=machine.meshes[index].min(axis=(0,1)),machine.meshes[index].max(axis=(0,1))
                expected=machine.transform[:3,:3]@np.r_[(lo[:2]+hi[:2])/2,lo[2]]+machine.transform[:3,3]
                name=machine.names[index]
                mask=silhouette(machine.models[name],angle,expected[:2]+[shift,0],self.pose,self.K,(720,1280),base_z=expected[2])
                before=np.full((720,1280,3),180,np.uint8)
                after=before.copy();after[mask!=0]=(230,50,20) if part_colour(name)=='blue' else (20,20,230)
                # Only yaw search is seeded here; colour segmentation, metric fit,
                # target comparison and advancement gates are real.
                with patch('perception.colour_change.expected_pose_seed',return_value=(angle,[],np.zeros_like(after))), \
                     patch('assembly.manual_guidance.verify') as identity:
                    passed,message,_=machine.validate_frames(before,after,self.pose,self.K)
                identity.assert_not_called()
                self.assertEqual(passed,expected_pass,message)
                self.assertEqual(machine.checked_index,index if expected_pass else None)

    def test_blue_end_to_end_with_real_orientation_search(self):
        machine=self.machine;machine.index=1
        lo,hi=machine.meshes[1].min(axis=(0,1)),machine.meshes[1].max(axis=(0,1))
        expected=machine.transform[:3,:3]@np.r_[(lo[:2]+hi[:2])/2,lo[2]]+machine.transform[:3,3]
        mask=silhouette(machine.models[machine.names[1]],0,expected[:2],self.pose,self.K,(720,1280),base_z=expected[2])
        before=np.full((720,1280,3),180,np.uint8)
        after=before.copy();after[mask!=0]=(230,50,20)
        passed,message,_=machine.validate_frames(before,after,self.pose,self.K)
        self.assertTrue(passed,message)
        self.assertIn('detected colour=blue',message)

    def test_wrong_colour_clears_previous_pass_and_blocks_advance(self):
        machine=self.machine;machine.index=1;machine.checked_index=1
        before=np.full((720,1280,3),180,np.uint8);after=before.copy();after[300:330,600:700]=(20,20,230)
        passed,_,_=machine.validate_frames(before,after,self.pose,self.K)
        self.assertFalse(passed)
        self.assertIsNone(machine.checked_index)
        with self.assertRaises(ValueError):machine.advance()


if __name__=='__main__':unittest.main()
