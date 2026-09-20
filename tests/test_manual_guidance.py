import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from unittest.mock import patch

from perception.change_detector import detect_change
from perception.stl_matcher import silhouette
from assembly.manual_guidance import ManualAssembly, surface_height
from perception.anchor import register_cad, plate_surface_heights, anchor_component_name
from perception.stl_matcher import load_models


class ManualGuidanceTests(unittest.TestCase):
    def test_surface_intersection_and_empty_space(self):
        tri = np.array([[[0,0,2],[10,0,2],[0,10,2]],
                        [[0,0,5],[10,0,5],[0,10,5]]],float)
        self.assertEqual(surface_height(tri,[2,2],0),5)
        self.assertEqual(surface_height(tri,[20,20],0),0)

    def test_four_cad_steps_and_raised_surfaces(self):
        folder = Path(__file__).resolve().parents[1]/'Fusion_output'
        if not folder.exists():
            self.skipTest('Local Fusion export unavailable')
        pose = SimpleNamespace(rvec=np.zeros(3),tvec=np.array([0.,0.,500.]))
        name = anchor_component_name(folder)
        data,T = register_cad(folder,name,{'xy_mm':np.array([150.,100.]),'yaw_deg':0.},pose)
        original = T.copy()
        machine = ManualAssembly(data,T,folder,plate_surface_heights(load_models(folder)[name]))
        placements = sum(len(s['operations']) for s in data['assembly_plan']['steps'])
        self.assertEqual(len(machine.operations),placements)
        for step in range(placements):
            self.assertEqual(machine.index,step)
            self.assertFalse(machine.complete)
            scene = machine.scene()
            self.assertTrue(scene[0])
            points = np.array([edge[0] for edge in scene[0]])
            self.assertTrue(np.all(points[:,2]<=1e-6))
            if step > 0:
                with self.assertRaises(ValueError):
                    machine.advance()
                machine.checked_index = step
            machine.advance()
        self.assertTrue(machine.complete)
        self.assertEqual(machine.scene(),([],[]))
        machine.advance()
        self.assertEqual(machine.index,placements)
        np.testing.assert_array_equal(T,original)

    def test_guidance_reuses_surface_geometry_and_tracks_new_transform(self):
        folder = Path(__file__).resolve().parents[1]/'Fusion_output'
        pose = SimpleNamespace(rvec=np.zeros(3),tvec=np.array([0.,0.,500.]))
        name = anchor_component_name(folder)
        data,T = register_cad(folder,name,{'xy_mm':np.array([150.,100.]),'yaw_deg':0.},pose)
        machine = ManualAssembly(data,T,folder,plate_surface_heights(load_models(folder)[name]))
        machine.index = 1
        with patch.object(machine,'_build_scene',wraps=machine._build_scene) as build:
            first = machine.scene()
            machine.transform[:2,3] += [5.,3.]
            second = machine.scene()
            self.assertEqual(build.call_count,1)
        for a,b in zip(first[0],second[0]):
            np.testing.assert_allclose(b[0]-a[0],[5.,3.,0.],atol=1e-8)
        fresh = machine._build_scene()
        np.testing.assert_allclose([e[0] for e in second[0]], [e[0] for e in fresh[0]],atol=1e-8)

    def test_multi_operation_step_becomes_sequential_placements(self):
        """A CAD step holding two placements must yield two verifiable steps.

        Perception judges one new part per baseline, so a grouped step cannot be
        presented to the user as a single placement.
        """
        folder = Path(__file__).resolve().parents[1]/'Fusion_output'
        if not folder.exists():
            self.skipTest('Local Fusion export unavailable')
        data = json.loads((folder/'assembly.json').read_text())
        steps = data['assembly_plan']['steps']
        placements = sum(len(s['operations']) for s in steps)
        pose = SimpleNamespace(rvec=np.zeros(3),tvec=np.array([0.,0.,500.]))
        name = anchor_component_name(folder)
        data,T = register_cad(folder,name,{'xy_mm':np.array([150.,100.]),'yaw_deg':0.},pose)
        machine = ManualAssembly(data,T,folder,plate_surface_heights(load_models(folder)[name]))
        self.assertEqual(len(machine.operations),placements)
        self.assertGreater(placements,len(steps),
                           'This export no longer exercises multi-operation grouping')
        # Every dependency must already be placed by the time its operation runs.
        placed = set()
        for op in machine.operations:
            self.assertTrue(set(op.get('dependencies',[])) <= placed,
                            f'{op["id"]} runs before its dependencies')
            placed.add(op['id'])

    def test_measured_position_and_angle_gate_at_cad_height(self):
        folder = Path(__file__).resolve().parents[1]/'Fusion_output'
        if not folder.exists():
            self.skipTest('Local Fusion export unavailable')
        pose = SimpleNamespace(rvec=np.zeros(3),tvec=np.array([-150.,-100.,500.]))
        K = np.array([[1100.,0,640],[0,1100.,360],[0,0,1]])
        name = anchor_component_name(folder)
        data,T = register_cad(folder,name,{'xy_mm':np.array([150.,100.]),'yaw_deg':0.},pose)
        machine = ManualAssembly(data,T,folder,plate_surface_heights(load_models(folder)[name]))
        last = len(machine.operations)-1
        for index,shift,angle,accepted in ((1,0.,0.,True),(1,7.,0.,False),(1,0.,18.,False),(last,0.,0.,True)):
            machine.index = index
            lo,hi = machine.meshes[index].min(axis=(0,1)),machine.meshes[index].max(axis=(0,1))
            root = np.array([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]])
            expected = T[:3,:3]@root+T[:3,3]
            component = machine.names[index]
            mask = silhouette(machine.models[component],angle,expected[:2]+[shift,0],pose,K,(720,1280),base_z=expected[2])
            before = np.full((720,1280,3),200,np.uint8)
            after = before.copy(); after[mask!=0]=30
            region = detect_change(before,after)
            # Isolate metric position fitting from identity classification.
            with patch('assembly.manual_guidance.verify',return_value=('CORRECT SHAPE',[(component,.95,angle)],None)):
                passed,message,_ = machine.validate(region,pose,K)
            self.assertEqual(passed,accepted,message)
            self.assertEqual(machine.checked_index,index if accepted else None)
