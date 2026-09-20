import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from perception.anchor import (estimate_anchor, register_cad, placement_scene,
                               plate_surface_heights, anchor_component_name)
from perception.change_detector import detect_change
from perception.stl_matcher import silhouette, load_models, read_stl, model_to_board
from tests.test_stl_matcher import box


def parts_in_plan(data):
    """Every placed part id, in placement order, across all plan steps."""
    return [op['part'] for step in data['assembly_plan']['steps'] for op in step['operations']]


class AnchorTests(unittest.TestCase):
    def test_metric_fit_corrects_height_parallax(self):
        K = np.array([[1100.,0,640],[0,1100.,360],[0,0,1]])
        mesh = box((8,80,5.18))
        for angle in (.5, 2.6):
            pose = SimpleNamespace(rvec=np.array([angle,0.,0.]), tvec=np.array([0.,0.,500.]))
            mask = silhouette(mesh, 33, (25,18), pose, K, (720,1280))
            before = np.full((720,1280,3), 200, np.uint8)
            after = before.copy()
            after[mask != 0] = 30
            region = detect_change(before, after)
            result = estimate_anchor(mesh, region, pose, K, 30)
            np.testing.assert_allclose(result['xy_mm'], [25,18], atol=1.)
            self.assertLess(abs(result['yaw_deg']-33), 2)

    def test_registration_maps_original_mesh_to_fitted_anchor(self):
        folder = Path(__file__).resolve().parents[1] / 'Fusion_output'
        if not folder.exists():
            self.skipTest('Local Fusion export missing')
        name = anchor_component_name(folder)
        mesh = load_models(folder)[name]
        pose = SimpleNamespace(rvec=np.zeros(3), tvec=np.array([0.,0.,500.]))
        estimate = {'xy_mm': np.array([150.,100.]), 'yaw_deg': 25}
        data, transform = register_cad(folder, name, estimate, pose)
        first = data['assembly_plan']['steps'][0]['operations'][0]['part']
        part = next(p for p in data['parts'] if p['id'] == first)
        local = read_stl(folder / 'meshes' / (name+'.stl'))
        root = local @ np.asarray(part['rotation']).T + np.asarray(part['position'])
        board = root @ transform[:3,:3].T + transform[:3,3]
        expected = model_to_board(mesh,25,[150,100],pose)
        np.testing.assert_allclose(board,expected,atol=1e-8)
        self.assertAlmostEqual(np.linalg.det(transform[:3,:3]),1.)
        for part_id in parts_in_plan(data):
            scene = placement_scene(data,transform,part_id)
            self.assertEqual(len(scene[0]),4)
            self.assertLess(scene[0][0][0][2],0.)
        # Height inference is asserted on the plate specifically: its 3.33 mm body
        # deck is the physically measured figure, independent of which part anchors.
        heights = plate_surface_heights(load_models(folder)['Plate 1x10 Silver'])
        self.assertAlmostEqual(heights['body'],3.33,places=2)
        self.assertAlmostEqual(heights['max'],5.18,places=2)
        original = transform.copy()
        for part_id in [p for p in parts_in_plan(data) if p.startswith('Plate 1x10 Silver')]:
            upper = placement_scene(data,transform,part_id,height_mm=heights['max'])
            lower = placement_scene(data,transform,part_id,height_mm=heights['body'])
            for a,b in zip(upper[0],lower[0]):
                np.testing.assert_allclose(a[0][:2],b[0][:2])
                self.assertAlmostEqual(b[0][2]-a[0][2],1.85,places=2)
        np.testing.assert_array_equal(transform,original)


if __name__ == '__main__':
    unittest.main()
