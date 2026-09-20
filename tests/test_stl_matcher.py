import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from perception.change_detector import detect_change
from perception.stl_matcher import read_stl, silhouette, verify, load_models, model_to_board


def box(size):
    x, y, z = size
    vertices = np.array([[-x/2, -y/2, 0], [x/2, -y/2, 0], [x/2, y/2, 0], [-x/2, y/2, 0],
                         [-x/2, -y/2, z], [x/2, -y/2, z], [x/2, y/2, z], [-x/2, y/2, z]])
    faces = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
             [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    return vertices[faces]


class StlTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[900., 0, 320], [0, 900., 240], [0, 0, 1]])
        self.pose = SimpleNamespace(rvec=np.array([2.6, 0., 0.]), tvec=np.array([0., 0., 400.]))
        self.models = {'long': box((48, 8, 11)), 'wide': box((24, 20, 16))}

    def region(self, mesh):
        mask = silhouette(mesh, 33, (0, 0), self.pose, self.K, (480, 640))
        before = np.full((480, 640, 3), 200, np.uint8)
        after = before.copy()
        after[mask != 0] = 30
        return detect_change(before, after)

    def test_perspective_shape_identity_and_wrong_part(self):
        region = self.region(self.models['long'])
        status, scores, _ = verify(self.models, 'long', region, self.pose, self.K)
        self.assertEqual(status, 'CORRECT SHAPE', scores)
        self.assertEqual(scores[0][0], 'long')
        status, _, _ = verify(self.models, 'wide', region, self.pose, self.K)
        self.assertEqual(status, 'INCORRECT SHAPE')

    def test_identical_shapes_are_uncertain(self):
        models = {'one': self.models['long'], 'two': self.models['long']}
        status, _, _ = verify(models, 'one', self.region(models['one']), self.pose, self.K)
        self.assertEqual(status, 'UNCERTAIN')

    def test_negative_board_normal_verifies_part(self):
        self.pose = SimpleNamespace(rvec=np.array([0.5, 0., 0.]), tvec=np.array([0., 0., 400.]))
        status, scores, _ = verify(self.models, 'long', self.region(self.models['long']),
                                   self.pose, self.K)
        self.assertEqual(status, 'CORRECT SHAPE', scores)

    def test_side_mapping_preserves_handedness_and_physical_up(self):
        negative = SimpleNamespace(rvec=np.zeros(3), tvec=np.array([0., 0., 400.]))
        basis = model_to_board(np.eye(3), 0, (0, 0), negative)
        self.assertAlmostEqual(np.linalg.det(basis), 1)
        np.testing.assert_allclose(basis, np.diag([1, -1, -1]))
        # Equivalent positive-normal board convention produces the same image.
        positive = SimpleNamespace(rvec=np.array([np.pi, 0., 0.]), tvec=negative.tvec)
        a = silhouette(self.models['long'], 0, (0, 0), negative, self.K, (480, 640))
        b = silhouette(self.models['long'], 0, (0, 0), positive, self.K, (480, 640))
        np.testing.assert_array_equal(a, b)

    def test_model_behind_camera_is_rejected(self):
        pose = SimpleNamespace(rvec=np.zeros(3), tvec=np.array([0., 0., 5.]))
        with self.assertRaisesRegex(ValueError, 'camera plane'):
            silhouette(self.models['long'], 0, (0, 0), pose, self.K, (480, 640))

    def test_ascii_reader(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'part.stl'
            path.write_text('solid test\n vertex 0 0 0\n vertex 1 0 0\n vertex 0 1 0\nendsolid')
            self.assertEqual(read_stl(path).shape, (1, 3, 3))

    def test_real_export_alias_and_metric_bounds(self):
        folder = Path(__file__).resolve().parents[1] / 'Fusion_output'
        if not folder.exists():
            self.skipTest('Local Fusion export not available')
        models = load_models(folder)
        self.assertEqual(len(models), 3)
        self.assertIn('Plate 1x10 Silver', models)
        for mesh in models.values():
            self.assertAlmostEqual(mesh[..., 2].min(), 0)


if __name__ == '__main__':
    unittest.main()
