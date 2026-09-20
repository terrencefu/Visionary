import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from perception.placement_evidence import save_check
from perception.change_detector import detect_change
from perception.stl_matcher import silhouette
from tests.test_stl_matcher import box


class PlacementEvidenceTests(unittest.TestCase):
    def test_uncertain_fit_still_saves_exact_inputs(self):
        model = box((32,16,8))
        K = np.array([[800.,0,200],[0,800.,150],[0,0,1]])
        pose = SimpleNamespace(rvec=np.zeros(3),tvec=np.array([0.,0.,400.]))
        mask = silhouette(model,0,[0,0],pose,K,(300,400),base_z=-19.2)
        before = np.full((300,400,3),200,np.uint8)
        after = before.copy(); after[mask!=0] = 20
        region = detect_change(before,after)
        assembly = SimpleNamespace(models={'test':model},last_check={
            'component':'test','expected_xyz':[0.,0.,-19.2],'expected_yaw':0.,
            'scores':[('test',.9,0.)],'fit_rejection':'overlap too low'})
        with tempfile.TemporaryDirectory() as directory, patch('config.DATA_DIR',Path(directory)):
            folder = save_check(assembly,before,after,region,pose,K,'PLACEMENT UNCERTAIN')
            report = json.loads((folder/'report.json').read_text())
            self.assertEqual(report['fit_rejection'],'overlap too low')
            with np.load(folder/'inputs.npz',allow_pickle=False) as data:
                np.testing.assert_array_equal(data['model'],model)
                np.testing.assert_array_equal(data['mask'],region.mask)
            self.assertTrue((folder/'overlay_yellow_observed_green_expected_red_fit.png').exists())
