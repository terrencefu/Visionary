import unittest
from types import SimpleNamespace
import numpy as np
from projection.stationary import StationaryMonitor


class StationaryTests(unittest.TestCase):
    def setUp(self):
        self.ref=SimpleNamespace(rvec=np.zeros(3),tvec=np.array([0.,0.,1000.]),
                                 object_points=np.array([[0.,0.,0.],[100.,0.,0.],[100.,100.,0.],[0.,100.,0.]]))
        self.guard=StationaryMonitor(self.ref,np.diag([1000.,1000.,1.]),np.zeros(5))

    def feed(self,x,t):
        pose=SimpleNamespace(rvec=np.zeros(3),tvec=np.array([x,0.,1000.]))
        return self.guard.update(pose,t)

    def test_isolated_noise_blanks_then_recovers(self):
        for i in range(12): self.feed(0.,i*.04)
        spike=self.feed(1.57,.5)
        self.assertFalse(spike['ready'])
        self.assertFalse(spike['stop'])
        self.assertTrue(self.feed(0.,.54)['ready'])

    def test_persistent_offset_stops_and_does_not_rebase(self):
        results=[self.feed(1.57,i*.04) for i in range(40)]
        self.assertTrue(results[-1]['stop'])
        self.assertIsNone(self.guard.base)

    def test_deliberate_movement_stops(self):
        for i in range(12): self.feed(0.,i*.04)
        results=[self.feed(3.,.5+i*.04) for i in range(40)]
        self.assertTrue(results[-1]['stop'])

    def test_loss_recovers_without_erasing_anchor(self):
        for i in range(12): self.feed(0.,i*.04)
        base=self.guard.base.copy()
        self.assertFalse(self.guard.update(None,.5)['stop'])
        for i in range(12): result=self.feed(0.,.6+i*.04)
        self.assertTrue(result['ready'])
        np.testing.assert_array_equal(base,self.guard.base)
        self.guard.update(None,2.)
        self.assertTrue(self.guard.update(None,4.1)['stop'])
