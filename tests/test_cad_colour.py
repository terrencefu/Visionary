import copy
import json
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
from perception.cad_colour import resolve_colour, matches_colour
from perception.colour_change import colour_mask
from assembly.manual_guidance import ManualAssembly


def record(rgb):
    return dict(status='solid',rgb=rgb)


class CADColourTests(unittest.TestCase):
    def profile(self,rgb):
        return resolve_colour({'id':'arbitrary:1','component':'arbitrary','color':record(rgb)},
                              {'id':'arbitrary'},strict=True)

    def test_rgb_order_and_red_wrap(self):
        red,blue=self.profile([255,0,0]),self.profile([0,0,255])
        pixels=np.array([[[0,0,255],[255,0,0]]],np.uint8)
        np.testing.assert_array_equal(colour_mask(pixels,red),[[255,0]])
        np.testing.assert_array_equal(colour_mask(pixels,blue),[[0,255]])
        self.assertEqual(len(red['hsv_ranges']),2)
        self.assertFalse(matches_colour(red,blue))
        self.assertTrue(matches_colour(red,self.profile([240,5,0])))

    def test_part_overrides_component_and_invalid_does_not_fall_back(self):
        part={'id':'thing:1','component':'thing','color':record([0,255,0])}
        definition={'id':'thing','color':record([255,0,0])}
        self.assertEqual(resolve_colour(part,definition,strict=True)['rgb'],[0,255,0])
        part['color']={'status':'textured','rgb':None}
        with self.assertRaises(ValueError): resolve_colour(part,definition,strict=True)
        del part['color']
        self.assertEqual(resolve_colour(part,definition,strict=True)['source'],'component.color')

    def test_missing_and_mixed_fail_strict_preflight(self):
        part={'id':'thing:1','component':'thing'}
        with self.assertRaisesRegex(ValueError,'missing CAD color'):
            resolve_colour(part,{'id':'thing'},strict=True)
        part['color']={'status':'mixed','rgb':[255,0,0],'representative_area_fraction':.5}
        with self.assertRaisesRegex(ValueError,'mixed-colour'):
            resolve_colour(part,{'id':'thing'},strict=True)

    def test_arbitrary_component_names_and_per_occurrence_colours(self):
        folder=Path('Fusion_output')
        data=json.loads((folder/'assembly.json').read_text())
        for i,part in enumerate(data['parts']):
            part['color']=record([255,0,0] if i%2 else [0,0,255])
        with patch('config.PLACEMENT_PART_COLOURS',{}):
            machine=ManualAssembly(data,np.eye(4),folder,{},strict_colours=True)
        parts={p['id']:p for p in data['parts']}
        for operation,profile,mesh in zip(machine.operations,machine.colours,machine.part_models):
            self.assertEqual(profile['rgb'],parts[operation['part']]['color']['rgb'])
            self.assertAlmostEqual(mesh[...,2].min(),0.)
        # A second occurrence may have a different orientation from the first.
        altered=copy.deepcopy(data)
        plates=[p for p in altered['parts'] if 'Plate' in p['component']]
        plates[1]['rotation']=(np.array([[0,-1,0],[1,0,0],[0,0,1]])@np.array(plates[1]['rotation'])).tolist()
        machine=ManualAssembly(altered,np.eye(4),folder,{},strict_colours=True)
        i=next(i for i,op in enumerate(machine.operations) if op['part']==plates[1]['id'])
        extent=np.ptp(machine.part_models[i],axis=(0,1))
        self.assertGreater(extent[1],extent[0])
