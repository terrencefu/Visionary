import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from tests.test_stl_matcher import box
from assembly.manual_guidance import ManualAssembly
from perception.assembly_matcher import depth_image
from perception.placement_evidence import replay
from assembly.demo_perception import main


class CADIntegrationTests(unittest.TestCase):
    def test_new_names_colours_preflight_and_placement_without_legacy_mapping(self):
        with tempfile.TemporaryDirectory() as directory,patch('config.PLACEMENT_PART_COLOURS',{}),patch('config.DATA_DIR',Path(directory)/'evidence'):
            root=Path(directory)
            definitions=[]; entries=[]
            for name,size in [('GenericSupport',(40,24,8)),('GenericInsert',(32,16,8))]:
                mesh=box(size)
                text='solid test\n'+''.join('facet normal 0 0 1\nouter loop\n'+''.join('vertex '+' '.join(map(str,p))+'\n' for p in tri)+'endloop\nendfacet\n' for tri in mesh)+'endsolid test\n'
                (root/f'{name}.stl').write_text(text)
                definitions.append(dict(id=name,bounding_box=dict(min=mesh.min(axis=(0,1)).tolist(),max=mesh.max(axis=(0,1)).tolist())))
                entries.append(dict(id=name,mesh=dict(relative_path=f'{name}.stl',units='mm',frame='component_local')))
            parts=[dict(id=name+':1',component=name,rotation=np.eye(3).tolist(),position=[0,0,z],color=dict(status='solid',rgb=rgb))
                   for name,z,rgb in [('GenericSupport',0,[120,120,120]),('GenericInsert',8,[0,255,0])]]
            data=dict(assembly=dict(units='mm'),components=definitions,parts=parts,assembly_plan=dict(steps=[dict(operations=[dict(id='a',type='PLACE',part=parts[0]['id']),dict(id='b',type='PLACE',part=parts[1]['id'],dependencies=['a'])])]))
            (root/'assembly.json').write_text(json.dumps(data));(root/'toCV_output.json').write_text(json.dumps(dict(components=entries)))
            with patch('sys.argv',['assemble','--cad',str(root),'--check-only']),patch('hardware.camera.Camera.__enter__',side_effect=AssertionError('No hardware during preflight')):
                main(integrated=True)
            T=np.diag([1.,-1.,-1.,1.])
            machine=ManualAssembly(data,T,root,{},strict_colours=True);machine.advance()
            pose=SimpleNamespace(rvec=np.zeros(3),tvec=np.array([0.,0.,400.]))
            K=np.array([[900.,0,320],[0,900.,240],[0,0,1]])
            before=np.full((480,640,3),180,np.uint8)
            for shift,wanted in ((0,True),(8,False)):
                target=machine.meshes[1]@T[:3,:3].T+[shift,0,0]
                mask=np.isfinite(depth_image(target,pose,K,(480,640)))
                after=before.copy();after[mask]=(0,255,0)
                result=machine.validate_frames(before,after,pose,K)
                self.assertEqual(result[0],wanted,result[1])
                self.assertEqual(machine.last_check['colour_verified']['rgb'],[0,255,0])
                if not wanted:
                    saved=next((root/'evidence').iterdir())
                    repeated=replay(saved)
                    self.assertEqual(repeated,result[:2])
