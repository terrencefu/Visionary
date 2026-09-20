import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


def collection(values):
    return NS(count=len(values),item=lambda i:values[i])


def appearance(name,rgb=None,textured=False,prop_id='opaque_albedo'):
    value=NS(red=rgb[0],green=rgb[1],blue=rgb[2]) if rgb else None
    prop=NS(id=prop_id,name='Localized colour',hasConnectedTexture=textured,
            hasMultipleValues=False,value=value)
    return NS(id=name,name=name,appearanceProperties=collection([prop]))


class FusionColorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core=NS(ColorProperty=NS(cast=lambda value:value))
        adsk=NS(core=core,fusion=NS())
        path=Path(__file__).resolve().parents[1]/'assembly/AssemblyGuide/AssemblyGuide.py'
        spec=importlib.util.spec_from_file_location('fusion_color_export_test',path)
        cls.module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules',{'adsk':adsk,'adsk.core':core,'adsk.fusion':adsk.fusion}):
            spec.loader.exec_module(cls.module)

    def owner(self,faces,override=None):
        return NS(appearance=override,bRepBodies=collection([
            NS(faces=collection([NS(appearance=a,area=area) for a,area in faces]),
               area=sum(area for _,area in faces))]))

    def test_solid_rgb_hex_json(self):
        result=self.module.export_color(self.owner([(appearance('Blue',[10,40,220]),3)]))
        self.assertEqual(result['rgb'],[10,40,220])
        self.assertEqual(result['hex'],'#0A28DC')
        self.assertEqual(result['status'],'solid')
        json.dumps(result)

    def test_occurrence_override_does_not_change_component_colour(self):
        faces=[(appearance('Blue',[0,0,255]),2)]
        owner=self.owner(faces,appearance('Red',[255,0,0]))
        self.assertEqual(self.module.export_color(owner,True)['rgb'],[255,0,0])
        self.assertEqual(self.module.export_color(owner)['rgb'],[0,0,255])

    def test_mixed_palette_weighted_by_area_not_face_count(self):
        blue,red=appearance('Blue',[0,0,255]),appearance('Red',[255,0,0])
        result=self.module.export_color(self.owner([(blue,1),(blue,1),(red,8)]),True)
        self.assertEqual(result['rgb'],[255,0,0])
        self.assertEqual(result['representative_area_fraction'],.8)
        self.assertEqual(result['status'],'mixed')
        self.assertEqual(len(result['palette']),2)

    def test_texture_and_missing_colours_are_not_guessed(self):
        for app in (None,appearance('Blue named texture',textured=True)):
            result=self.module.export_color(self.owner([(app,2)]))
            self.assertIsNone(result['rgb'])
            self.assertEqual(result['status'],'unavailable')

    def test_ambiguous_properties_preserved_without_arbitrary_selection(self):
        app=appearance('Shader',[100,110,120],prop_id='unrecognized_tint')
        other=appearance('Reflect',[255,255,255],prop_id='reflection_tint')
        app.appearanceProperties=collection([app.appearanceProperties.item(0),other.appearanceProperties.item(0)])
        result=self.module.appearance_color(app)
        self.assertEqual(result['status'],'ambiguous_properties')
        self.assertIsNone(result['rgb'])
        self.assertEqual(len(result['color_properties']),2)
