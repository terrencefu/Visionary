"""Manually confirmed CAD steps, with guidance on already installed surfaces."""
import json
from pathlib import Path
import cv2
import numpy as np
import config
from perception.anchor import placement_scene, estimate_anchor
from perception.stl_matcher import read_stl, load_models, verify


class ManualAssembly:
    def __init__(self, data, transform, folder, plate_heights):
        self.data, self.transform = data, transform.copy()
        self.heights = plate_heights
        self.index = 0
        self.checked_index = None
        self.models = load_models(folder)
        self.names = []
        self.operations = []
        seen = set()
        # A CAD step may hold several operations: the planner groups placements
        # whose dependencies allow them in any order. Perception can only judge
        # ONE new part per baseline, so flatten to a linear placement sequence.
        # Within a step the JSON order is kept; dependencies are checked against
        # everything already placed, which is stricter than per-step checking.
        for step in data['assembly_plan']['steps']:
            for op in step['operations']:
                if op['type'] != 'PLACE':
                    raise ValueError(f'Unsupported operation type {op["type"]!r}; '
                                     'the manual MVP only places parts')
                if not set(op.get('dependencies',[])) <= seen:
                    raise ValueError(f'Operation {op["id"]!r} depends on a part that is '
                                     'not placed earlier in the plan')
                self.operations.append(op)
                seen.add(op['id'])
        manifest = json.loads((Path(folder)/'toCV_output.json').read_text())
        entries = {c['id']:c for c in manifest['components']}
        definitions = {c['id']:c for c in data['components']}
        parts = {p['id']:p for p in data['parts']}
        self.meshes = []
        for op in self.operations:
            part = parts[op['part']]
            definition = definitions[part['component']]
            name = definition.get('fusion_component_name',part['component'])
            self.names.append(name)
            entry = entries[name]['mesh']
            if entry['units'] != 'mm' or entry['frame'] != 'component_local':
                raise ValueError('Meshes must use component-local mm')
            mesh = read_stl(Path(folder)/entry['relative_path'])
            self.meshes.append(mesh @ np.asarray(part['rotation']).T + np.asarray(part['position']))
        self.floor = float(self.meshes[0][...,2].min())

    @property
    def complete(self):
        return self.index >= len(self.operations)

    @property
    def message(self):
        if self.complete:
            return 'Assembly complete (placement checks passed)'
        return f"Step {self.index+1}/{len(self.operations)}: {self.operations[self.index]['part']}"

    def advance(self):
        if not self.complete:
            if self.index > 0 and self.checked_index != self.index:
                raise ValueError('Placement has not passed its position/angle check')
            self.index += 1
            self.checked_index = None

    def validate_frames(self, before, after, pose, K, *, exclude_quads=None, search_mask=None, before_pose=None, before_transform=None):
        """Later coloured parts use a colour-selected change mask, then metric checks."""
        from perception.colour_change import part_colour, detect_colour_change
        from perception.change_detector import detect_change
        self.checked_index = None
        colour = part_colour(self.names[self.index]) if self.index > 0 else None
        if colour:
            previous_mask = None
            if before_pose is not None:
                from perception.motion_compensation import aligned_colour_history
                previous_mask, visible = aligned_colour_history(
                    before, colour, before_pose, pose, K,
                    self.transform if before_transform is None else before_transform,
                    self.transform, self.meshes[:self.index+1])
                search_mask = visible if search_mask is None else search_mask & visible
            region, mask = detect_colour_change(before, after, colour,
                exclude_quads=exclude_quads, search_mask=search_mask, previous_mask=previous_mask)
            if region is None:
                debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                return False, f'Expected a NEW {colour} part; no accepted colour change. Cannot advance.', debug
        else:
            region = detect_change(before, after, exclude_quads=exclude_quads, search_mask=search_mask)
            if region is None:
                return False, 'No accepted new part change; cannot advance', np.zeros_like(after)
        return self.validate(region, pose, K, colour_verified=colour)

    def validate(self, region, pose, K, *, colour_verified=None):
        """Check the current addition at its expected CAD support height."""
        self.checked_index = None
        name = self.names[self.index]
        mesh = self.meshes[self.index]
        lo,hi = mesh.min(axis=(0,1)),mesh.max(axis=(0,1))
        root_center = np.array([(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]])
        expected = self.transform[:3,:3] @ root_center + self.transform[:3,3]
        yaw = float(np.degrees(np.arctan2(self.transform[1,0],self.transform[0,0])) % 360)
        if colour_verified:
            from perception.colour_change import expected_pose_seed
            initial_yaw,scores,debug = expected_pose_seed(self.models[name],name,region,pose,K,expected[2])
        else:
            status,scores,debug = verify(self.models,name,region,pose,K,base_z=expected[2])
            if status != 'CORRECT SHAPE':
                return False, f'{status}: {scores}', debug
            initial_yaw = scores[0][2]
        measured = estimate_anchor(self.models[name],region,pose,K,initial_yaw,base_z=expected[2])
        correction = expected[:2]-measured['xy_mm']
        distance = float(np.linalg.norm(correction))
        # User's LEGO MVP explicitly treats 180-degree symmetry as equivalent.
        angle = float((yaw-measured['yaw_deg']+90)%180-90)
        passed = distance <= config.PLACEMENT_TOLERANCE_MM and abs(angle) <= config.PLACEMENT_TOLERANCE_DEG
        if passed:
            self.checked_index = self.index
        message = (f'{"PLACEMENT OK" if passed else "ADJUST PLACEMENT"}: '
                   f'expected XY={np.round(expected[:2],2)}, observed XY={np.round(measured["xy_mm"],2)} mm; '
                   f'correction X={correction[0]:+.2f}, Y={correction[1]:+.2f} mm, yaw={angle:+.1f} deg; '
                   f'position error={distance:.2f} mm; CAD base Z={expected[2]:.2f} mm; '
                   f'overlap={measured["overlap"]:.3f}'
                   + (f'; detected colour={colour_verified}' if colour_verified else ''))
        return passed,message,debug

    def scene(self, height_mode='body'):
        if self.complete:
            return [],[]
        if self.index == 0:
            # Nothing is installed yet, so there is no surface to sample onto.
            # The height toggle belongs to the anchor, whose mesh it was measured
            # from; applying it to any later part would be a different part's mm.
            return placement_scene(self.data,self.transform,self.operations[0]['part'],
                                   self.heights[height_mode])
        target = self.meshes[self.index]
        low, high = target.min(axis=(0,1)), target.max(axis=(0,1))
        xy = np.array([[low[0],low[1]],[high[0],low[1]],
                       [high[0],high[1]],[low[0],high[1]]])
        installed = np.concatenate(self.meshes[:self.index])
        edges = []
        label = None
        # XY envelope is only a placement cue. Land each sample on the highest
        # existing CAD surface beneath it, not a future part's imaginary top.
        for a,b in zip(xy,np.roll(xy,-1,axis=0)):
            samples = np.linspace(a,b,max(2,int(np.linalg.norm(b-a)/1.)+1))
            previous = None
            for point in samples:
                z = surface_height(installed,point,self.floor)
                board = self.transform[:3,:3] @ np.r_[point,z] + self.transform[:3,3]
                if cv2.pointPolygonTest(np.asarray(config.DETECTION_WORKSPACE_MM,np.float32),
                                       tuple(map(float,board[:2])),False)<0:
                    raise ValueError('Assembly footprint extends beyond cardboard; reset anchor')
                if label is None:
                    label = board
                # Do not draw interpolated lines across a step/void in the support.
                if previous is not None and abs(previous[2]-board[2])<.75:
                    edges.append((previous,board,(0,255,0)))
                previous = board
        return edges,[(label,f'Step {self.index+1}',(0,255,0))]


def surface_height(triangles, xy, floor):
    """Highest mesh intersection of a vertical ray, preserving holes/overhangs."""
    a,b,c = triangles[:,0],triangles[:,1],triangles[:,2]
    v0,v1,v2 = b[:,:2]-a[:,:2],c[:,:2]-a[:,:2],np.asarray(xy)-a[:,:2]
    den = v0[:,0]*v1[:,1]-v0[:,1]*v1[:,0]
    valid = np.abs(den)>1e-10
    u = np.zeros(len(a)); v = np.zeros(len(a))
    u[valid] = (v2[valid,0]*v1[valid,1]-v2[valid,1]*v1[valid,0])/den[valid]
    v[valid] = (v0[valid,0]*v2[valid,1]-v0[valid,1]*v2[valid,0])/den[valid]
    valid &= (u>=-1e-6)&(v>=-1e-6)&(u+v<=1+1e-6)
    z = a[:,2]+u*(b[:,2]-a[:,2])+v*(c[:,2]-a[:,2])
    return max(float(floor),float(z[valid].max())) if np.any(valid) else float(floor)
