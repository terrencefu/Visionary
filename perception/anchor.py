"""Metric, perspective anchor fitting and CAD-root registration for the MVP."""
import json
from pathlib import Path

import cv2
import numpy as np

import config
from perception.geometry import board_point_from_undistorted_pixel
from perception.stl_matcher import model_to_board, silhouette


def estimate_anchor(mesh, region, pose, K, initial_yaw, base_z=0.0):
    """Fit XY/yaw in board mm. No image-space alignment in the final score."""
    xy = board_point_from_undistorted_pixel(*region.centroid_px, pose.rvec, pose.tvec, K)[:2]
    x, y, w, h = region.bbox
    pad = max(w, h)
    x0, y0 = max(0, x-pad), max(0, y-pad)
    x1, y1 = min(region.mask.shape[1], x+w+pad), min(region.mask.shape[0], y+h+pad)
    scale = min(1., 320 / max(x1-x0, y1-y0))
    observed = cv2.resize(region.mask[y0:y1, x0:x1], None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)
    crop_K = np.array([[scale, 0, -scale*x0], [0, scale, -scale*y0], [0, 0, 1]]) @ K
    def evaluate(params):
        rendered = silhouette(mesh, params[2], params[:2], pose, crop_K, observed.shape, base_z=base_z)
        union = np.count_nonzero(observed | rendered)
        score = np.count_nonzero(observed & rendered) / max(1, union)
        return score, rendered
    params = np.array([*xy, initial_yaw], float)
    # Remove height-induced centroid parallax by solving centroid residuals in mm.
    obs_m = cv2.moments(observed)
    target = np.array([obs_m['m10'], obs_m['m01']]) / obs_m['m00']
    def centroid(p):
        m = cv2.moments(evaluate(p)[1])
        if not m['m00']:
            raise ValueError('Anchor model is outside the camera view')
        return np.array([m['m10'], m['m01']]) / m['m00']
    for _ in range(5):
        current = centroid(params)
        jac = np.column_stack([centroid(params + d)-current
                               for d in ([1, 0, 0], [0, 1, 0])])
        delta = np.linalg.lstsq(jac, target-current, rcond=None)[0]
        params[:2] += np.clip(delta, -10, 10)
        if np.linalg.norm(delta) < .1:
            break
    best = evaluate(params)[0]
    for step_mm, step_deg in ((2., 3.), (.75, 1.), (.25, .5)):
        for _ in range(4):
            improved = False
            for axis, step in enumerate((step_mm, step_mm, step_deg)):
                for direction in (-1, 1):
                    candidate = params.copy()
                    candidate[axis] += direction * step
                    score = evaluate(candidate)[0]
                    if score > best:
                        best, params, improved = score, candidate, True
            if not improved:
                break
    if best < config.ANCHOR_MIN_OVERLAP:
        raise ValueError(f'Anchor pose uncertain: unshifted silhouette overlap '
                         f'{best:.3f} < {config.ANCHOR_MIN_OVERLAP:.3f}')
    return {'xy_mm': params[:2], 'yaw_deg': float(params[2] % 360), 'overlap': float(best)}


def anchor_component_name(folder):
    """Component name the plan places first. The anchor follows the CAD, not a
    fixed part: re-exporting the assembly can change which part comes first."""
    data = json.loads((Path(folder) / 'assembly.json').read_text())
    first = data['assembly_plan']['steps'][0]['operations'][0]['part']
    part = next(p for p in data['parts'] if p['id'] == first)
    definition = next(c for c in data['components'] if c['id'] == part['component'])
    return definition.get('fusion_component_name', part['component'])


def register_cad(folder, anchor_component, estimate, pose):
    data = json.loads((Path(folder) / 'assembly.json').read_text())
    first = data['assembly_plan']['steps'][0]['operations'][0]
    parts = {p['id']: p for p in data['parts']}
    definitions = {c['id']: c for c in data['components']}
    part = parts[first['part']]
    definition = definitions[part['component']]
    if definition.get('fusion_component_name', part['component']) != anchor_component:
        raise ValueError(
            f'Anchor must be the first placed component. This plan starts with '
            f'{definition.get("fusion_component_name", part["component"])!r}, '
            f'not {anchor_component!r}.')
    from perception.stl_matcher import read_stl
    manifest = json.loads((Path(folder) / 'toCV_output.json').read_text())
    entry = next(c for c in manifest['components'] if c['id'] == anchor_component)
    mesh = read_stl(Path(folder) / entry['mesh']['relative_path'])
    rotated = mesh @ np.asarray(part['rotation']).T
    lo, hi = rotated.min(axis=(0, 1)), rotated.max(axis=(0, 1))
    shift = np.array([(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, lo[2]])
    rotation = model_to_board(np.eye(3), estimate['yaw_deg'], (0, 0), pose).T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.r_[estimate['xy_mm'], 0.] - rotation @ (shift + np.asarray(part['position']))
    return data, transform


def plate_surface_heights(mesh):
    """Infer the broad body deck from horizontal STL triangle area below studs."""
    mesh = np.asarray(mesh, float)
    low, high = mesh[..., 2].min(), mesh[..., 2].max()
    flat = np.ptp(mesh[..., 2], axis=1) < 1e-5
    heights = np.round(mesh[..., 2].mean(axis=1), 4)
    area = np.abs(np.cross(mesh[:,1]-mesh[:,0], mesh[:,2]-mesh[:,0])[:,2]) / 2
    candidates = [h for h in np.unique(heights[flat]) if low+0.1 < h < high-0.1]
    if not candidates:
        raise ValueError('No distinct body deck found in STL; cannot run height comparison')
    body = max(candidates, key=lambda h: area[flat & (heights == h)].sum())
    return {'max': float(high-low), 'body': float(body-low)}


def placement_scene(data, transform, part_id, height_mm=None, color=(0, 255, 0)):
    """Top bounding-face outline for one placed part, in board mm.

    Addressed by part id rather than plan-step index: a CAD step may hold
    several operations, so step number and placement number are not the same.
    """
    part = next(p for p in data['parts'] if p['id'] == part_id)
    definition = next(c for c in data['components'] if c['id'] == part['component'])
    lo, hi = np.array(definition['bounding_box']['min']), np.array(definition['bounding_box']['max'])
    corners = np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
    root = corners @ np.asarray(part['rotation']).T + np.asarray(part['position'])
    top = root[np.isclose(root[:,2], root[:,2].max())]
    if len(top) != 4:
        raise ValueError('Expected a horizontal four-corner top face')
    if height_mm is not None:
        if not np.isfinite(height_mm) or not 0 <= height_mm <= np.ptp(root[:,2])+1e-5:
            raise ValueError('Diagnostic surface height outside part bounds')
        top[:,2] = root[:,2].min() + height_mm
    center = top.mean(axis=0)
    top = top[np.argsort(np.arctan2(top[:,1]-center[1], top[:,0]-center[0]))]
    board = top @ transform[:3,:3].T + transform[:3,3]
    import config
    workspace = np.asarray(config.DETECTION_WORKSPACE_MM, np.float32)
    if any(cv2.pointPolygonTest(workspace, tuple(map(float,p[:2])), False) < 0 for p in board):
        raise ValueError('Target is outside the cardboard; reposition the anchor and retry')
    return ([(a,b,color) for a,b in zip(board,np.roll(board,-1,axis=0))],
            [(board.mean(axis=0), part_id.split(':')[0][:28], color)])
