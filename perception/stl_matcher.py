"""Experimental visible-silhouette verification for isolated parts on the board.

Uses metric perspective renders, not the STL bounding box. Assumes the part
rests in its exported assembly orientation, allowing yaw but not tumbling.
Scores are overlap measures, not calibrated probabilities or placement proof.
"""
import json
from pathlib import Path
import struct

import cv2
import numpy as np

import config
from perception.geometry import board_point_from_undistorted_pixel, camera_height_above_board


def read_stl(path):
    data = Path(path).read_bytes()
    count = struct.unpack_from('<I', data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + count * 50:
        dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])
        triangles = np.frombuffer(data, dtype=dtype, offset=84)['vertices'].astype(float)
    else:
        vertices = [list(map(float, line.split()[1:]))
                    for line in data.decode('ascii').splitlines()
                    if line.strip().startswith('vertex ')]
        triangles = np.asarray(vertices, float).reshape(-1, 3, 3)
    if not triangles.size or not np.isfinite(triangles).all():
        raise ValueError(f'Invalid STL geometry: {path}')
    return triangles


def load_models(folder):
    folder = Path(folder)
    assembly = json.loads((folder / 'assembly.json').read_text())
    manifest = json.loads((folder / 'toCV_output.json').read_text())
    if assembly['assembly']['units'] != 'mm':
        raise ValueError('Assembly units must be mm')
    meshes = {c['id']: c['mesh'] for c in manifest['components']}
    definitions = {c['id']: c for c in assembly['components']}
    models = {}
    for part in assembly['parts']:
        definition = definitions[part['component']]
        name = definition.get('fusion_component_name', part['component'])
        if name in models:
            continue
        mesh = meshes[name]
        if mesh['units'] != 'mm' or mesh['frame'] != 'component_local':
            raise ValueError(f'Unsupported mesh units/frame: {name}')
        triangles = read_stl(folder / mesh['relative_path'])
        actual = np.array([triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))])
        expected = np.array([definition['bounding_box']['min'], definition['bounding_box']['max']])
        if not np.allclose(actual, expected, atol=0.1):
            raise ValueError(f'Mesh bounds do not match assembly component: {name}')
        rotation = np.asarray(part['rotation'], float)
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0:
            raise ValueError(f'Invalid rotation: {name}')
        triangles = triangles @ rotation.T
        low, high = triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))
        triangles -= np.array([(low[0]+high[0])/2, (low[1]+high[1])/2, low[2]])
        models[name] = triangles
    return models


def model_to_board(triangles, yaw, xy, pose):
    """Put CAD up on the camera-facing side with a proper (non-mirror) rotation.

    Board XY is fixed by the marker map; its +Z need not face the camera.
    For the negative side, rotate 180 degrees about X (flip Y AND Z).
    Flipping only Z would reflect chiral parts rather than rotate them.
    """
    height = camera_height_above_board(pose.rvec, pose.tvec)
    if not np.isfinite(height) or abs(height) < 1e-6:
        raise ValueError('Camera is on the board plane or pose is invalid')
    side = 1.0 if height > 0 else -1.0
    upright = np.asarray(triangles) * np.array([1.0, side, side])
    t = np.radians(yaw)
    rotation = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    return upright @ rotation.T + np.array([xy[0], xy[1], 0])


def silhouette(triangles, yaw, xy, pose, K, shape, base_z=0.0):
    points = model_to_board(triangles, yaw, xy, pose)
    points[...,2] += base_z
    camera = points @ cv2.Rodrigues(pose.rvec)[0].T + np.asarray(pose.tvec).reshape(3)
    if not np.isfinite(camera).all() or np.any(camera[..., 2] <= 0):
        raise ValueError('Model crosses camera plane')
    projected = cv2.projectPoints(points.reshape(-1, 3), pose.rvec, pose.tvec, K,
                                  np.zeros(5))[0].reshape(-1, 3, 2)
    mask = np.zeros(shape, np.uint8)
    for triangle in np.round(projected).astype(np.int32):
        cv2.fillConvexPoly(mask, triangle, 255)
    return mask


def centered_iou(observed, rendered):
    """Align silhouette centroids in image pixels without rescaling geometry."""
    a, b = cv2.moments(observed), cv2.moments(rendered)
    if not a['m00'] or not b['m00']:
        return 0.0, rendered
    delta = [a['m10']/a['m00']-b['m10']/b['m00'], a['m01']/a['m00']-b['m01']/b['m00']]
    aligned = cv2.warpAffine(rendered, np.array([[1, 0, delta[0]], [0, 1, delta[1]]], float),
                             (observed.shape[1], observed.shape[0]), flags=cv2.INTER_NEAREST)
    union = np.count_nonzero(observed | aligned)
    return np.count_nonzero(observed & aligned) / max(union, 1), aligned


def verify(models, expected, region, pose, K, min_score=None, margin=None, base_z=0.0):
    min_score = config.STL_MATCH_MIN_OVERLAP if min_score is None else min_score
    margin = config.STL_MATCH_MARGIN if margin is None else margin
    height = camera_height_above_board(pose.rvec, pose.tvec)
    print(f'STL board convention: camera Z={height:.1f} mm; '
          f'CAD up maps to board {"+Z" if height > 0 else "-Z"} (proper rotation).')
    if expected not in models:
        raise ValueError(f'Unknown component {expected!r}. Choose: {list(models)}')
    # Work on a padded image crop at reduced resolution; K follows the same transform.
    x, y, w, h = region.bbox
    pad = max(w, h)
    x0, y0 = max(0, x-pad), max(0, y-pad)
    x1, y1 = min(region.mask.shape[1], x+w+pad), min(region.mask.shape[0], y+h+pad)
    scale = min(1.0, 240 / max(x1-x0, y1-y0))
    observed = cv2.resize(region.mask[y0:y1, x0:x1], None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)
    crop_K = np.array([[scale, 0, -scale*x0], [0, scale, -scale*y0], [0, 0, 1]]) @ K
    xy = board_point_from_undistorted_pixel(*region.centroid_px, pose.rvec, pose.tvec, K)[:2]
    results = []
    for name, triangles in models.items():
        best = (-1, 0, None)
        def evaluate(angle):
            rendered = silhouette(triangles, angle, xy, pose, crop_K, observed.shape, base_z=base_z)
            if (np.any(rendered[0]) or np.any(rendered[-1]) or
                    np.any(rendered[:, 0]) or np.any(rendered[:, -1])):
                return 0.0, float(angle), rendered  # Never reward a clipped template.
            score, aligned = centered_iou(observed, rendered)
            return score, float(angle), aligned
        for angle in range(0, 360, 15):
            candidate = evaluate(angle)
            if candidate[0] > best[0]:
                best = candidate
        for angle in np.arange(best[1]-15, best[1]+15, 3):
            candidate = evaluate(angle)
            if candidate[0] > best[0]:
                best = candidate
        results.append((name, *best))
    results.sort(key=lambda row: row[1], reverse=True)
    winner = results[0]
    gap = winner[1] - results[1][1] if len(results) > 1 else winner[1]
    status = 'UNCERTAIN'
    if winner[1] >= min_score and gap >= margin:
        status = 'CORRECT SHAPE' if winner[0] == expected else 'INCORRECT SHAPE'
    debug = np.zeros((*observed.shape, 3), np.uint8)
    debug[..., 1] = observed
    debug[..., 2] = winner[3]
    return status, [(name, score, angle) for name, score, angle, _ in results], debug
