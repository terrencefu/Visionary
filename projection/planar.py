"""Stationary board-XY to projector-UV mapping; independent of 3D calibration."""
import cv2
import numpy as np
import config


def transform(H, xy):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    q = np.c_[xy, np.ones(len(xy))] @ np.asarray(H).T
    if not np.isfinite(q).all() or np.any(np.abs(q[:, 2]) < 1e-9):
        raise ValueError("Invalid planar projection / homography horizon.")
    return q[:, :2] / q[:, 2:]


def fit_mapping(xy, uv):
    xy, uv = np.asarray(xy, float), np.asarray(uv, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape != uv.shape or len(xy) < 4:
        raise ValueError("Need at least four paired board XY / projector UV points; prefer 8-12+.")
    if not np.isfinite(xy).all() or not np.isfinite(uv).all():
        raise ValueError("Non-finite planar correspondences.")
    H, mask = cv2.findHomography(xy, uv, cv2.RANSAC, config.PLANAR_RANSAC_PX)
    if H is None or mask is None or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        raise ValueError("Degenerate planar fit; collect non-collinear points.")
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 4:
        raise ValueError("Fewer than four homography inliers.")
    hull = cv2.convexHull(xy[inliers].astype(np.float32)).reshape(-1, 2)
    x0, y0, x1, y1 = config.BOARD_BOUNDS_MM
    coverage = cv2.contourArea(hull) / ((x1-x0)*(y1-y0))
    span = np.ptp(xy[inliers], axis=0) / [x1-x0, y1-y0]
    if coverage < config.PLANAR_MIN_COVERAGE or np.any(span < .4):
        raise ValueError("Inliers are too clustered: cover >=20% board area and >=40% of each axis.")
    errors = np.linalg.norm(transform(H, xy)-uv, axis=1)
    return dict(H=H.tolist(), hull=hull.tolist(), inliers=inliers.tolist(),
                errors_px=errors.tolist(), rms_px=float(np.sqrt(np.mean(errors[inliers]**2))),
                coverage=float(coverage), board_xy=xy.tolist(), projector_uv=uv.tolist())


def contains(mapping, xy):
    x, y = xy
    x0, y0, x1, y1 = config.BOARD_BOUNDS_MM
    return (x0 <= x <= x1 and y0 <= y <= y1 and
            cv2.pointPolygonTest(np.array(mapping['hull'], np.float32), (float(x), float(y)), False) >= 0)


def board_to_pixel(mapping, xy):
    if not contains(mapping, xy):
        raise ValueError("Target outside calibrated inlier hull / usable board region; no extrapolation.")
    uv = transform(mapping['H'], [xy])[0]
    if np.any(uv < 0) or np.any(uv >= config.PROJECTOR_SIZE):
        raise ValueError("Target outside projector image.")
    return uv


def warp_overlay(mapping, image):
    """Image spans BOARD_BOUNDS_MM: top-left=min XY, bottom-right=max XY."""
    height, width = image.shape[:2]
    if min(height, width) < 2:
        raise ValueError("Overlay must be at least 2x2 pixels.")
    x0, y0, x1, y1 = config.BOARD_BOUNDS_MM
    image_to_board = np.array([[(x1-x0)/(width-1), 0, x0],
                               [0, (y1-y0)/(height-1), y0], [0, 0, 1]])
    # Clip source to measured support, preventing guidance outside the calibrated hull.
    hull_image = transform(np.linalg.inv(image_to_board), mapping['hull'])
    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(hull_image).astype(np.int32), 255)
    clipped = cv2.bitwise_and(image, image, mask=mask)
    return cv2.warpPerspective(clipped, np.array(mapping['H']) @ image_to_board,
                               config.PROJECTOR_SIZE)


def held_out_targets(mapping, count=5):
    """Choose interior targets separated from ALL collected fit observations."""
    hull = np.asarray(mapping['hull'])
    center = hull.mean(axis=0)
    lo, hi = hull.min(axis=0), hull.max(axis=0)
    candidates = [center + .7*(np.array([x, y])-center)
                  for y in np.linspace(lo[1], hi[1], 9)
                  for x in np.linspace(lo[0], hi[0], 9)]
    observed = np.asarray(mapping['board_xy'])
    candidates = [p for p in candidates if contains(mapping, p)
                  and np.min(np.linalg.norm(observed-p, axis=1)) >= 5.0]
    chosen = []
    while candidates and len(chosen) < count:
        scores = [min(np.linalg.norm(p-q) for q in chosen) if chosen else -np.linalg.norm(p-center)
                  for p in candidates]
        chosen.append(candidates.pop(int(np.argmax(scores))))
    if len(chosen) < count:
        raise ValueError("Not enough distinct held-out targets in calibrated region.")
    return np.asarray(chosen)
