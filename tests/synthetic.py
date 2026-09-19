"""Hardware-free rendering of parts, for perception tests.

Image-space convention everywhere: x right, y DOWN, and a positive angle is
counter-clockwise *as seen on screen*.
"""
import cv2
import numpy as np


def rotate_ccw_on_screen(points, theta_deg):
    """Rotate (N,2) image-space points CCW on screen. Image y points down, so
    this is the transpose of the usual math-convention rotation."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.asarray(points, float) @ np.array([[c, s], [-s, c]]).T


def polygon_centroid(polygon):
    """Area centroid of a simple polygon (N,2). Not the vertex mean."""
    p = np.asarray(polygon, float)
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-12:
        return p.mean(axis=0)
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * area)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6.0 * area)
    return np.array([cx, cy])


def place_outline(outline_mm, center_px, theta_deg, px_per_mm):
    """Scale an outline (mm) to pixels, rotate it, and put its AREA CENTROID
    at center_px. Returns an (N,2) float polygon."""
    pts = np.asarray(outline_mm, float) * float(px_per_mm)
    pts = pts - polygon_centroid(pts)
    return rotate_ccw_on_screen(pts, theta_deg) + np.asarray(center_px, float)


def fill_polygon(shape, polygon_px, value=255):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(polygon_px).astype(np.int32)], int(value))
    return mask


def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def render_part(frame, outline_mm, center_px, theta_deg, px_per_mm, bgr):
    """Paint a filled part silhouette onto a BGR frame, in place."""
    poly = place_outline(outline_mm, center_px, theta_deg, px_per_mm)
    cv2.fillPoly(frame, [np.round(poly).astype(np.int32)], bgr)
    return frame


def blank_scene(shape=(480, 640), bgr=(90, 85, 80), noise=0):
    """A flat, slightly brown table-ish background."""
    frame = np.full((shape[0], shape[1], 3), bgr, np.uint8)
    if noise:
        rng = np.random.default_rng(0)
        frame = np.clip(frame.astype(np.int16)
                        + rng.integers(-noise, noise + 1, frame.shape), 0, 255).astype(np.uint8)
    return frame


def rotate_in_board(points, theta_deg):
    """Rotate (N,2) BOARD-frame points CCW from +x toward +y."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.asarray(points, float) @ np.array([[c, -s], [s, c]]).T


def project_part_to_image(outline_mm, x_mm, y_mm, theta_deg, rvec, tvec, K, dist):
    """Where a flat part lying on the board lands in the distorted camera image.
    The outline's AREA CENTROID is placed at (x_mm, y_mm)."""
    centred = np.asarray(outline_mm, float) - polygon_centroid(outline_mm)
    board_xy = rotate_in_board(centred, theta_deg) + np.array([float(x_mm), float(y_mm)])
    board_xyz = np.c_[board_xy, np.zeros(len(board_xy))]
    return cv2.projectPoints(board_xyz, rvec, tvec, K, dist)[0].reshape(-1, 2)


def render_part_on_board(frame, outline_mm, x_mm, y_mm, theta_deg, rvec, tvec, K, dist, bgr):
    """Paint a flat part onto a BGR frame as the camera would see it."""
    polygon = project_part_to_image(outline_mm, x_mm, y_mm, theta_deg, rvec, tvec, K, dist)
    cv2.fillPoly(frame, [np.round(polygon).astype(np.int32)], bgr)
    return frame
