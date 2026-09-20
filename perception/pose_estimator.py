"""In-plane pose of a part silhouette: centre, angle, and how well it fits.

Image-space convention, matching the rest of this repo: x right, y DOWN, and a
positive angle is counter-clockwise *as seen on screen*.

`theta_deg` is defined by the part's outline drawing: a part lying exactly like
its `outline_mm` polygon reads 0 deg. That makes the outline, not the code, the
definition of the part's zero.

Angle comes from template alignment rather than `minAreaRect`, which is
ambiguous modulo 90 deg and cannot separate a part from its mirror image.
"""
from dataclasses import dataclass

import cv2
import numpy as np

COARSE_STEP_DEG = 4.0
REFINE_STEP_DEG = 0.5
CANVAS_PAD_PX = 6


@dataclass
class PlanarPose:
    x_px: float          # silhouette area centroid, image pixels
    y_px: float
    theta_deg: float     # [0, 360), CCW on screen, 0 = lying like outline_mm
    score: float         # template IoU in [0, 1]

    @property
    def part_axis_px(self):
        """Unit vector of the part's +x axis in image coordinates."""
        t = np.radians(self.theta_deg)
        return np.array([np.cos(t), -np.sin(t)])

    def as_dict(self):
        return {"x_px": self.x_px, "y_px": self.y_px,
                "theta_deg": self.theta_deg, "score": self.score}


def rotate_ccw_on_screen(points, theta_deg):
    """Rotate (N,2) image-space points CCW on screen (image y points down)."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.asarray(points, float) @ np.array([[c, s], [-s, c]]).T


def polygon_centroid(polygon):
    """Area centroid of a simple polygon (N,2), not the vertex mean."""
    p = np.asarray(polygon, float)
    x, y = p[:, 0], p[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-12:
        return p.mean(axis=0)
    return np.array([((x + np.roll(x, -1)) * cross).sum(),
                     ((y + np.roll(y, -1)) * cross).sum()]) / (6.0 * area)


def contour_centroid(contour):
    # Shape matters: cv2.moments reads an (N,2) array as a raster image, and
    # only an (N,1,2) point set as a contour.
    points = np.asarray(contour, np.float32).reshape(-1, 1, 2)
    m = cv2.moments(points)
    if abs(m["m00"]) < 1e-9:
        return points.reshape(-1, 2).astype(float).mean(axis=0)
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])


def _canvas_size(*point_sets):
    radius = max(float(np.abs(p).max()) for p in point_sets)
    return int(2 * np.ceil(radius)) + 2 * CANVAS_PAD_PX + 1


def _rasterise(points_centred, size):
    mask = np.zeros((size, size), np.uint8)
    offset = np.array([size // 2, size // 2], float)
    cv2.fillPoly(mask, [np.round(points_centred + offset).astype(np.int32)], 255)
    return mask


def _iou(a, b):
    union = np.count_nonzero(a | b)
    return 0.0 if union == 0 else float(np.count_nonzero(a & b)) / union


def align_outline(contour, outline_mm, px_per_mm,
                  coarse_step_deg=COARSE_STEP_DEG, refine_step_deg=REFINE_STEP_DEG):
    """Fit `outline_mm` to an observed contour; return its in-plane pose.

    Both shapes are centred on their own area centroid, so only rotation is
    searched. The score is the best intersection-over-union achieved.
    """
    contour = np.asarray(contour, float).reshape(-1, 2)
    if len(contour) < 3:
        raise ValueError("Contour needs at least 3 points.")
    outline = np.asarray(outline_mm, float).reshape(-1, 2) * float(px_per_mm)
    if len(outline) < 3:
        raise ValueError("Outline needs at least 3 points.")

    centre = contour_centroid(contour)
    contour_centred = contour - centre
    outline_centred = outline - polygon_centroid(outline)

    size = _canvas_size(contour_centred, outline_centred)
    observed = _rasterise(contour_centred, size)

    def score_at(angle):
        return _iou(observed, _rasterise(rotate_ccw_on_screen(outline_centred, angle), size))

    coarse = np.arange(0.0, 360.0, coarse_step_deg)
    best = max(coarse, key=score_at)
    fine = np.arange(best - coarse_step_deg, best + coarse_step_deg + 1e-9, refine_step_deg)
    best = max(fine, key=score_at)

    return PlanarPose(float(centre[0]), float(centre[1]),
                      float(best % 360.0), float(score_at(best)))


@dataclass
class BoardPlanarPose:
    """A part's placement on the board plane, in millimetres and degrees.

    x_mm / y_mm are the AREA CENTROID of the part silhouette, not the origin of
    its outline drawing. A CAD target pose must use the same reference point.

    theta_deg is the angle of the part's +x axis measured in the BOARD frame,
    CCW from board +x toward board +y, in [0, 360). It is measured in a metric
    rectification of the board plane (perception/rectify.py), so neither the
    camera angle nor the upside-down mount can leak into it.
    """
    x_mm: float
    y_mm: float
    theta_deg: float
    z_mm: float = 0.0
    score: float = 0.0

    @property
    def axis_board(self):
        """Unit vector of the part's +x axis in board coordinates."""
        t = np.radians(self.theta_deg)
        return np.array([np.cos(t), np.sin(t)])

    def as_dict(self):
        return {"x_mm": self.x_mm, "y_mm": self.y_mm,
                "z_mm": self.z_mm, "theta_deg": self.theta_deg}


@dataclass
class BoardPlanarPose:
    """A part's placement on the board plane, in millimetres and degrees.

    x_mm / y_mm are the AREA CENTROID of the part silhouette, not the origin of
    its outline drawing. A CAD target pose must use the same reference point.

    theta_deg is the angle of the part's +x axis measured in the BOARD frame,
    CCW from board +x toward board +y, in [0, 360). It is measured in a metric
    rectification of the board plane (perception/rectify.py), so neither the
    camera angle nor the upside-down mount can leak into it.
    """
    x_mm: float
    y_mm: float
    theta_deg: float
    z_mm: float = 0.0
    score: float = 0.0

    @property
    def axis_board(self):
        """Unit vector of the part's +x axis in board coordinates."""
        t = np.radians(self.theta_deg)
        return np.array([np.cos(t), np.sin(t)])

    def as_dict(self):
        return {"x_mm": self.x_mm, "y_mm": self.y_mm,
                "z_mm": self.z_mm, "theta_deg": self.theta_deg}
