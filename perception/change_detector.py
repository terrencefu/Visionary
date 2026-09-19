"""Where did something appear between two frames? Localisation only.

Identity is deliberately NOT decided here. The diff blob is contaminated by
shadow, glare and projector light, so its shape is unreliable even when its
location is fine -- the matcher re-segments the part properly inside the region
this module returns.

Three hazards this module exists to handle:
  * the projector's own light changes the scene without anything being placed,
  * a hand or forearm dominates the difference while the user is reaching in,
  * markers change appearance under projected light and must never be candidates.
"""
from dataclasses import dataclass

import cv2
import numpy as np

import config

DIFF_THRESHOLD = 25          # per-pixel intensity change that counts as changed
MIN_CHANGE_AREA_PX = 400     # smaller than any real part at working distance
MAX_CHANGE_FRACTION = 0.25   # larger than this is a hand, not a part
STILLNESS_FRACTION = 0.002   # changed-pixel fraction still considered motionless
MARKER_PAD_PX = 12


@dataclass
class ChangeRegion:
    bbox: tuple           # x, y, w, h in the pixel space of the frames given
    area_px: float
    centroid_px: tuple
    contour: np.ndarray
    mask: np.ndarray


def _changed_mask(before, after, threshold=DIFF_THRESHOLD):
    if before.shape != after.shape:
        raise ValueError("Before and after frame sizes differ.")
    diff = cv2.absdiff(before, after)
    if diff.ndim == 3:
        diff = diff.max(axis=2)          # a colour swap at equal luma still counts
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    mask = (diff >= threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _blank_quads(mask, quads, pad=MARKER_PAD_PX):
    for quad in quads or ():
        points = np.asarray(quad, float).reshape(-1, 2)
        centre = points.mean(axis=0)
        grown = centre + (points - centre) * (1.0 + pad / max(1e-6, np.abs(points - centre).max()))
        cv2.fillConvexPoly(mask, np.round(grown).astype(np.int32), 0)
    return mask


def detect_change(before, after, exclude_quads=None, threshold=DIFF_THRESHOLD,
                  min_area_px=MIN_CHANGE_AREA_PX, max_change_fraction=MAX_CHANGE_FRACTION):
    """Largest plausible newly-changed region, or None.

    `exclude_quads` are marker corner quads, in the same pixel space as the
    frames; anything inside them is ignored.
    """
    mask = _blank_quads(_changed_mask(before, after, threshold), exclude_quads)
    frame_area = mask.shape[0] * mask.shape[1]

    if np.count_nonzero(mask) > max_change_fraction * frame_area:
        return None                      # a hand, a light change, or the board moved

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = [c for c in contours
                  if min_area_px <= cv2.contourArea(c) <= max_change_fraction * frame_area]
    if not candidates:
        return None

    contour = max(candidates, key=cv2.contourArea)
    moments = cv2.moments(contour)
    centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    isolated = np.zeros_like(mask)
    cv2.drawContours(isolated, [contour], -1, 255, cv2.FILLED)
    return ChangeRegion(tuple(int(v) for v in cv2.boundingRect(contour)),
                        float(cv2.contourArea(contour)), centroid, contour, isolated)


def frames_are_still(previous, current, threshold=DIFF_THRESHOLD,
                     stillness_fraction=STILLNESS_FRACTION):
    """Has the scene stopped moving? Cheap enough to run on every raw frame.

    Used to wait for the user's hand to leave before anything is measured.
    """
    mask = _changed_mask(previous, current, threshold)
    changed = np.count_nonzero(mask) / float(mask.shape[0] * mask.shape[1])
    return changed <= stillness_fraction


class StillnessGate:
    """Fires once the scene has been still for `required` consecutive frames."""

    def __init__(self, required=None):
        self.required = required if required is not None \
            else getattr(config, "PERCEPTION_STILL_FRAMES", 6)
        self.previous = None
        self.still_count = 0

    def update(self, frame):
        if self.previous is None:
            self.previous = frame.copy()
            self.still_count = 0
            return False
        still = frames_are_still(self.previous, frame)
        self.still_count = self.still_count + 1 if still else 0
        self.previous = frame.copy()
        return self.still_count >= self.required

    def reset(self):
        self.previous = None
        self.still_count = 0
