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
# Fragments within this many pixels of each other belong to one placement.
# Large enough to bridge a same-colour overlap that produces no detected change,
# small enough to leave a genuinely separate object as its own region.
MERGE_GAP_PX = 14


@dataclass
class ChangeRegion:
    bbox: tuple           # x, y, w, h in the pixel space of the frames given
    area_px: float
    centroid_px: tuple
    contour: np.ndarray
    mask: np.ndarray


def _changed_mask(before, after, threshold=DIFF_THRESHOLD, search_mask=None):
    if before.shape != after.shape:
        raise ValueError("Before and after frame sizes differ.")
    diff = cv2.absdiff(before, after)
    if diff.ndim == 3:
        diff = diff.max(axis=2)          # a colour swap at equal luma still counts
    if search_mask is not None:
        diff = np.where(search_mask != 0, diff, 0).astype(np.uint8)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    mask = (diff >= threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if search_mask is not None:
        mask[search_mask == 0] = 0
    return mask


def _blank_quads(mask, quads, pad=MARKER_PAD_PX):
    for quad in quads or ():
        points = np.asarray(quad, float).reshape(-1, 2)
        centre = points.mean(axis=0)
        grown = centre + (points - centre) * (1.0 + pad / max(1e-6, np.abs(points - centre).max()))
        cv2.fillConvexPoly(mask, np.round(grown).astype(np.int32), 0)
    return mask


def detect_change(before, after, exclude_quads=None, threshold=DIFF_THRESHOLD,
                  min_area_px=MIN_CHANGE_AREA_PX, max_change_fraction=MAX_CHANGE_FRACTION,
                  merge_gap_px=None,
                  search_mask=None):
    """Largest plausible newly-changed region, or None.

    `exclude_quads` are marker corner quads, in the same pixel space as the
    frames; anything inside them is ignored.
    """
    mask = _blank_quads(_changed_mask(before, after, threshold, search_mask), exclude_quads)
    frame_area = mask.size if search_mask is None else np.count_nonzero(search_mask)
    if frame_area == 0:
        return None

    if np.count_nonzero(mask) > max_change_fraction * frame_area:
        return None                      # a hand, a light change, or the board moved

    # One placement can read as several blobs: where a part overlaps a
    # similarly coloured one, few pixels change and the silhouette splits.
    # Keeping only the biggest fragment discards the rest of the same part, so
    # group fragments that sit within merge_gap_px of each other and keep the
    # largest GROUP. The gap is small enough that a separate object elsewhere
    # on the board stays its own region.
    gap = MERGE_GAP_PX if merge_gap_px is None else int(merge_gap_px)
    if gap > 0:
        bridge = cv2.dilate(mask, np.ones((2*gap+1, 2*gap+1), np.uint8))
    else:
        bridge = mask
    groups, _ = cv2.findContours(bridge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for group in groups:
        # Score and report the ORIGINAL changed pixels; the dilation only decides
        # which fragments belong together, it must not inflate the silhouette.
        member = np.zeros_like(mask)
        cv2.drawContours(member, [group], -1, 255, cv2.FILLED)
        member &= mask
        area = float(np.count_nonzero(member))
        if min_area_px <= area <= max_change_fraction * frame_area:
            candidates.append((area, member, group))
    if not candidates:
        return None

    area, isolated, group = max(candidates, key=lambda item: item[0])
    moments = cv2.moments(isolated, binaryImage=True)
    centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    ys, xs = np.nonzero(isolated)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))
    return ChangeRegion(bbox, area, centroid, group, isolated)


def frames_are_still(previous, current, threshold=DIFF_THRESHOLD,
                     stillness_fraction=STILLNESS_FRACTION, search_mask=None):
    """Has the scene stopped moving? Cheap enough to run on every raw frame.

    Used to wait for the user's hand to leave before anything is measured.
    """
    mask = _changed_mask(previous, current, threshold, search_mask)
    area = mask.size if search_mask is None else np.count_nonzero(search_mask)
    if area == 0:
        return False
    changed = np.count_nonzero(mask) / float(area)
    return changed <= stillness_fraction


class StillnessGate:
    """Fires once the scene has been still for `required` consecutive frames."""

    def __init__(self, required=None):
        self.required = required if required is not None \
            else getattr(config, "PERCEPTION_STILL_FRAMES", 6)
        self.previous = None
        self.still_count = 0

    def update(self, frame, search_mask=None):
        if self.previous is None:
            self.previous = frame.copy()
            self.still_count = 0
            return False
        still = frames_are_still(self.previous, frame, search_mask=search_mask)
        self.still_count = self.still_count + 1 if still else 0
        self.previous = frame.copy()
        return self.still_count >= self.required

    def reset(self):
        self.previous = None
        self.still_count = 0
