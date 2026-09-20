"""New expected-colour pixels for later CAD placements; geometry is still checked."""
import cv2
import numpy as np
import config
from perception.change_detector import (
    _changed_mask, _blank_quads, region_from_mask, MAX_CHANGE_FRACTION,
)


def part_colour(name):
    return config.PLACEMENT_PART_COLOURS.get(name)


def colour_mask(frame, colour):
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError('Colour detection needs uint8 BGR frames.')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], np.uint8)
    for low, high in config.PLACEMENT_COLOUR_HSV[colour]:
        mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    return mask


def detect_colour_change(before, after, colour, *, exclude_quads=None, search_mask=None):
    """Return region and candidate mask; unchanged earlier same-colour parts are excluded.

    The ordinary diff is a motion/noise gate, not the silhouette. Colour gates
    remove grey engine surfaces and shadows. We never fill holes or bridge gaps
    with invented part pixels. Incomplete/occluded colour can still fail pose fit.
    """
    if before.shape != after.shape:
        raise ValueError('Before and after frame sizes differ.')
    changed = _blank_quads(_changed_mask(before, after, search_mask=search_mask), exclude_quads)
    area = changed.size if search_mask is None else np.count_nonzero(search_mask)
    if not area or np.count_nonzero(changed) > MAX_CHANGE_FRACTION * area:
        return None, np.zeros(changed.shape, np.uint8)
    previous = colour_mask(before, colour)
    current = colour_mask(after, colour)
    # Ignore tiny edge jitter around an already-installed piece of this colour.
    previous = cv2.dilate(previous, np.ones((3, 3), np.uint8))
    added = current & cv2.bitwise_not(previous) & changed
    added = cv2.morphologyEx(added, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    if search_mask is not None:
        added[search_mask == 0] = 0
    _blank_quads(added, exclude_quads)
    return region_from_mask(added, search_mask=search_mask), added


def expected_pose_seed(model, name, region, pose, K, base_z):
    """Orient ONLY the expected CAD mesh; identity was selected by colour already."""
    from perception.stl_matcher import verify
    # No competing engine/plate shapes and no identity score gate here. The
    # subsequent metric fit still enforces ANCHOR_MIN_OVERLAP and XY/yaw limits.
    _, scores, debug = verify({name: model}, name, region, pose, K,
                              min_score=0., margin=0., base_z=base_z)
    return scores[0][2], scores, debug
