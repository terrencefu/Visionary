"""Answer one constrained question: did the EXPECTED part appear, and where?

This is deliberately not open-set object recognition. The assembly state
machine always knows which part is due, so the matcher only has to verify one
hypothesis, and only searches the whole catalogue to explain a failure.

`HsvOutlineMatcher` is the MVP implementation: colour range plus top-down
outline. It is swappable -- anything with a `verify` method of the same shape
(a CAD-render embedding matcher, say) can replace it without the pipeline or
the state machine changing. See `PartMatcher` below.
"""
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from perception import part_catalog
from perception.pose_estimator import PlanarPose, align_outline

MIN_SCORE = 0.85
# A rival part must beat the expected part by this much before we are willing
# to call a placement the WRONG part rather than just a poor match.
WRONG_PART_MARGIN = 0.10
MIN_BLOB_AREA_PX = 60
ROI_PAD_PX = 24


@dataclass
class MatchResult:
    status: str                  # ok | not_found | wrong_part | low_confidence
    part_id: str | None          # what was actually recognised, if anything
    contour: np.ndarray | None
    score: float
    pose_px: PlanarPose | None   # alignment is reused, never recomputed
    message: str

    @property
    def ok(self):
        return self.status == "ok"


class PartMatcher(Protocol):
    """The seam a CAD-render matcher plugs into."""

    def verify(self, frame, region, expected_part_id, px_per_mm) -> MatchResult:
        ...


def colour_mask(frame_bgr, hsv_ranges):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for low, high in hsv_ranges:
        mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _region_mask(shape, region, pad=ROI_PAD_PX):
    """A full-frame mask limiting the search to the changed region."""
    if region is None:
        return None
    box = region.bbox if hasattr(region, "bbox") else region
    x, y, w, h = (int(v) for v in box)
    mask = np.zeros(shape[:2], np.uint8)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(shape[1], x + w + pad), min(shape[0], y + h + pad)
    mask[y0:y1, x0:x1] = 255
    return mask


class HsvOutlineMatcher:
    """Colour range + top-down outline. The MVP; replace behind `PartMatcher`."""

    def __init__(self, catalog=None, min_score=MIN_SCORE,
                 wrong_part_margin=WRONG_PART_MARGIN):
        self.catalog = catalog if catalog is not None else part_catalog.CATALOG
        self.min_score = min_score
        self.wrong_part_margin = wrong_part_margin

    # --- internals ------------------------------------------------------------

    def _best_contour(self, frame, region_mask, signature, px_per_mm):
        mask = colour_mask(frame, signature.hsv_ranges)
        if region_mask is not None:
            mask &= region_mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        expected_px2 = signature.area_mm2 * px_per_mm ** 2
        low, high = signature.area_tolerance
        plausible = [c for c in contours
                     if cv2.contourArea(c) >= MIN_BLOB_AREA_PX
                     and low * expected_px2 <= cv2.contourArea(c) <= high * expected_px2]
        if not plausible:
            return None
        return max(plausible, key=cv2.contourArea)

    def _score_part(self, frame, region_mask, signature, px_per_mm):
        contour = self._best_contour(frame, region_mask, signature, px_per_mm)
        if contour is None:
            return None, None
        return contour, align_outline(contour, signature.outline_mm, px_per_mm)

    # --- interface ------------------------------------------------------------

    def verify(self, frame, region, expected_part_id, px_per_mm):
        """Did `expected_part_id` appear in `region`? Explain a 'no'."""
        if expected_part_id not in self.catalog:
            raise KeyError(f"unknown part id {expected_part_id!r}; "
                           f"known: {sorted(self.catalog)}")
        signature = self.catalog[expected_part_id]
        region_mask = _region_mask(frame.shape, region)
        contour, pose = self._score_part(frame, region_mask, signature, px_per_mm)

        if pose is not None and pose.score >= self.min_score:
            return MatchResult("ok", expected_part_id, contour, pose.score, pose,
                               f"{expected_part_id} matched at {pose.score:.2f}")

        rival = self._identify(frame, region_mask, px_per_mm, skip=expected_part_id)
        expected_score = pose.score if pose is not None else 0.0
        if rival is not None and rival.score >= self.min_score \
                and rival.score >= expected_score + self.wrong_part_margin:
            return MatchResult("wrong_part", rival.part_id, rival.contour, rival.score,
                               rival.pose_px,
                               f"expected {expected_part_id}, but this looks like "
                               f"{rival.part_id} ({rival.score:.2f})")

        if contour is None:
            return MatchResult("not_found", None, None, 0.0, None,
                               f"no {expected_part_id}-coloured candidate in the changed region")
        return MatchResult("low_confidence", None, contour, expected_score, pose,
                           f"candidate does not fit {expected_part_id} "
                           f"({expected_score:.2f} < {self.min_score:.2f})")

    def identify(self, frame, region, px_per_mm):
        """Search the whole catalogue. Diagnostic and wrong-part reporting only."""
        found = self._identify(frame, _region_mask(frame.shape, region), px_per_mm)
        if found is None:
            return MatchResult("not_found", None, None, 0.0, None,
                               "no catalogue part found in the region")
        return found

    def _identify(self, frame, region_mask, px_per_mm, skip=None):
        best = None
        for part_id, signature in self.catalog.items():
            if part_id == skip:
                continue
            contour, pose = self._score_part(frame, region_mask, signature, px_per_mm)
            if pose is None:
                continue
            if best is None or pose.score > best.score:
                best = MatchResult("ok", part_id, contour, pose.score, pose,
                                   f"{part_id} at {pose.score:.2f}")
        return best
