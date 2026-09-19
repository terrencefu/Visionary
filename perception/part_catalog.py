"""Part signatures: the colour + shape knowledge the matcher is constrained to.

EDIT THIS FILE for your real parts. Everything here is a placeholder measured
from nominal LEGO geometry, not from your actual bricks under your actual light.

For every part the assembly sequence can ask for:

  * hsv_ranges  -- one or more OpenCV HSV ranges (H 0-179, S/V 0-255). Red needs
                   two, because hue wraps at 0/180.
  * outline_mm  -- the top-down silhouette as a polygon in millimetres, drawn in
                   board axes (x right, y DOWN). This polygon DEFINES theta = 0:
                   a part lying exactly like the drawing reads 0 deg.
  * symmetry_deg -- the part's rotational symmetry. A 2x4 brick looks identical
                   after 180 deg, so an angle error is only ever judged modulo
                   this. Use 360 for a part with no rotational symmetry.

Choosing parts that work well:
  * Prefer strong concavities. A near-rectangle is hard to angle-fit.
  * Prefer CHIRAL outlines. An equal-arm L is its own mirror image, so a
    flipped placement cannot be detected -- see yellow_l_plate below.
  * Distinct colours per part make the wrong-part check far more reliable.

LEGO geometry: 1 stud = 8.0 mm pitch.
"""
from dataclasses import dataclass, field

import numpy as np

STUD_MM = 8.0

HsvRange = tuple[tuple[int, int, int], tuple[int, int, int]]


def _studs(*points):
    return np.array(points, dtype=np.float64) * STUD_MM


def polygon_area(points):
    p = np.asarray(points, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


@dataclass(frozen=True)
class PartSignature:
    part_id: str
    hsv_ranges: tuple[HsvRange, ...]
    outline_mm: np.ndarray
    symmetry_deg: int = 360
    description: str = ""
    # Accept blobs whose area is within [min, max] x the area predicted from
    # outline_mm and the local board scale. Loose on purpose: a silhouette
    # grows a little under glare and shrinks a little under shadow.
    area_tolerance: tuple[float, float] = (0.35, 2.5)
    extra: dict = field(default_factory=dict)

    @property
    def area_mm2(self):
        return abs(polygon_area(self.outline_mm))


# --- HSV presets (TUNE THESE UNDER VENUE LIGHTING) ---------------------------
RED: tuple[HsvRange, ...] = (((0, 120, 70), (8, 255, 255)), ((170, 120, 70), (179, 255, 255)))
YELLOW: tuple[HsvRange, ...] = (((18, 110, 110), (34, 255, 255)),)
GREEN: tuple[HsvRange, ...] = (((40, 90, 50), (85, 255, 255)),)
BLUE: tuple[HsvRange, ...] = (((95, 130, 50), (125, 255, 255)),)

CATALOG: dict[str, PartSignature] = {}


def register(signature):
    CATALOG[signature.part_id] = signature
    return signature


register(PartSignature(
    part_id="red_l_plate",
    description="Corner plate, unequal arms (6 studs right, 4 studs down)",
    hsv_ranges=RED,
    # Chiral: no rotation maps this onto its mirror, so a flip is detectable.
    outline_mm=_studs((0, 0), (6, 0), (6, 2), (2, 2), (2, 4), (0, 4)),
    symmetry_deg=360,
))

register(PartSignature(
    part_id="yellow_l_plate",
    description="Plate 4x4 corner (LEGO 2639): equal 2-stud-wide arms",
    hsv_ranges=YELLOW,
    # KNOWN LIMITATION: equal arms make this its own mirror image (reflect
    # about y = x), so a flipped placement is invisible. Position and angle
    # are still recovered correctly.
    outline_mm=_studs((0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)),
    symmetry_deg=360,
))

register(PartSignature(
    part_id="green_wedge",
    description="Wedge plate 4x2: full width at the back, tapering to the tip",
    hsv_ranges=GREEN,
    outline_mm=_studs((0, 0), (4, 1), (4, 2), (0, 2)),
    symmetry_deg=360,
))

register(PartSignature(
    part_id="blue_2x4",
    description="Brick 2x4. Rectangular, so it is only distinguishable mod 180 deg.",
    hsv_ranges=BLUE,
    outline_mm=_studs((0, 0), (4, 0), (4, 2), (0, 2)),
    symmetry_deg=180,
))


def get(part_id):
    try:
        return CATALOG[part_id]
    except KeyError:
        raise KeyError(f"unknown part id {part_id!r}; known: {sorted(CATALOG)}") from None
