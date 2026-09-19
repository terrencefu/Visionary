"""Compare an observed placement with the CAD target and say what to fix.

Sign convention, chosen so the output is directly actionable:

    dx_mm, dy_mm, dtheta_deg are the CORRECTION STILL TO APPLY,
    i.e. expected - observed. Positive dx means "move further along board +X".

Directions are given in BOARD axes, not as LEFT/RIGHT. Which way is "left"
depends on where the user is standing and where the projector is, so turning
+X into an arrow is the projection subsystem's job, not this module's.
"""
from dataclasses import dataclass

import numpy as np

TOLERANCE_MM = 3.0
TOLERANCE_DEG = 8.0


@dataclass
class PoseError:
    correct: bool
    status: str          # ok | wrong_position | wrong_angle | wrong_position_and_angle
    dx_mm: float
    dy_mm: float
    dtheta_deg: float
    distance_mm: float
    tol_mm: float
    tol_deg: float

    def as_dict(self):
        return {"correct": self.correct, "dx_mm": self.dx_mm, "dy_mm": self.dy_mm,
                "dtheta_deg": self.dtheta_deg}


def _field(pose, name):
    if isinstance(pose, dict):
        return float(pose[name])
    try:
        return float(getattr(pose, name))
    except AttributeError:
        raise KeyError(name) from None


def wrap_symmetric(angle_deg, symmetry_deg=360):
    """Fold an angle error into the smallest equivalent for a part with the
    given rotational symmetry. A 2x4 brick (180 deg) turned end for end is not
    misplaced, so its error is 0, not 180."""
    period = float(symmetry_deg)
    folded = float(angle_deg) % period
    if folded > period / 2.0:
        folded -= period
    return folded


def compare(observed, expected, tol_mm=TOLERANCE_MM, tol_deg=TOLERANCE_DEG,
            symmetry_deg=360):
    """Correction needed to bring `observed` onto `expected`. Both are poses in
    board millimetres/degrees, as a dict or any object with the same fields."""
    dx = _field(expected, "x_mm") - _field(observed, "x_mm")
    dy = _field(expected, "y_mm") - _field(observed, "y_mm")
    dtheta = wrap_symmetric(_field(expected, "theta_deg") - _field(observed, "theta_deg"),
                            symmetry_deg)
    distance = float(np.hypot(dx, dy))

    position_ok = distance <= tol_mm
    angle_ok = abs(dtheta) <= tol_deg
    status = {(True, True): "ok",
              (False, True): "wrong_position",
              (True, False): "wrong_angle",
              (False, False): "wrong_position_and_angle"}[(position_ok, angle_ok)]

    return PoseError(position_ok and angle_ok, status, float(dx), float(dy),
                     float(dtheta), distance, float(tol_mm), float(tol_deg))


def correction_text(error):
    """One short operator-facing line. Board axes, not arrows."""
    if error.correct:
        return "OK - placement within tolerance"

    parts = []
    if abs(error.dx_mm) > error.tol_mm / 2.0:
        parts.append(f"{'+X' if error.dx_mm > 0 else '-X'} {abs(error.dx_mm):.1f} mm")
    if abs(error.dy_mm) > error.tol_mm / 2.0:
        parts.append(f"{'+Y' if error.dy_mm > 0 else '-Y'} {abs(error.dy_mm):.1f} mm")
    if abs(error.dtheta_deg) > error.tol_deg / 2.0:
        parts.append(f"ROTATE {'CCW' if error.dtheta_deg > 0 else 'CW'} "
                     f"{abs(error.dtheta_deg):.1f} deg")
    return "MOVE " + ", ".join(parts) if parts else "OK - placement within tolerance"
