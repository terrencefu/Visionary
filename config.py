"""All physical measurements use millimeters. Enter your actual fixture here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAMERA_INDEX = 0  # DirectShow: webcam AC310 = 0; Integrated Camera = 1 on this setup.
CAMERA_SIZE = (1920, 1080)
PREVIEW_ROTATE_180 = True
CAMERA_CALIBRATION = ROOT / "camera_calibration_1080p.npz"
PROJECTOR_CALIBRATION = ROOT / "projector_calibration.npz"
DATA_DIR = ROOT / "data"
PLANAR_CALIBRATION = ROOT / "planar_calibration.json"
PLANAR_RANSAC_PX = 3.0
PLANAR_MIN_COVERAGE = 0.20  # Inlier convex hull / usable board area.
PLANAR_VALIDATION_MAX_MM = 5.0

ARUCO_DICTIONARY = "DICT_4X4_50"
# No defaults from the old temporary fixture. All marker tops face board -Y.
MARKER_SIZE_MM = 30.0  # User confirmed black-square measurement, excluding white border.
# Origin: marker 0 center. +X points toward marker 1; +Y toward the lower row.
# Latest approximate centers: horizontal spacing 303 mm; left 203 mm, right 204 mm.
# Preserve the measured asymmetry using the coordinates supplied by the user.
MARKER_CENTERS_MM = {
    0: (0.0, 0.0),
    1: (303.0, 0.0),
    2: (0.0, 203.0),
    3: (303.0, 204.0),
}
# Interior calibration region; visually verify the dots actually hit flat cardboard.
BOARD_BOUNDS_MM = (30.0, 30.0, 273.0, 173.0)
# Detection only: cardboard extends 30 mm beyond the marker-center boundary.
# Preserve the lower edge's measured 1 mm slope. Not projector calibration bounds.
DETECTION_WORKSPACE_MM = ((-30.0, -30.0), (333.0, -30.0),
                          (333.0, 234.0), (-30.0, 233.0))
MIN_VISIBLE_MARKERS = 3
MAX_ARUCO_RMS_PX = 3.0
COLLECTOR_MAX_ARUCO_RMS_PX = 5.0  # Projector collection only; tighten here later.

# --- Perception: part detection and placement validation ---------------------
# Consecutive still frames required before anything is measured. Raise it if
# the user's hand is being validated instead of the part.
PERCEPTION_STILL_FRAMES = 6
# How close a placement must be to count as correct. These are acceptance
# limits for the demo, not a statement of measured accuracy.
PLACEMENT_TOLERANCE_MM = 3.0
PLACEMENT_TOLERANCE_DEG = 8.0

CHESSBOARD_INNER_CORNERS = (9, 6)
CHESSBOARD_SQUARE_MM = 25.0  # MEASURE your print before calibration
CHESSBOARD_TARGET_FRAMES = 30

PROJECTOR_MONITOR_INDEX = None  # auto only if exactly one non-primary monitor
PROJECTOR_SIZE = (1920, 1080)
DOT_RADIUS_PX = 28
DOT_COLOR = "green"
MIN_GREEN_INCREASE = 35
MIN_GREEN_DOMINANCE = 20
MIN_DOT_AREA_PX = 12
MAX_DOT_AREA_PX = 30000
SETTLE_SECONDS = 0.5
# Inclusive projector-pixel bounds for calibration collection.
GRID_U_RANGE_PX = (520, 1160)
GRID_V_RANGE_PX = (470, 610)
GRID_COLS = 5
GRID_ROWS = 4
MAX_BOARD_DRIFT_PX = 1.5
# Starting acceptance limits, not a guarantee of physical accuracy.
MAX_PROJECTOR_RMS_PX = 4.0  # Hackathon demo tolerance; verify physical landing error.
MAX_RIG_ROTATION_SPREAD_DEG = 2.0
MAX_RIG_TRANSLATION_SPREAD_MM = 15.0  # Demo tolerance for per-pose transform consistency.


def validate_fixture(require_bounds=True):
    import numpy as np
    if MARKER_SIZE_MM is None or not np.isfinite(MARKER_SIZE_MM) or MARKER_SIZE_MM <= 0:
        raise ValueError("Set measured MARKER_SIZE_MM in config.py first.")
    if len(MARKER_CENTERS_MM) < MIN_VISIBLE_MARKERS:
        raise ValueError("Set measured MARKER_CENTERS_MM in config.py first.")
    for marker_id, center in MARKER_CENTERS_MM.items():
        if not isinstance(marker_id, int) or not 0 <= marker_id < 50:
            raise ValueError("Marker IDs must be integers in DICT_4X4_50 (0..49).")
        if np.shape(center) != (2,) or not np.isfinite(center).all():
            raise ValueError(f"Invalid center for marker {marker_id}.")
    if BOARD_BOUNDS_MM is None and not require_bounds:
        return
    if BOARD_BOUNDS_MM is None or len(BOARD_BOUNDS_MM) != 4:
        raise ValueError("Set BOARD_BOUNDS_MM to the measured flat calibration surface.")
    x0, y0, x1, y1 = BOARD_BOUNDS_MM
    if not np.isfinite(BOARD_BOUNDS_MM).all() or x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid BOARD_BOUNDS_MM.")
