from dataclasses import dataclass, field
import cv2
import numpy as np
import config


@dataclass
class BoardPose:
    rvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    visible_ids: list
    reprojection_error: float
    object_points: np.ndarray
    image_points: np.ndarray

    @property
    def camera_position_board(self):
        return -self.R.T @ self.tvec.reshape(3)


@dataclass
class PoseDiagnostics:
    """Evidence from the latest RAW frame, including frames rejected by estimate()."""
    detected_ids: list = field(default_factory=list)
    known_ids: list = field(default_factory=list)
    unknown_ids: list = field(default_factory=list)
    known_marker_count: int = 0
    used_marker_count: int = 0
    required_markers: int = 0
    solvepnp_attempted: bool = False
    solvepnp_succeeded: bool | None = None
    mean_error_px: float | None = None
    max_error_px: float | None = None
    rms_error_px: float | None = None
    max_acceptable_rms_px: float = 0.0
    per_marker_errors: dict = field(default_factory=dict)
    rejection_code: str | None = None
    rejection_reason: str | None = None
    corners: list = field(default_factory=list, repr=False)
    projected_points: np.ndarray | None = field(default=None, repr=False)
    object_points: np.ndarray | None = field(default=None, repr=False)
    image_points: np.ndarray | None = field(default=None, repr=False)

    def reject(self, code, reason):
        self.rejection_code, self.rejection_reason = code, reason


def make_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.ARUCO_DICTIONARY))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def marker_corners(marker_id):
    cx, cy = config.MARKER_CENTERS_MM[marker_id]
    h = config.MARKER_SIZE_MM / 2
    return np.array([[cx-h, cy-h, 0], [cx+h, cy-h, 0],
                     [cx+h, cy+h, 0], [cx-h, cy+h, 0]], dtype=np.float64)


class BoardTracker:
    def __init__(self, K, dist, *, max_rms_px=None):
        config.validate_fixture(require_bounds=False)
        self.K, self.dist = K, dist
        self.max_rms_px = max_rms_px
        if max_rms_px is not None and (not np.isfinite(max_rms_px) or max_rms_px <= 0):
            raise ValueError("max_rms_px must be finite and positive.")
        self.detector = make_detector()
        self.last_diagnostics = None

    def estimate(self, raw):
        # The detector, correspondences, solvePnP method and acceptance gates are
        # unchanged. Diagnostics retain the evidence before each existing return.
        limit = config.MAX_ARUCO_RMS_PX if self.max_rms_px is None else self.max_rms_px
        debug = PoseDiagnostics(required_markers=config.MIN_VISIBLE_MARKERS,
                                max_acceptable_rms_px=limit)
        self.last_diagnostics = debug
        corners, ids, _ = self.detector.detectMarkers(raw)
        debug.corners = corners
        if ids is None:
            debug.reject("no_markers", "No markers detected; solvePnP not attempted.")
            return None
        debug.detected_ids = [int(i) for i in ids.flatten()]
        debug.known_ids = [i for i in debug.detected_ids if i in config.MARKER_CENTERS_MM]
        debug.unknown_ids = [i for i in debug.detected_ids if i not in config.MARKER_CENTERS_MM]
        known = [(int(i), c.reshape(4, 2)) for i, c in zip(ids.flatten(), corners)
                 if int(i) in config.MARKER_CENTERS_MM]
        debug.known_marker_count = len(known)
        if len(known) < config.MIN_VISIBLE_MARKERS or len({i for i, _ in known}) != len(known):
            reasons = []
            if len(known) < config.MIN_VISIBLE_MARKERS:
                reasons.append(f"Only {len(known)} known markers; need {config.MIN_VISIBLE_MARKERS}")
            duplicates = sorted({i for i in debug.known_ids if debug.known_ids.count(i) > 1})
            if duplicates:
                reasons.append(f"Duplicate configured marker IDs: {duplicates}")
            debug.reject("duplicate_ids" if duplicates else "insufficient_known_markers",
                         "; ".join(reasons) + "; solvePnP not attempted.")
            return None
        known.sort(key=lambda item: item[0])
        obj = np.concatenate([marker_corners(i) for i, _ in known])
        img = np.concatenate([c for _, c in known]).astype(np.float64)
        debug.object_points, debug.image_points = obj, img
        debug.used_marker_count = len(known)
        debug.solvepnp_attempted = True
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error as exc:
            debug.reject("solvepnp_exception", "solvePnP raised OpenCV error: " + str(exc))
            raise  # Preserve exception behavior for existing callers.
        debug.solvepnp_succeeded = bool(ok)
        if not ok:
            debug.reject("solvepnp_failed", "solvePnP returned success=False.")
            return None
        try:
            R = cv2.Rodrigues(rvec)[0]
            projected = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)[0].reshape(-1, 2)
        except cv2.error as exc:
            debug.reject("projection_exception", "Pose reprojection raised OpenCV error: " + str(exc))
            raise
        rms = float(np.sqrt(np.mean(np.sum((projected - img)**2, axis=1))))
        errors = np.linalg.norm(projected - img, axis=1)
        debug.projected_points = projected
        debug.rms_error_px = rms
        debug.mean_error_px = float(np.mean(errors))
        debug.max_error_px = float(np.max(errors))
        for (marker_id, _), marker_errors in zip(known, errors.reshape(-1, 4)):
            debug.per_marker_errors[marker_id] = dict(
                mean=float(np.mean(marker_errors)), max=float(np.max(marker_errors)),
                rms=float(np.sqrt(np.mean(marker_errors**2))), corners=marker_errors.tolist())
        depths = (obj @ R.T + tvec.reshape(3))[:, 2]
        behind = int(np.count_nonzero(~np.isfinite(depths) | (depths <= 0)))
        if not np.isfinite(rms) or rms > limit or behind:
            reasons = []
            if not np.isfinite(rms):
                reasons.append("Non-finite reprojection RMS")
            elif rms > limit:
                reasons.append(f"Reprojection RMS {rms:.3f} px exceeds limit {limit:.3f} px")
            if behind:
                reasons.append(f"{behind}/{len(obj)} marker corners have non-finite or non-positive depth")
            code = "nonfinite_reprojection" if not np.isfinite(rms) else (
                "reprojection_error" if rms > limit else "nonpositive_depth")
            debug.reject(code, "; ".join(reasons) + ".")
            return None
        return BoardPose(rvec, tvec, R, [i for i, _ in known], rms, obj, img)


def board_drift_px(reference, current, K, dist):
    if current is None:
        return float("inf")
    a = cv2.projectPoints(reference.object_points, reference.rvec, reference.tvec, K, dist)[0]
    b = cv2.projectPoints(reference.object_points, current.rvec, current.tvec, K, dist)[0]
    return float(np.max(np.linalg.norm((a-b).reshape(-1, 2), axis=1)))
