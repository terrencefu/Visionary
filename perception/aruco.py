from dataclasses import dataclass
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
    def __init__(self, K, dist):
        config.validate_fixture(require_bounds=False)
        self.K, self.dist = K, dist
        self.detector = make_detector()

    def estimate(self, raw):
        corners, ids, _ = self.detector.detectMarkers(raw)
        if ids is None:
            return None
        known = [(int(i), c.reshape(4, 2)) for i, c in zip(ids.flatten(), corners)
                 if int(i) in config.MARKER_CENTERS_MM]
        if len(known) < config.MIN_VISIBLE_MARKERS or len({i for i, _ in known}) != len(known):
            return None
        known.sort(key=lambda item: item[0])
        obj = np.concatenate([marker_corners(i) for i, _ in known])
        img = np.concatenate([c for _, c in known]).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        R = cv2.Rodrigues(rvec)[0]
        projected = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)[0].reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum((projected - img)**2, axis=1))))
        if not np.isfinite(rms) or rms > config.MAX_ARUCO_RMS_PX or np.any((obj @ R.T + tvec.reshape(3))[:, 2] <= 0):
            return None
        return BoardPose(rvec, tvec, R, [i for i, _ in known], rms, obj, img)


def board_drift_px(reference, current, K, dist):
    if current is None:
        return float("inf")
    a = cv2.projectPoints(reference.object_points, reference.rvec, reference.tvec, K, dist)[0]
    b = cv2.projectPoints(reference.object_points, current.rvec, current.tvec, K, dist)[0]
    return float(np.max(np.linalg.norm((a-b).reshape(-1, 2), axis=1)))
