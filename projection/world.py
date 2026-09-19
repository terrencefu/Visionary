import cv2
import numpy as np
import config
from calibration.common import camera_signature, fixture_signature
from perception.geometry import pose_matrix


def load_projector():
    if not config.PROJECTOR_CALIBRATION.exists():
        raise FileNotFoundError("Missing projector_calibration.npz. Collect rigid-mount poses and solve first.")
    with np.load(config.PROJECTOR_CALIBRATION, allow_pickle=False) as data:
        if (tuple(data["image_size"]) != config.PROJECTOR_SIZE
                or str(data["camera_signature"]) != camera_signature()
                or str(data["fixture_signature"]) != fixture_signature()
                or not bool(data["quality_passed"])):
            raise ValueError("Projector calibration does not match current camera, fixture, resolution, or quality gate.")
        return data["camera_matrix"].copy(), data["dist_coeffs"].copy(), data["T_projector_camera"].copy()


def board_point_to_projector(point, rvec, tvec, K_projector, dist_projector, T_projector_camera):
    T_pb = T_projector_camera @ pose_matrix(rvec, tvec)
    point = np.asarray(point, dtype=float).reshape(1, 3)
    depth = (point @ T_pb[:3, :3].T + T_pb[:3, 3])[0, 2]
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError("Target is behind the projector or invalid.")
    result = cv2.projectPoints(point, cv2.Rodrigues(T_pb[:3, :3])[0],
                              T_pb[:3, 3], K_projector, dist_projector)[0].reshape(2)
    return result
