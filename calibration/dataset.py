import json
import cv2
import numpy as np
import config
from calibration.common import camera_signature, fixture_signature
from perception.geometry import pose_matrix


def load_poses(directory):
    files = sorted(directory.glob("pose_*.json"))
    if not files:
        raise ValueError(f"No pose_*.json files in {directory}.")
    poses = []
    expected_fixture, expected_camera = fixture_signature(), camera_signature()
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta, rows = data["metadata"], data["points"]
        if (meta["fixture_signature"] != expected_fixture or meta["camera_signature"] != expected_camera
                or tuple(meta["projector_size"]) != config.PROJECTOR_SIZE
                or tuple(meta["camera_size"]) != config.CAMERA_SIZE or not meta["rigid_mount_ready"]):
            raise ValueError(f"{path.name}: incompatible fixture, calibration, resolution, or non-rigid mount.")
        obj = np.array([r["board_xyz"] for r in rows], np.float32)
        uv = np.array([r["projector_uv"] for r in rows], np.float32)
        cam_uv = np.array([r["camera_uv"] for r in rows], np.float32)
        if len(rows) < 8 or obj.shape != (len(rows), 3) or uv.shape != (len(rows), 2):
            raise ValueError(f"{path.name}: insufficient/malformed correspondences.")
        if not all(np.isfinite(a).all() for a in (obj, uv, cam_uv)) or not np.allclose(obj[:, 2], 0):
            raise ValueError(f"{path.name}: invalid plane data.")
        if np.linalg.matrix_rank(obj[:, :2]-obj[:, :2].mean(axis=0)) < 2:
            raise ValueError(f"{path.name}: collinear board points.")
        transforms = [pose_matrix(r["rvec"], r["tvec"]) for r in rows]
        if not all(np.isfinite(t).all() for t in transforms):
            raise ValueError(f"{path.name}: invalid camera pose.")
        poses.append(dict(name=path.name, objects=obj, projector_uv=uv, camera_uv=cam_uv,
                          transforms=transforms, rows=rows))
    return poses


def mean_transform(transforms):
    rotations = np.array([t[:3, :3] for t in transforms])
    U, _, Vt = np.linalg.svd(rotations.mean(axis=0))
    correction = np.diag([1, 1, np.linalg.det(U @ Vt)])
    result = np.eye(4)
    result[:3, :3] = U @ correction @ Vt
    result[:3, 3] = np.mean([t[:3, 3] for t in transforms], axis=0)
    return result


def rotation_difference_deg(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a @ b.T)-1)/2, -1, 1))))
