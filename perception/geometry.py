import cv2
import numpy as np


def pose_matrix(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))[0]
    T[:3, 3] = np.asarray(tvec).reshape(3)
    return T


def camera_pixel_to_board(u, v, rvec, tvec, K, dist):
    """Intersect a RAW distorted camera pixel with board z=0. Output is mm."""
    xy = cv2.undistortPoints(np.array([[[u, v]]], dtype=float), K, dist).reshape(2)
    return _board_intersection(xy, rvec, tvec)


def board_point_from_undistorted_pixel(u, v, rvec, tvec, K):
    """Same intersection, for a pixel that is ALREADY undistorted in the same K.

    Part detection measures silhouettes on undistorted frames -- lens distortion
    bends an outline before its angle is measured, and that cannot be undone
    afterwards. Feeding those pixels to camera_pixel_to_board would undistort
    them a second time.
    """
    K = np.asarray(K, dtype=float)
    xy = np.linalg.solve(K, np.array([float(u), float(v), 1.0]))[:2]
    return _board_intersection(xy, rvec, tvec)


def _board_intersection(normalised_xy, rvec, tvec):
    """Intersect a normalised camera ray with the board plane z=0. Output is mm."""
    xy = np.asarray(normalised_xy, dtype=float).reshape(2)
    R = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))[0]
    origin = -R.T @ np.asarray(tvec, dtype=float).reshape(3)
    ray = R.T @ np.array([xy[0], xy[1], 1.0])
    if abs(ray[2]) < 1e-9:
        raise ValueError("Camera ray is parallel to board plane.")
    distance = -origin[2] / ray[2]
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Board intersection is behind the camera or invalid.")
    result = origin + distance * ray
    result[2] = 0.0
    return result
