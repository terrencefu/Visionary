"""Conservative colour history reprojection for a rigid, flat CAD assembly.

Frames are already undistorted. Sweep CAD heights rather than assuming coloured
surfaces lie on the board. This can reject ambiguous additions near old pieces;
it must not invent a new part merely because the camera/base moved.
"""
import cv2
import numpy as np
from perception.geometry import pose_matrix
from perception.colour_change import colour_mask


def plane_projection(camera_from_cad, K, z):
    projection = K @ camera_from_cad[:3]
    return np.column_stack((projection[:, 0], projection[:, 1],
                            projection[:, 3] + z * projection[:, 2]))


def aligned_colour_history(before, colour, before_pose, current_pose, K,
                           before_transform, current_transform, meshes):
    old_camera = pose_matrix(before_pose.rvec, before_pose.tvec) @ before_transform
    new_camera = pose_matrix(current_pose.rvec, current_pose.tvec) @ current_transform
    points = np.concatenate(meshes).reshape(-1, 3)
    for camera in (old_camera, new_camera):
        depths = points @ camera[2, :3] + camera[2, 3]
        if not np.isfinite(camera).all() or np.any(depths <= 0):
            raise ValueError('Assembly is behind camera; reacquire markers')
    low, high = points[:, 2].min(), points[:, 2].max()
    source = colour_mask(before, colour)
    h, w = source.shape
    history = np.zeros_like(source)
    visible = np.full_like(source, 255)
    full = visible.copy()
    # 1 mm layers include vertical sides and different installed part heights.
    for z in np.linspace(low, high, max(2, int(np.ceil(high-low))+1)):
        old = plane_projection(old_camera, K, z)
        new = plane_projection(new_camera, K, z)
        if np.linalg.cond(old) > 1e12 or np.linalg.cond(new) > 1e12:
            raise ValueError('View too oblique for placement comparison')
        H = new @ np.linalg.inv(old)
        history |= cv2.warpPerspective(source, H, (w,h), flags=cv2.INTER_NEAREST)
        visible &= cv2.warpPerspective(full, H, (w,h), flags=cv2.INTER_NEAREST)
    # Do not treat newly revealed image borders as an observed addition.
    visible = cv2.erode(visible, np.ones((5,5),np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return history, visible
