"""One checked black/dot capture, shared by collection and physical validation."""
import cv2
import numpy as np
import config
from calibration.common import inside_board
from perception.aruco import board_drift_px
from perception.geometry import camera_pixel_to_board
from perception.green_dot import detect_green_dot


def measure_dot(camera, projector, tracker, u, v, reference=None):
    projector.black()
    background = camera.settled_frame()
    before = tracker.estimate(background)
    if before is None:
        raise ValueError("No reliable board pose on black background.")
    if reference is not None and board_drift_px(reference, before, tracker.K, tracker.dist) > config.MAX_BOARD_DRIFT_PX:
        raise ValueError("Board/rig moved during this pose; start a new pose.")
    rendered_uv = projector.dot(u, v)
    raw = camera.settled_frame()
    try:
        after = tracker.estimate(raw)
        if board_drift_px(before, after, tracker.K, tracker.dist) > config.MAX_BOARD_DRIFT_PX:
            raise ValueError("Board/rig moved or markers became unreliable during dot capture.")
        dot, _ = detect_green_dot(background, raw)
        if dot is None:
            raise ValueError("No unambiguous green dot.")
        point = camera_pixel_to_board(dot.u, dot.v, before.rvec, before.tvec, tracker.K, tracker.dist)
        if not inside_board(point):
            raise ValueError("Dot center is outside the configured flat board region.")
        # Reject footprints crossing board edges (a biased centroid at two depths).
        for pixel in dot.contour.reshape(-1, 2):
            edge = camera_pixel_to_board(*pixel, before.rvec, before.tvec, tracker.K, tracker.dist)
            if not inside_board(edge):
                raise ValueError("Dot footprint crosses the configured board boundary.")
        # Avoid calibration dots hitting black marker ink, which biases their centroids.
        blob = np.zeros(raw.shape[:2], np.uint8)
        cv2.drawContours(blob, [dot.contour], -1, 255, -1)
        for marker in before.image_points.reshape(-1, 4, 2):
            region = np.zeros(raw.shape[:2], np.uint8)
            cv2.fillConvexPoly(region, np.rint(marker).astype(np.int32), 255)
            if np.any(cv2.bitwise_and(region, blob)):
                raise ValueError("Dot overlaps marker ink; choose another grid position.")
        cv2.circle(raw, (round(dot.u), round(dot.v)), 12, (0, 0, 255), 2)
        return dict(projector_uv=rendered_uv, camera_uv=[dot.u, dot.v],
                    board_xyz=point.tolist(), rvec=before.rvec.ravel().tolist(),
                    tvec=before.tvec.ravel().tolist(), aruco_rms=before.reprojection_error,
                    visible_ids=before.visible_ids), raw, before
    finally:
        projector.black()
