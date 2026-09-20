"""Two frames in, a correction out. The perception subsystem's front door.

    result = detect_and_validate(frame_before, frame_after, expected_part,
                                 board_pose, camera_matrix, dist_coeffs)

Two coordinate rules carry the whole design:

**Undistort first.** Measured on the current 1080p calibration this lens
displaces pixels by up to 104 px and skews local angles with them. That error
cannot be removed from a centroid or an angle afterwards, so it is removed from
the image first.

**Then rectify.** The camera views the workspace from 30-45 degrees, so the raw
image foreshortens the board plane, varies in scale across it, and shows board
+Y upward while part_catalog draws it downward -- meaning a raw-space template
match compares a part against its own mirror. Matching therefore happens in a
metric top-down rectification of the board plane, where board +X is right,
board +Y is down, and px/mm is exact. See perception/rectify.py.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from perception import part_catalog
from perception.change_detector import detect_change
from perception.geometry import board_point_from_undistorted_pixel, camera_height_above_board
from perception.part_matcher import HsvOutlineMatcher
from perception.pose_estimator import BoardPlanarPose
from perception.rectify import BoardRectifier
from perception.validator import TOLERANCE_DEG, TOLERANCE_MM, compare

REGION_PAD_MM = 14.0

_DEFAULT_MATCHER = None


def default_matcher():
    global _DEFAULT_MATCHER
    if _DEFAULT_MATCHER is None:
        _DEFAULT_MATCHER = HsvOutlineMatcher()
    return _DEFAULT_MATCHER


@dataclass
class PerceptionResult:
    detected: bool
    correct_part: bool
    part_id: str | None
    observed_pose: dict | None
    confidence: float
    status: str          # ok | no_change | not_found | wrong_part | low_confidence
                         #    | wrong_position | wrong_angle | wrong_position_and_angle
    message: str
    error: object = None          # validator.PoseError, when a target was given
    region: object = field(default=None, repr=False)
    pose_px: object = field(default=None, repr=False)   # in RECTIFIED pixels
    rect: object = field(default=None, repr=False)      # the BoardRectifier used

    def as_dict(self):
        """The shape agreed with the pipeline owner in the README."""
        return {"detected": self.detected, "correct_part": self.correct_part,
                "part_id": self.part_id, "observed_pose": self.observed_pose,
                "confidence": self.confidence}


def _nothing(status, message, **kwargs):
    return PerceptionResult(False, False, None, None, 0.0, status, message, **kwargs)


def marker_quads(board_pose, K, dist):
    """Marker corner quads as UNDISTORTED pixels, so they can be excluded from
    the change search. Markers change appearance under projected light and must
    never become candidate parts."""
    points = getattr(board_pose, "image_points", None)
    if points is None:
        return []
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) < 4:
        return []
    undistorted = cv2.undistortPoints(points.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    return [undistorted[i:i + 4] for i in range(0, len(undistorted) - 3, 4)]


def detect_and_validate(frame_before, frame_after, expected_part, board_pose,
                        camera_matrix, dist_coeffs, expected_pose=None, matcher=None,
                        tol_mm=TOLERANCE_MM, tol_deg=TOLERANCE_DEG, symmetry_deg=None,
                        z_mm=0.0, already_undistorted=False, rect_px_per_mm=None):
    """Did `expected_part` appear between the two frames, where, and is it right?

    `board_pose` is a perception.aruco.BoardPose (anything with rvec/tvec).
    `expected_pose` is the CAD target; omit it to observe without judging.
    `z_mm` is the height of the plane the part rests on -- 0 for the table, or
    the layer height for a part placed on earlier ones. Measuring a raised part
    on the z=0 plane gives a systematic outward error of about r*z/(H-z).
    """
    if frame_before is None or frame_after is None:
        raise ValueError("Both a before and an after frame are required.")
    if frame_before.shape != frame_after.shape:
        raise ValueError(f"Frame sizes differ: {frame_before.shape} vs {frame_after.shape}.")
    if board_pose is None:
        raise ValueError("A board pose is required; perception cannot report board mm without it.")

    rvec, tvec = board_pose.rvec, board_pose.tvec
    K = np.asarray(camera_matrix, float)
    dist = np.asarray(dist_coeffs, float)
    matcher = matcher if matcher is not None else default_matcher()

    height = camera_height_above_board(rvec, tvec)
    if height <= 0:
        raise ValueError(
            f"Board pose puts the camera {height:.0f} mm BELOW the board plane. "
            "Board +z must point toward the camera; every silhouette is mirrored otherwise.")

    if already_undistorted:
        before, after = frame_before, frame_after
    else:
        before = cv2.undistort(frame_before, K, dist)
        after = cv2.undistort(frame_after, K, dist)

    region = detect_change(before, after, exclude_quads=marker_quads(board_pose, K, dist))
    if region is None:
        return _nothing("no_change", "no plausible new part in the changed region")

    # Match in a metric top-down view of the plane the part rests on.
    rect = BoardRectifier.for_region(rvec, tvec, K, region.bbox, pad_mm=REGION_PAD_MM,
                                     px_per_mm=rect_px_per_mm, z_mm=z_mm)
    patch = rect.warp(after)

    match = matcher.verify(patch, None, expected_part, rect.px_per_mm)
    if match.status == "wrong_part":
        return PerceptionResult(True, False, match.part_id, None, match.score,
                                "wrong_part", match.message, region=region, rect=rect)
    if not match.ok:
        return _nothing(match.status, match.message, region=region, rect=rect)

    # Rectified pixels ARE board millimetres, up to the scale we chose.
    x_mm, y_mm = rect.to_board(match.pose_px.x_px, match.pose_px.y_px)
    world = BoardPlanarPose(float(x_mm), float(y_mm),
                            rect.theta_to_board(match.pose_px.theta_deg),
                            float(z_mm), match.score)

    result = PerceptionResult(True, True, expected_part, world.as_dict(), match.score,
                              "ok", match.message, region=region,
                              pose_px=match.pose_px, rect=rect)
    if expected_pose is None:
        return result

    if symmetry_deg is None:
        symmetry_deg = getattr(part_catalog.CATALOG.get(expected_part), "symmetry_deg", 360)
    result.error = compare(world, expected_pose, tol_mm=tol_mm, tol_deg=tol_deg,
                           symmetry_deg=symmetry_deg)
    result.status = result.error.status
    return result
