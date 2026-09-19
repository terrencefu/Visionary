"""Two frames in, a correction out. The perception subsystem's front door.

    result = detect_and_validate(frame_before, frame_after, expected_part,
                                 board_pose, camera_matrix, dist_coeffs)

Everything downstream of the change detection runs on UNDISTORTED frames, in
the same camera matrix K. Measured on the current 1080p calibration, this lens
displaces pixels by up to 104 px (88 px at mid-edge, 74 px at the corners) and
skews local angles with them; that error cannot be removed from a centroid or
an angle afterwards, so it is removed from the image first. Board geometry is
then done with the undistorted-pixel variant, never by undistorting twice.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from perception import part_catalog
from perception.change_detector import detect_change
from perception.geometry import board_point_from_undistorted_pixel, camera_pixel_to_board
from perception.part_matcher import HsvOutlineMatcher
from perception.pose_estimator import board_scale_px_per_mm, planar_pose_to_board
from perception.validator import TOLERANCE_DEG, TOLERANCE_MM, compare

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
    pose_px: object = field(default=None, repr=False)

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
                        undistort=True, already_undistorted=False):
    """Did `expected_part` appear between the two frames, where, and is it right?

    `board_pose` is a perception.aruco.BoardPose (anything with rvec/tvec).
    `expected_pose` is the CAD target; omit it to only observe, not judge.
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

    # The caller may hand in frames it has already undistorted -- the state
    # machine does, so an unchanged baseline is not re-undistorted every frame.
    if already_undistorted:
        before, after, working_dist, undistort = frame_before, frame_after, np.zeros(5), True
    elif undistort:
        before = cv2.undistort(frame_before, K, dist)
        after = cv2.undistort(frame_after, K, dist)
        working_dist = np.zeros(5)
    else:
        before, after = frame_before, frame_after
        working_dist = dist

    region = detect_change(before, after, exclude_quads=marker_quads(board_pose, K, dist))
    if region is None:
        return _nothing("no_change", "no plausible new part in the changed region")

    centre = (board_point_from_undistorted_pixel(*region.centroid_px, rvec, tvec, K)
              if undistort else
              camera_pixel_to_board(*region.centroid_px, rvec, tvec, K, dist))
    px_per_mm = board_scale_px_per_mm(centre[:2], rvec, tvec, K, working_dist,
                                      undistorted=undistort)

    match = matcher.verify(after, region, expected_part, px_per_mm)
    if match.status == "wrong_part":
        return PerceptionResult(True, False, match.part_id, None, match.score,
                                "wrong_part", match.message, region=region)
    if not match.ok:
        return _nothing(match.status, match.message, region=region)

    world = planar_pose_to_board(match.pose_px, rvec, tvec, K, working_dist,
                                 undistorted=undistort)
    result = PerceptionResult(True, True, expected_part, world.as_dict(), match.score,
                              "ok", match.message, region=region, pose_px=match.pose_px)

    if expected_pose is None:
        return result

    if symmetry_deg is None:
        symmetry_deg = getattr(part_catalog.CATALOG.get(expected_part), "symmetry_deg", 360)
    result.error = compare(world, expected_pose, tol_mm=tol_mm, tol_deg=tol_deg,
                           symmetry_deg=symmetry_deg)
    result.status = result.error.status
    return result
