"""Read-only fixture and pose diagnostics. Never feed audit results into solvePnP."""
from dataclasses import asdict
import json
from pathlib import Path
import cv2
import numpy as np
import config
from hardware.camera import preview_frame


def fixture_lines():
    lines = [f"Config: {config.__file__}", f"Calibration: {config.CAMERA_CALIBRATION}",
             f"Dictionary: {config.ARUCO_DICTIONARY}; black-square side: {config.MARKER_SIZE_MM} mm",
             "Model rotation: 0 deg for EVERY marker; canonical corner 0 at (-half,-half).",
             "All printed top edges must point along board +X (marker 0 toward marker 1)."]
    for i, xy in sorted(config.MARKER_CENTERS_MM.items()):
        lines.append(f"ID {i}: center XY={xy} mm, side={config.MARKER_SIZE_MM} mm, rotation=0 deg")
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3), (1, 2)]:
        if a in config.MARKER_CENTERS_MM and b in config.MARKER_CENTERS_MM:
            distance = np.linalg.norm(np.array(config.MARKER_CENTERS_MM[a])-config.MARKER_CENTERS_MM[b])
            lines.append(f"Configured center distance {a}->{b}: {distance:.3f} mm")
    lines += ["Uses configured fixture coordinates as supplied; no automatic geometry adjustment.",
              "A camera image cannot independently prove tape-measure dimensions or flatness.",
              "Side/rotation audit below is conditional on the configured centers and camera calibration."]
    return lines


def audit_marker_geometry(debug, K, dist):
    """Use a center-only planar homography to check canonical side directions.

    Unlike the rejected pose, this diagnostic does not fit marker corners to the
    unrotated-square model. It can therefore expose a 90/180-degree printed-marker
    rotation. It still depends on the measured centers, flatness and intrinsics.
    """
    known = [(i, np.asarray(c).reshape(4, 2)) for i, c in zip(debug.detected_ids, debug.corners)
             if i in config.MARKER_CENTERS_MM]
    if len(known) < 4 or len({i for i, _ in known}) != len(known):
        return {}, "Side/rotation audit needs >=4 unique configured markers."
    try:
        quads = [cv2.undistortPoints(c.astype(float).reshape(-1, 1, 2), K, dist,
                                    P=K).reshape(4, 2) for _, c in known]
        centers = []
        for q in quads:
            homogeneous = np.c_[q, np.ones(4)]
            # The mean of four perspective-projected corners is NOT the square center.
            intersection = np.cross(np.cross(homogeneous[0], homogeneous[2]),
                                    np.cross(homogeneous[1], homogeneous[3]))
            if abs(intersection[2]) < 1e-10:
                return {}, "Degenerate marker diagonals; side/rotation audit unavailable."
            centers.append(intersection[:2]/intersection[2])
        H, _ = cv2.findHomography(np.array(centers), np.array([
            config.MARKER_CENTERS_MM[i] for i, _ in known], dtype=float), method=0)
        if H is None or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
            return {}, "Degenerate marker-center layout; side/rotation audit unavailable."
        result = {}
        for (marker_id, _), quad in zip(known, quads):
            q = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), H).reshape(4, 2)
            edges = np.roll(q, -1, axis=0)-q
            lengths = np.linalg.norm(edges, axis=1)
            x_direction = (edges[0]-edges[2])/2
            angle = float(np.degrees(np.arctan2(x_direction[1], x_direction[0])))
            if not np.isfinite(lengths).all() or not np.isfinite(angle):
                return {}, "Non-finite side/rotation audit; check center geometry and calibration."
            result[marker_id] = dict(side_lengths_mm=lengths.tolist(),
                                     width_mm=float(np.mean(lengths[[0, 2]])),
                                     height_mm=float(np.mean(lengths[[1, 3]])),
                                     rotation_deg=angle)
        return result, "Audit only: expected rotation 0 deg, side " + str(config.MARKER_SIZE_MM) + " mm."
    except cv2.error as exc:
        return {}, "Side/rotation audit failed: " + str(exc).splitlines()[-1]


def diagnostic_lines(debug, audit=None, audit_note=""):
    number = lambda x: "n/a" if x is None else f"{x:.3f}"
    if not debug.solvepnp_attempted:
        pnp = "NOT ATTEMPTED"
    elif debug.solvepnp_succeeded is None:
        pnp = "ERROR (no success flag returned)"
    else:
        pnp = str(debug.solvepnp_succeeded)
    lines = [f"Detected IDs: {debug.detected_ids}",
             f"Configured matches: {debug.known_ids}; unknown IDs: {debug.unknown_ids}",
             f"Known detections: {debug.known_marker_count}; required: {debug.required_markers}",
             f"Used for solvePnP: {debug.used_marker_count} markers / {4*debug.used_marker_count} corners",
             f"solvePnP success: {pnp}",
             f"Reprojection mean: {number(debug.mean_error_px)} px; maximum: {number(debug.max_error_px)} px",
             f"Reprojection RMS: {number(debug.rms_error_px)} px; acceptable RMS <= {debug.max_acceptable_rms_px:.3f} px",
             "Gate uses RMS, not mean or maximum corner error.",
             "Rejection reason: " + (debug.rejection_reason or "NONE - pose accepted")]
    for i, errors in debug.per_marker_errors.items():
        lines.append(f"ID {i} residuals: mean={errors['mean']:.3f}, max={errors['max']:.3f}, RMS={errors['rms']:.3f} px")
    if audit_note:
        lines += ["", audit_note]
    for i, entry in sorted((audit or {}).items()):
        lines.append(f"ID {i} audit: rotation={entry['rotation_deg']:+.1f} deg, sides={entry['width_mm']:.1f} x {entry['height_mm']:.1f} mm")
    lines += ["", "Cyan arrow: detected canonical corner 0 -> 1 (printed +X).",
              "Magenta dots: model-projected corners; yellow lines: residuals.",
              "Geometry uses RAW frames. Only the human preview rotates."]
    return lines


def annotate(raw, debug):
    view = raw.copy()
    if debug.detected_ids:
        cv2.aruco.drawDetectedMarkers(view, debug.corners, np.array(debug.detected_ids, np.int32).reshape(-1, 1))
    for quad in debug.corners:
        q = np.rint(quad.reshape(4, 2)).astype(int)
        cv2.circle(view, tuple(q[0]), 5, (255, 255, 0), -1)
        cv2.arrowedLine(view, tuple(q[0]), tuple(q[1]), (255, 255, 0), 2, tipLength=0.3)
    if debug.projected_points is not None:
        for observed, predicted in zip(debug.image_points, debug.projected_points):
            if not np.isfinite(predicted).all() or np.max(np.abs(predicted)) > 1e7:
                continue
            start, end = tuple(np.rint(observed).astype(int)), tuple(np.rint(predicted).astype(int))
            cv2.line(view, start, end, (0, 255, 255), 1)
            cv2.circle(view, end, 3, (255, 0, 255), -1)
    return view


def save_debug(directory, raw, view, debug, audit, audit_note, comparison=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = asdict(debug)
    for field in ("corners", "projected_points", "object_points", "image_points"):
        value = report[field]
        report[field] = None if value is None else np.asarray(value).tolist()
    report.update(fixture=fixture_lines(), marker_audit=audit, marker_audit_note=audit_note,
                  camera_index=config.CAMERA_INDEX, camera_size=config.CAMERA_SIZE)
    if comparison is not None:
        report['pose_comparison'] = comparison
    # Strict JSON even for intentionally diagnosed NaN/Inf failure cases.
    def clean(value):
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value
    lines = diagnostic_lines(debug, audit, audit_note)
    frames = [("raw.png", raw)]
    if comparison is not None:
        from calibration.aruco_pose_diagnostics import comparison_lines, annotate_comparison
        comparison_text = comparison_lines(comparison)
        numbered_raw = annotate_comparison(raw, debug, comparison)
        frames += [("correspondence_raw.png", numbered_raw),
                   ("correspondence_panel_raw.png", preview_frame(numbered_raw, debug_lines=comparison_text, rotate=False))]
        lines += comparison_text
    frames.append(("preview.png", preview_frame(view, debug_lines=lines)))
    for filename, frame in frames:
        if not cv2.imwrite(str(directory / filename), frame):
            raise OSError(f"Could not write {directory / filename}")
    (directory / "report.json").write_text(json.dumps(clean(report), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved RAW frame, annotated preview, and report: {directory}")
