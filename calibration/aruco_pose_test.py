import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import config
from calibration.common import load_camera, run_cli
from calibration.aruco_debug import (annotate, audit_marker_geometry, diagnostic_lines,
                                     fixture_lines, save_debug)
from calibration.aruco_pose_diagnostics import (parse_rotations, compare_poses,
                                               comparison_lines, annotate_comparison)
from hardware.camera import Camera, show_preview
from perception.aruco import BoardTracker, PoseDiagnostics, make_detector


def main():
    parser = argparse.ArgumentParser(description="Detect markers first, then verify world pose.")
    parser.add_argument("--detect-only", action="store_true", help="No geometry or intrinsics needed.")
    parser.add_argument("--diagnose-poses", action="store_true",
                        help="Compare independent markers, all known markers, and diagnostic subset 0/1/3.")
    parser.add_argument("--marker-rotation", action="append", default=[], metavar="ID=DEGREES",
                        help="Declare actual printed rotation for DIAGNOSTIC fits only; repeat per marker. Default all 0.")
    parser.add_argument("--diagnostic-fit", choices=['subset', 'individual', 'all'], default='subset',
                        help="Fit used for numbered corners on the unrotated diagnostic view (default subset).")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames (0 = interactive).")
    parser.add_argument("--no-preview", action="store_true", help="Terminal only; requires --frames or --image.")
    parser.add_argument("--image", type=Path, help="Diagnose one saved RAW 1920x1080 camera image.")
    parser.add_argument("--save-debug", type=Path, help="Save last raw.png, preview.png and report.json to this folder.")
    parser.add_argument("--log-interval", type=float, default=1.0, help="Seconds between terminal reports (default 1).")
    args = parser.parse_args()
    if args.frames < 0 or not np.isfinite(args.log_interval) or args.log_interval <= 0:
        raise ValueError("--frames must be >=0 and --log-interval must be positive.")
    if args.no_preview and not args.frames and args.image is None:
        raise ValueError("Use --frames N or --image FILE with --no-preview.")
    if args.diagnose_poses and args.detect_only:
        raise ValueError("--diagnose-poses needs calibrated pose estimation; omit --detect-only.")
    if not args.diagnose_poses and (args.marker_rotation or args.diagnostic_fit != 'subset'):
        raise ValueError("Diagnostic rotation/fit options require --diagnose-poses.")
    rotations = parse_rotations(args.marker_rotation)
    tracker = None if args.detect_only else BoardTracker(*load_camera())
    detector = make_detector() if args.detect_only else None
    if tracker:
        print("\n".join(fixture_lines()))
        print(f"Pose gate unchanged: >= {config.MIN_VISIBLE_MARKERS} known markers, RMS <= {config.MAX_ARUCO_RMS_PX} px, positive depth.")
    print("Esc/Q: exit | S: save raw frame + annotated preview + diagnostic JSON. Terminal report every", args.log_interval, "s.")
    last_log = 0
    last_frame = None
    last_lines = None
    frame_number = 0

    def process(raw):
        nonlocal last_log, last_frame, last_lines, frame_number
        frame_number += 1
        if (raw.shape[1], raw.shape[0]) != config.CAMERA_SIZE:
            raise ValueError(f"Require RAW camera image at {config.CAMERA_SIZE}; do not resize or rotate it.")
        pose, audit, audit_note, comparison = None, {}, "", None
        if tracker:
            try:
                pose = tracker.estimate(raw)
            except cv2.error:
                # Show pose/projection exceptions on this diagnostic screen; all other
                # callers of estimate() still receive the original exception.
                if tracker.last_diagnostics.rejection_code not in {"solvepnp_exception", "projection_exception"}:
                    raise
            debug = tracker.last_diagnostics
            audit, audit_note = audit_marker_geometry(debug, tracker.K, tracker.dist)
            lines = diagnostic_lines(debug, audit, audit_note)
        else:
            corners, ids, _ = detector.detectMarkers(raw)
            detected = [] if ids is None else [int(i) for i in ids.flatten()]
            known = [i for i in detected if i in config.MARKER_CENTERS_MM]
            debug = PoseDiagnostics(detected_ids=detected, known_ids=known,
                                    unknown_ids=[i for i in detected if i not in config.MARKER_CENTERS_MM],
                                    known_marker_count=len(known), corners=corners,
                                    required_markers=config.MIN_VISIBLE_MARKERS,
                                    max_acceptable_rms_px=config.MAX_ARUCO_RMS_PX,
                                    rejection_code="detection_only", rejection_reason="Pose not evaluated: --detect-only.")
            lines = diagnostic_lines(debug)
        view = annotate(raw, debug)
        if pose is not None:
            cv2.drawFrameAxes(view, tracker.K, tracker.dist, pose.rvec, pose.tvec, 50)
            lines.append("Camera in board mm: " + str(pose.camera_position_board.round(1)))
        log_lines = lines.copy()
        if args.diagnose_poses:
            comparison = compare_poses(debug, tracker.K, tracker.dist, rotations, audit, args.diagnostic_fit)
            log_lines += comparison_lines(comparison, detailed=True)
        last_frame = (raw, view, debug, audit, audit_note, comparison)
        last_lines = log_lines
        now = time.monotonic()
        if frame_number == 1 or now-last_log >= args.log_interval or args.image is not None:
            print(f"\nFrame {frame_number}\n" + "\n".join(log_lines), flush=True)
            last_log = now
        if not args.no_preview:
            show_preview(view, debug_lines=lines)
            if comparison is not None:
                show_preview(annotate_comparison(raw, debug, comparison),
                             name="ArUco correspondences - RAW / UNROTATED",
                             debug_lines=comparison_lines(comparison), rotate=False)
            key = cv2.waitKey(0 if args.image else 1) & 0xFF
            if key == ord('s'):
                save_debug(args.save_debug or config.DATA_DIR / "aruco_debug", *last_frame)
            if key in (27, ord('q')):
                return False
        return True

    if args.image:
        raw = cv2.imread(str(args.image))
        if raw is None:
            raise ValueError(f"Cannot read {args.image}.")
        process(raw)
    else:
        with Camera() as camera:
            while process(camera.read()):
                if args.frames and frame_number >= args.frames:
                    break
    if last_frame is not None:
        if args.save_debug:
            save_debug(args.save_debug, *last_frame)
        if args.frames and not args.image:
            print(f"\nFinal frame {frame_number}\n" + "\n".join(last_lines))


if __name__ == "__main__":
    run_cli(main)
