"""Live perception demo: place a part, see where it landed in board millimetres.

    python main.py perceive

Needs, like every board-frame tool here:
  * camera_calibration_1080p.npz  (python main.py camera-calibration)
  * MARKER_CENTERS_MM and BOARD_BOUNDS_MM measured in config.py

The assembly steps below are MOCK CAD DATA. Replace them with the Fusion side's
output; `Step.from_dict` already accepts that shape.

Keys: SPACE re-arms the baseline for this step, n skips a step, q/Esc quits.
"""
import argparse
import json

import cv2
import numpy as np

import config
from assembly.state_machine import AssemblyState, Phase, Step
from calibration.common import load_camera, run_cli
from hardware.camera import Camera, show_preview
from perception.aruco import BoardTracker
from perception.validator import correction_text

# Mock CAD input. Targets are the part silhouette's AREA CENTROID, in board mm.
MOCK_STEPS = [
    {"part_id": "red_l_plate", "target_pose": {"x_mm": 125.0, "y_mm": 95.0, "theta_deg": 0.0}},
    {"part_id": "blue_2x4", "target_pose": {"x_mm": 60.0, "y_mm": 140.0, "theta_deg": 90.0}},
]

COLOUR = {Phase.WAITING: (200, 200, 200), Phase.SETTLING: (120, 120, 120),
          Phase.CORRECTING: (0, 165, 255), Phase.WRONG_PART: (0, 0, 255),
          Phase.UNCLEAR: (0, 140, 200), Phase.STEP_COMPLETE: (0, 220, 0),
          Phase.COMPLETE: (0, 220, 0)}


def annotate(view, update, tracker, board_pose, total_steps):
    """Draw onto the UNDISTORTED frame, before show_preview rotates it."""
    colour = COLOUR[update.phase]
    result = update.result

    if board_pose is not None:
        cv2.drawFrameAxes(view, tracker.K, np.zeros(5), board_pose.rvec, board_pose.tvec, 50)

    if result is not None and result.region is not None:
        x, y, w, h = result.region.bbox
        cv2.rectangle(view, (x, y), (x + w, y + h), colour, 2)

    if result is not None and result.pose_px is not None and result.rect is not None:
        # The pose is measured in the rectified board plane; map it back to the
        # image to draw it, rather than pretending rectified pixels are image ones.
        rect, pose = result.rect, result.pose_px
        centre_mm = rect.to_board(pose.x_px, pose.y_px)
        theta = np.radians(rect.theta_to_board(pose.theta_deg))
        tip_mm = centre_mm + 25.0 * np.array([np.cos(theta), np.sin(theta)])
        points = cv2.projectPoints(
            np.array([[*centre_mm, rect.z_mm], [*tip_mm, rect.z_mm]]),
            board_pose.rvec, board_pose.tvec, tracker.K, np.zeros(5))[0].reshape(-1, 2)
        centre, tip = points[0].astype(int), points[1].astype(int)
        cv2.circle(view, tuple(centre), 5, colour, -1)
        cv2.arrowedLine(view, tuple(centre), tuple(tip), colour, 2, tipLength=0.25)

    lines = [f"[{update.phase.value}] {update.guidance}"]
    if update.step is not None:
        target = update.step.target_pose
        lines.append(f"step {update.step_index + 1}/{total_steps}  expect {update.step.part_id}"
                     f"  target x={target['x_mm']:.1f} y={target['y_mm']:.1f} "
                     f"th={target['theta_deg']:.1f}")
    if result is not None and result.observed_pose:
        observed = result.observed_pose
        lines.append(f"observed  x={observed['x_mm']:.1f} y={observed['y_mm']:.1f} "
                     f"th={observed['theta_deg']:.1f}  conf {result.confidence:.2f}")
    for i, line in enumerate(lines):
        cv2.putText(view, line, (20, 80 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    return view


def report(update, total_steps):
    """The terminal report: expected vs observed vs correction."""
    result = update.result
    print(f"\n--- step {update.step_index + 1}/{total_steps}: expect "
          f"{update.step.part_id} --- {update.phase.value}")
    if result is None or not result.detected:
        print(f"  {update.guidance}")
        return
    if not result.correct_part:
        print(f"  {result.message}")
        return

    target, observed = update.step.target_pose, result.observed_pose
    print(f"  expected   x={target['x_mm']:8.1f}  y={target['y_mm']:8.1f}  "
          f"theta={target['theta_deg']:7.1f}")
    print(f"  observed   x={observed['x_mm']:8.1f}  y={observed['y_mm']:8.1f}  "
          f"theta={observed['theta_deg']:7.1f}   confidence {result.confidence:.2f}")
    if result.error is not None:
        print(f"  correction {correction_text(result.error)}")


def main():
    parser = argparse.ArgumentParser(description="Live expected-part detection and validation.")
    parser.add_argument("--steps", help="JSON file of assembly steps; default is mock CAD data.")
    parser.add_argument("--change-only", action="store_true", help="Test change detection without part matching.")
    parser.add_argument("--cad", help="Fusion_output folder for manual V-key STL verification.")
    parser.add_argument("--part-id", help="Expected CAD component name for --cad.")
    parser.add_argument("--anchor", action="store_true", help="Fit first silver plate pose and test projected registration.")
    parser.add_argument('--base-marker-id',type=int,help='Enable moving-base assembly (e.g. 5)')
    parser.add_argument('--base-marker-size',type=float,help='Moving marker black-square side in mm')
    parser.add_argument("--tol-mm", type=float, default=getattr(config, "PLACEMENT_TOLERANCE_MM", 3.0))
    parser.add_argument("--tol-deg", type=float, default=getattr(config, "PLACEMENT_TOLERANCE_DEG", 8.0))
    args = parser.parse_args()

    if args.change_only and args.cad:
        parser.error('--change-only and --cad are separate test modes')
    if args.cad and not args.part_id:
        parser.error('--cad requires --part-id')
    if args.anchor and (not args.cad or args.part_id != 'Plate 1x10 Silver'):
        parser.error('--anchor requires --cad and --part-id "Plate 1x10 Silver"')
    if args.base_marker_id is not None and (not args.anchor or args.base_marker_size is None):
        parser.error('--base-marker-id requires --anchor and --base-marker-size')
    if args.base_marker_size is not None and args.base_marker_id is None:
        parser.error('--base-marker-size requires --base-marker-id')
    if args.change_only or args.cad:
        from perception.demo_change import run
        run(cad=args.cad, part_id=args.part_id, anchor=args.anchor,
            base_marker_id=args.base_marker_id,base_marker_size=args.base_marker_size)
        return

    steps = json.loads(open(args.steps).read()) if args.steps else MOCK_STEPS
    machine = AssemblyState([Step.from_dict(s) for s in steps],
                            tol_mm=args.tol_mm, tol_deg=args.tol_deg)

    K, dist = load_camera()
    tracker = BoardTracker(K, dist)
    print(f"{len(steps)} steps. Keep the workspace clear, then place one part at a time.")
    print("SPACE re-arms the baseline for this step, n skips, q quits.\n")

    last_phase = None
    with Camera() as camera:
        while True:
            raw = camera.read()                      # geometry always uses RAW
            board_pose = tracker.estimate(raw)

            if board_pose is None:
                show_preview(raw, "No reliable board pose (need >=3 known markers)")
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                continue

            update = machine.update(raw, board_pose, K, dist)
            if update.phase is not last_phase and update.phase is not Phase.SETTLING:
                if update.step is not None:
                    report(update, len(steps))
                last_phase = update.phase

            view = annotate(cv2.undistort(raw, K, dist), update, tracker, board_pose, len(steps))
            show_preview(view, "")

            key = cv2.waitKey(1) & 0xFF
            if machine.is_complete:
                print("\nAssembly complete.")
                cv2.waitKey(1500)
                break
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                machine.set_baseline(raw)
                last_phase = None
                print("baseline re-armed")
            if key == ord("n"):
                machine.advance()
                machine.set_baseline(raw)
                last_phase = None
                print(f"skipped to step {machine.index + 1}")


if __name__ == "__main__":
    run_cli(main)
