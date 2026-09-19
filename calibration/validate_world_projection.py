import argparse
import csv
from datetime import datetime, timezone
import cv2
import numpy as np
import config
from calibration.common import load_camera, inside_board, run_cli
from calibration.measurement import measure_dot
from hardware.camera import Camera, show_preview
from hardware.projector import Projector
from perception.aruco import BoardTracker
from projection.world import load_projector, board_point_to_projector


def main():
    parser = argparse.ArgumentParser(description="Independently measure physical world-lock error in mm.")
    parser.add_argument("--target", nargs=2, type=float, metavar=("X_MM", "Y_MM"))
    parser.add_argument("--experimental", action="store_true", help="Allow an explicitly experimental, UNVALIDATED calibration.")
    args = parser.parse_args()
    tracker = BoardTracker(*load_camera())
    Kp, dp, Tpc = load_projector(allow_experimental=args.experimental)
    x0, y0, x1, y1 = config.BOARD_BOUNDS_MM
    xy = args.target if args.target else [(x0+x1)/2, (y0+y1)/2]
    target = np.array([*xy, 0.0])
    if not np.isfinite(target).all() or not inside_board(target):
        raise ValueError("Choose a target inside BOARD_BOUNDS_MM.")
    config.DATA_DIR.mkdir(exist_ok=True)
    path = config.DATA_DIR / "world_validation.csv"
    print("Space: measure. Move the entire rig, settle, then repeat. Esc: exit.")
    with Projector() as projector, Camera() as camera:
        while True:
            raw = camera.read()
            pose = tracker.estimate(raw)
            label = "EXPERIMENTAL / UNVALIDATED | " if args.experimental else ""
            show_preview(raw, label + f"Target {xy} mm | Board {'READY' if pose else 'NOT READY'} | Space: measure | Esc: exit")
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key != 32 or pose is None:
                continue
            try:
                uv = board_point_to_projector(target, pose.rvec, pose.tvec, Kp, dp, Tpc)
                measured, view, _ = measure_dot(camera, projector, tracker, *uv, reference=pose)
                observed = np.array(measured["board_xyz"])
                error = float(np.linalg.norm(observed[:2]-target[:2]))
                print(f"Target {target[:2]} mm -> observed {observed[:2].round(2)} mm; error {error:.2f} mm; projector {measured['projector_uv']}")
                new_file = not path.exists()
                with path.open("a", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    if new_file:
                        writer.writerow(["timestamp", "target_x_mm", "target_y_mm", "observed_x_mm", "observed_y_mm", "error_mm", "projector_u", "projector_v"])
                    writer.writerow([datetime.now(timezone.utc).isoformat(), *target[:2], *observed[:2], error, *measured["projector_uv"]])
                show_preview(view, f"Physical error {error:.2f} mm | any key: continue")
                if cv2.waitKey(0) & 0xFF in (27, ord('q')):
                    break
            except ValueError as exc:
                print(f"Rejected validation: {exc}")


if __name__ == "__main__":
    run_cli(main)
