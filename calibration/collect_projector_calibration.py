import argparse
import json
from datetime import datetime, timezone
import cv2
import numpy as np
import config
from calibration.common import camera_signature, fixture_signature, load_camera, run_cli
from calibration.measurement import measure_dot
from hardware.camera import Camera, show_preview
from hardware.projector import Projector
from perception.aruco import BoardTracker


def main():
    parser = argparse.ArgumentParser(description="Collect ONE stationary pose per invocation; move the entire rigid rig between runs.")
    parser.add_argument("--rigid-mount-ready", action="store_true", required=True,
                        help="Confirm camera/projector mount is rigid and projector geometry settings are fixed.")
    args = parser.parse_args()
    K, dist = load_camera()
    tracker = BoardTracker(K, dist, max_rms_px=config.COLLECTOR_MAX_ARUCO_RMS_PX)
    metadata = dict(schema=1, timestamp=datetime.now(timezone.utc).isoformat(),
                    fixture_signature=fixture_signature(), camera_signature=camera_signature(),
                    camera_size=config.CAMERA_SIZE, projector_size=config.PROJECTOR_SIZE,
                    rigid_mount_ready=args.rigid_mount_ready)
    rows = []
    with Projector() as projector, Camera() as camera:
        print("Hold board and rig stationary. Space begins the grid; Esc cancels.")
        while True:
            raw = camera.read()
            pose = tracker.estimate(raw)
            show_preview(raw, f"Board {'READY' if pose else 'NOT READY'} | Space: collect | Esc: cancel")
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                return
            if key == 32 and pose is not None:
                break
        xs = np.rint(np.linspace(*config.GRID_U_RANGE_PX, config.GRID_COLS)).astype(int)
        ys = np.rint(np.linspace(*config.GRID_V_RANGE_PX, config.GRID_ROWS)).astype(int)
        if (len(xs) < 2 or len(ys) < 2 or np.any(np.diff(xs) <= 0)
                or np.any(np.diff(ys) <= 0) or xs[0] < 0 or xs[-1] >= projector.width
                or ys[0] < 0 or ys[-1] >= projector.height):
            raise ValueError("Grid must have >=2 distinct increasing pixels per axis within the projector image.")
        for v in ys:
            for u in xs:
                try:
                    row, view, _ = measure_dot(camera, projector, tracker, int(u), int(v), reference=pose)
                    rows.append(row)
                    show_preview(view, f"Accepted {len(rows)} | Esc: abort")
                    print(f"Accept projector ({u},{v}) -> board {np.round(row['board_xyz'], 2)} mm; ArUco {row['aruco_rms']:.2f} px")
                except ValueError as exc:
                    print(f"Reject ({u},{v}): {exc}")
    if len(rows) < 8:
        raise ValueError(f"Only {len(rows)} usable points. Need >=8 spread across board/projector; pose not saved.")
    config.DATA_DIR.mkdir(exist_ok=True)
    path = config.DATA_DIR / ("pose_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f") + ".json")
    path.write_text(json.dumps(dict(metadata=metadata, points=rows), indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} correspondences to {path}")
    print("Move/tilt the entire rigid camera-projector rig for the next pose. Collect 6-8 distinct poses.")


if __name__ == "__main__":
    run_cli(main)
