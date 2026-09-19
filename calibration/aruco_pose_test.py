import argparse
import time
import cv2
from calibration.common import load_camera, run_cli
from hardware.camera import Camera, show_preview
from perception.aruco import BoardTracker, make_detector


def main():
    parser = argparse.ArgumentParser(description="Detect markers first, then verify world pose.")
    parser.add_argument("--detect-only", action="store_true", help="No geometry or intrinsics needed.")
    args = parser.parse_args()
    detector = make_detector()
    tracker = None if args.detect_only else BoardTracker(*load_camera())
    last_log = 0
    with Camera() as camera:
        while True:
            raw = camera.read()
            view = raw.copy()
            corners, ids, _ = detector.detectMarkers(raw)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(view, corners, ids)
            status = f"Detected IDs: {ids.flatten().tolist() if ids is not None else []}"
            pose = tracker.estimate(raw) if tracker else None
            if tracker:
                status = "No reliable board pose (need >=3 known markers, low residual)"
            if pose:
                cv2.drawFrameAxes(view, tracker.K, tracker.dist, pose.rvec, pose.tvec, 50)
                status = f"IDs {pose.visible_ids} | RMS {pose.reprojection_error:.2f} px"
                if time.monotonic()-last_log > 1:
                    print(status, "| Camera in board mm:", pose.camera_position_board.round(1))
                    last_log = time.monotonic()
            show_preview(view, status)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break


if __name__ == "__main__":
    run_cli(main)
