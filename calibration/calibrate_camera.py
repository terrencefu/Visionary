import argparse
import cv2
import numpy as np
import config
from hardware.camera import Camera, show_preview
from calibration.common import run_cli


def main():
    parser = argparse.ArgumentParser(description="Fresh camera intrinsic calibration at 1080p.")
    parser.add_argument("--square-mm", type=float, default=config.CHESSBOARD_SQUARE_MM)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.square_mm) or args.square_mm <= 0:
        raise ValueError("Measure the printed square size in millimeters.")
    if config.CAMERA_CALIBRATION.exists() and not args.overwrite:
        raise ValueError("Calibration already exists; use --overwrite to replace it.")
    size = config.CHESSBOARD_INNER_CORNERS
    obj = np.zeros((size[0]*size[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2) * args.square_mm
    objects, images = [], []
    print(f"{size} inner corners, {args.square_mm} mm squares. Space captures; C solves; Esc cancels.")
    print("Vary tilt, depth, and image position. Keep focus/zoom and 1080p fixed.")
    with Camera() as camera:
        while True:
            raw = camera.read()
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            view = raw.copy()
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                          (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                cv2.drawChessboardCorners(view, size, corners, found)
            show_preview(view, f"{len(images)}/{config.CHESSBOARD_TARGET_FRAMES} | Space: capture | C: solve | Esc: cancel")
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                return
            if key == 32 and found:
                if any(np.sqrt(np.mean((corners-previous)**2)) < 8 for previous in images):
                    print("Rejected near-duplicate. Change angle, position, or distance.")
                else:
                    images.append(corners.copy())
                    objects.append(obj.copy())
                    print(f"Accepted frame {len(images)}")
            if key == ord('c'):
                if len(images) < 15:
                    print("Need at least 15 varied views; aim for 30.")
                    continue
                break
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objects, images, config.CAMERA_SIZE, None, None)
    errors = []
    for o, im, r, t in zip(objects, images, rvecs, tvecs):
        predicted = cv2.projectPoints(o, r, t, K, dist)[0]
        errors.append(float(np.sqrt(np.mean(np.sum((predicted-im)**2, axis=2)))))
    if not np.isfinite(rms) or not np.isfinite(K).all() or not np.isfinite(dist).all():
        raise ValueError("Non-finite calibration; no file saved.")
    np.savez(config.CAMERA_CALIBRATION, camera_matrix=K, dist_coeffs=dist,
             image_size=config.CAMERA_SIZE, rms=rms, per_view_rms=errors,
             square_mm=args.square_mm, object_points=objects, image_points=images)
    print(f"Saved {config.CAMERA_CALIBRATION}\nRMS: {rms:.3f} px\nK:\n{K}\nDistortion: {dist.ravel()}")
    print("Per-view RMS:", np.round(errors, 3))
    if rms > 1.0:
        print("WARNING: RMS exceeds 1 px. Inspect blur, corners, focus, and view diversity before proceeding.")


if __name__ == "__main__":
    run_cli(main)
