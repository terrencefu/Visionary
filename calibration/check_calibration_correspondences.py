import argparse
from pathlib import Path
import cv2
import numpy as np
import config
from calibration.dataset import load_poses
from calibration.common import run_cli


def main():
    parser = argparse.ArgumentParser(description="Per-pose board XY -> projector UV homography residuals.")
    parser.add_argument("--data", type=Path, default=config.DATA_DIR)
    args = parser.parse_args()
    all_errors = []
    for pose in load_poses(args.data):
        xy, uv = pose["objects"][:, :2], pose["projector_uv"]
        H, mask = cv2.findHomography(xy, uv, cv2.RANSAC, 3.0)
        if H is None:
            raise ValueError(f"Cannot fit {pose['name']}.")
        pred = cv2.perspectiveTransform(xy.reshape(-1, 1, 2), H).reshape(-1, 2)
        errors = np.linalg.norm(pred-uv, axis=1)
        all_errors.extend(errors.tolist())
        print(f"{pose['name']}: n={len(xy)}, inliers={int(mask.sum())}, mean={errors.mean():.2f}, RMS={np.sqrt(np.mean(errors**2)):.2f}, max={errors.max():.2f} px")
        print("  point residuals:", np.round(errors, 2))
    print(f"Overall mean: {np.mean(all_errors):.2f} px; RMS: {np.sqrt(np.mean(np.square(all_errors))):.2f} px")
    print("Homography is a diagnostic; projector lens distortion can also cause residuals.")


if __name__ == "__main__":
    run_cli(main)
