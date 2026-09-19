import argparse
from pathlib import Path
import cv2
import numpy as np
import config
from calibration.common import camera_signature, fixture_signature, run_cli
from calibration.dataset import load_poses, mean_transform, rotation_difference_deg
from perception.geometry import pose_matrix


def solve(poses):
    if len(poses) < 6:
        raise ValueError("Need at least 6 genuinely different poses; aim for 6-8.")
    camera_poses = [mean_transform(p["transforms"]) for p in poses]
    angle_span = max(rotation_difference_deg(a[:3, :3], b[:3, :3])
                     for a in camera_poses for b in camera_poses)
    depth_span = float(np.ptp([t[2, 3] for t in camera_poses]))
    print(f"Pose diversity: max rotation separation {angle_span:.1f} deg; depth span {depth_span:.1f} mm")
    if angle_span < 10 or depth_span < 50:
        raise ValueError("Insufficient view diversity: need >=10 deg rotation and >=50 mm depth span.")
    objects = [p["objects"] for p in poses]
    pixels = [p["projector_uv"] for p in poses]
    # Standard 5-coefficient pinhole model, no keystone/dynamic warp compensation.
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objects, pixels, config.PROJECTOR_SIZE, None, None)
    if not all(np.isfinite(a).all() for a in (K, dist)) or not np.isfinite(rms):
        raise ValueError("Non-finite projector calibration.")
    relative, per_view = [], []
    for pose, rvec, tvec, T_cb in zip(poses, rvecs, tvecs, camera_poses):
        T_pb = pose_matrix(rvec, tvec)
        if np.any((pose["objects"] @ T_pb[:3, :3].T + T_pb[:3, 3])[:, 2] <= 0):
            raise ValueError("Calibration placed board behind projector.")
        relative.append(T_pb @ np.linalg.inv(T_cb))
        pred = cv2.projectPoints(pose["objects"], rvec, tvec, K, dist)[0].reshape(-1, 2)
        per_view.append(float(np.sqrt(np.mean(np.sum((pred-pose["projector_uv"])**2, axis=1)))))
    mean = mean_transform(relative)
    rotation_spread = np.array([rotation_difference_deg(t[:3, :3], mean[:3, :3]) for t in relative])
    translation_spread = np.array([np.linalg.norm(t[:3, 3]-mean[:3, 3]) for t in relative])
    # Fit ONE camera->projector pose to all reconstructed camera-space points.
    # Each dot uses the camera board pose from its own black-background frame.
    camera_points = np.concatenate([
        np.array([T[:3, :3] @ p + T[:3, 3] for p, T in zip(pose["objects"], pose["transforms"])])
        for pose in poses]).astype(np.float64)
    projector_points = np.concatenate(pixels).astype(np.float64)
    initial_rvec = cv2.Rodrigues(mean[:3, :3])[0]
    ok, fixed_rvec, fixed_tvec = cv2.solvePnP(camera_points, projector_points, K, dist,
                                            initial_rvec, mean[:3, 3].reshape(3, 1).copy(),
                                            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("Failed to fit fixed camera->projector transform.")
    T_pc = pose_matrix(fixed_rvec, fixed_tvec)
    if np.any((camera_points @ T_pc[:3, :3].T + T_pc[:3, 3])[:, 2] <= 0):
        raise ValueError("Fixed transform places observations behind projector.")
    pred = cv2.projectPoints(camera_points, fixed_rvec, fixed_tvec, K, dist)[0].reshape(-1, 2)
    fixed_errors = np.linalg.norm(pred-projector_points, axis=1)
    fixed_rms = float(np.sqrt(np.mean(fixed_errors**2)))
    return dict(camera_matrix=K, dist_coeffs=dist, T_projector_camera=T_pc,
                rms=rms, fixed_rms=fixed_rms, per_view_rms=np.array(per_view),
                relative_transforms=np.array(relative), rotation_spread_deg=rotation_spread,
                translation_spread_mm=translation_spread, fixed_point_errors=fixed_errors,
                angle_span_deg=angle_span, depth_span_mm=depth_span)


def quality_failures(result):
    failures = []
    if max(result["rms"], result["fixed_rms"], max(result["per_view_rms"])) > config.MAX_PROJECTOR_RMS_PX:
        failures.append("projector reprojection error exceeds configured limit")
    if max(result["rotation_spread_deg"]) > config.MAX_RIG_ROTATION_SPREAD_DEG:
        failures.append("per-pose camera->projector rotation is inconsistent")
    if max(result["translation_spread_mm"]) > config.MAX_RIG_TRANSLATION_SPREAD_MM:
        failures.append("per-pose camera->projector translation is inconsistent")
    return failures


def main():
    parser = argparse.ArgumentParser(description="Solve inverse-camera intrinsics and a fixed camera->projector transform.")
    parser.add_argument("--data", type=Path, default=config.DATA_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--experimental", action="store_true",
                        help="Export existing solve diagnostics unchanged as UNVALIDATED, even if accuracy checks fail.")
    args = parser.parse_args()
    if config.PROJECTOR_CALIBRATION.exists() and not args.overwrite:
        raise ValueError("Projector calibration exists; use --overwrite to replace it.")
    poses = load_poses(args.data)
    diagnostics = args.data / "projector_solve_diagnostics.npz"
    if args.experimental:
        with np.load(diagnostics, allow_pickle=False) as saved:
            result = {key: saved[key].copy() for key in saved.files}
        if not all(np.isfinite(value).all() for value in result.values()):
            raise ValueError("Cannot export non-finite diagnostic results.")
    else:
        result = solve(poses)
    print(f"Projector RMS: {result['rms']:.3f} px; fixed-rig RMS: {result['fixed_rms']:.3f} px")
    for i, pose in enumerate(poses):
        print(f"{pose['name']}: RMS {result['per_view_rms'][i]:.2f} px, relative transform deviation "
              f"{result['rotation_spread_deg'][i]:.2f} deg / {result['translation_spread_mm'][i]:.2f} mm")
    args.data.mkdir(parents=True, exist_ok=True)
    if not args.experimental:
        np.savez(diagnostics, **result)
    failures = quality_failures(result)
    if failures and not args.experimental:
        raise ValueError("Calibration NOT published: " + "; ".join(failures) +
                         ". Inspect correspondences and rigid mount; diagnostics saved in data directory.")
    np.savez(config.PROJECTOR_CALIBRATION, **result, image_size=config.PROJECTOR_SIZE,
             fixture_signature=fixture_signature(), camera_signature=camera_signature(),
             pose_names=[p["name"] for p in poses], quality_passed=not failures,
             experimental=args.experimental, validation_status="UNVALIDATED",
             quality_failures=np.asarray(failures, dtype=str))
    if args.experimental:
        print("EXPERIMENTAL / UNVALIDATED: " + ("; ".join(failures) or "Physical validation pending."))
    print(f"Saved {config.PROJECTOR_CALIBRATION}. Physically validate world locking before using it for assembly.")


if __name__ == "__main__":
    run_cli(main)
