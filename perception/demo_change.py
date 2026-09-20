"""Live stage-one test; no part matcher or CAD data is used."""
import cv2
import numpy as np
import config

from calibration.common import load_camera
from hardware.camera import Camera, show_preview
from perception.aruco import BoardTracker
from perception.change_detector import StillnessGate, detect_change
from perception.pipeline import marker_quads


def workspace_polygon(pose, K):
    """Detection-only cardboard polygon in the undistorted camera image."""
    points = np.column_stack([np.asarray(config.DETECTION_WORKSPACE_MM, float), np.zeros(4)])
    return np.round(cv2.projectPoints(points, pose.rvec, pose.tvec, K,
                                     np.zeros(5))[0].reshape(-1, 2)).astype(np.int32)


def run(cad=None, part_id=None):
    K, dist = load_camera()
    tracker = BoardTracker(K, dist)
    gate = StillnessGate()
    baseline = None
    search_mask = None
    polygon = None
    last_status = None
    models = None
    if cad is not None:
        from perception.stl_matcher import load_models
        models = load_models(cad)
        if part_id not in models:
            raise ValueError(f'Choose --part-id from: {list(models)}')
        print(f'Expected: {part_id}. Press V after detection to verify STL shape.')
        print('Assumes an isolated part resting in its CAD upright orientation; yaw is free.')
    print("Clear the board, then press Space to capture a baseline.")
    print("Place any object and remove your hand. Space resets; Q/Esc exits.")
    with Camera() as camera:
        while True:
            raw = camera.read()
            pose = tracker.estimate(raw)
            frame = cv2.undistort(raw, K, dist)
            view = frame.copy()
            region = None
            if polygon is not None:
                cv2.polylines(view, [polygon], True, (0, 255, 255), 2)
            if pose is None:
                gate.reset()
                status = "Tracking unavailable - keep markers visible"
            elif baseline is None:
                status = "Clear board, then press Space"
            elif not gate.update(frame, search_mask=search_mask):
                status = "Settling - remove your hand"
            else:
                region = detect_change(baseline, frame,
                                       exclude_quads=marker_quads(pose, K, dist),
                                       search_mask=search_mask)
                status = "No accepted change"
                if region is not None:
                    status = "CHANGE DETECTED - identity not checked"
                    x, y, w, h = region.bbox
                    cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if status != last_status:
                print(status)
                last_status = status
            show_preview(view, status)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord('v') and models is not None:
                if region is None:
                    print('Verification needs a settled detected change and valid board pose.')
                else:
                    from perception.stl_matcher import verify
                    print('Comparing STL silhouettes; keep the rig and object stationary...')
                    result, scores, debug = verify(models, part_id, region, pose, K)
                    print(f'{result} (visible shape only; not assembly placement validation)')
                    for name, score, yaw in scores:
                        print(f'  {name}: overlap={score:.3f}, sampled yaw={yaw:.0f} deg')
                    cv2.imshow('STL comparison: green=observed red=model yellow=overlap', debug)
            if key == ord(" ") and pose is not None:
                polygon = workspace_polygon(pose, K)
                search_mask = np.zeros(frame.shape[:2], np.uint8)
                cv2.fillConvexPoly(search_mask, polygon, 255)
                baseline = frame.copy()
                gate.reset()
                print("Baseline captured")
    cv2.destroyAllWindows()
