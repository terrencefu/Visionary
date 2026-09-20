"""Live stage-one test; no part matcher or CAD data is used."""
import cv2
import numpy as np
import config
import time
from contextlib import ExitStack

from calibration.common import load_camera
from hardware.camera import Camera, show_preview
from perception.aruco import BoardTracker
from perception.change_detector import StillnessGate, detect_change
from perception.pipeline import marker_quads


def capture_placement(camera, projector, tracker, K, dist, moving=None):
    """Measure under blank projection, after the hand/scene settles."""
    projector.black()
    start = time.monotonic()
    gate = StillnessGate()
    while time.monotonic()-start < 3.0:
        raw = camera.read()
        if time.monotonic()-start < .25:
            continue  # Flush lit camera frames and exposure transition.
        pose = tracker.estimate(raw)
        if moving is not None:
            moving.observe(raw)
            if moving.current is None:
                gate.reset()
                continue
        if pose is None:
            gate.reset()
            continue
        frame = cv2.undistort(raw,K,dist)
        mask = np.zeros(frame.shape[:2],np.uint8)
        cv2.fillConvexPoly(mask,workspace_polygon(pose,K),255)
        if gate.update(frame,search_mask=mask):
            return frame,pose,mask
    raise ValueError('Placement check needs a stable workspace and valid ArUco pose; remove hand and retry')


def workspace_polygon(pose, K):
    """Detection-only cardboard polygon in the undistorted camera image."""
    points = np.column_stack([np.asarray(config.DETECTION_WORKSPACE_MM, float), np.zeros(4)])
    return np.round(cv2.projectPoints(points, pose.rvec, pose.tvec, K,
                                     np.zeros(5))[0].reshape(-1, 2)).astype(np.int32)


def run(cad=None, part_id=None, anchor=False, base_marker_id=None, base_marker_size=None):
    K, dist = load_camera()
    tracker = BoardTracker(K, dist)
    gate = StillnessGate()
    baseline = None
    search_mask = None
    polygon = None
    last_status = None
    models = None
    scene = None
    registration = None
    projector = None
    height_mode = 'body'
    heights = None
    assembly = None
    step_before = None
    step_pose = None
    step_transform = None
    moving = None
    preparing = False
    if base_marker_id is not None:
        from assembly.base_registration import MovingRegistration
        moving = MovingRegistration(base_marker_id,base_marker_size,K,dist)
    if cad is not None:
        from perception.stl_matcher import load_models
        models = load_models(cad)
        if part_id not in models:
            raise ValueError(f'Choose --part-id from: {list(models)}')
        if anchor:
            from perception.anchor import plate_surface_heights
            heights = plate_surface_heights(models[part_id])
            print(f"Height test: 1 = STL maximum {heights['max']:.2f} mm; "
                  f"2 = inferred body deck {heights['body']:.2f} mm. Anchor XY/yaw stay fixed.")
        print(f'Expected: {part_id}. Press V after detection to verify STL shape.')
        print('Assumes an isolated part resting in its CAD upright orientation; yaw is free.')
    print("Clear the board, then press Space to capture a baseline.")
    print("Place any object and remove your hand. Space resets; Q/Esc exits.")
    with ExitStack() as stack:
        if anchor:
            from hardware.projector import Projector
            from projection.world import load_projector
            calibration = load_projector()
            projector = stack.enter_context(Projector())
            print('ASSEMBLY: V fits anchor. Later V checks placement; Enter checks again and advances only on PASS.')
            if moving is None:
                print('Keep the anchor fixed. B blanks and resets. CAD height is assumed; no anchor recovery.')
            if moving is not None:
                print('MOVING BASE: Space before each addition. Blue/red checks adapt to camera/base motion; settle before V.')
                print('M cancels a pending placement: remove any unverified new part BEFORE M/recapturing baseline.')
        camera = stack.enter_context(Camera())
        while True:
            raw = camera.read()
            pose = tracker.estimate(raw)
            if moving is not None:
                moving.observe(raw)
            frame = cv2.undistort(raw, K, dist)
            view = frame.copy()
            region = None
            if scene is not None:
                from projection.guidance import render_scene
                if moving is not None and not preparing and moving.current is not None and pose is not None:
                    try:
                        assembly.transform = moving.transform(pose)
                        scene = assembly.scene(height_mode)
                    except ValueError as exc:
                        projector.black()
                        show_preview(view, str(exc))
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord('q')):
                            break
                        continue
                status = assembly.message + ' | V=check Enter=check/next B=reset Q=quit'
                if moving is not None and (preparing or moving.current is None or moving.invalidated):
                    projector.black()
                    if preparing:
                        status = assembly.message+' | MOVE BASE ONLY; hands off, Space=arm next placement'
                    elif moving.current is None:
                        status = moving.reason+'; projection paused'
                    else:
                        status = 'Base moved during placement: remove unverified part, M=move phase, Space=re-arm'
                elif pose is None:
                    projector.black()
                    status = 'ArUco tracking unavailable; projection blank, advancement paused'
                else:
                    try:
                        cv2.imshow(projector.name, render_scene(scene, pose, calibration))
                    except ValueError as exc:
                        projector.black()
                        scene = registration = None
                        print(f'Projection stopped: {exc}')
                show_preview(view,status)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                if moving is not None and key==ord('m'):
                    projector.black()
                    preparing = True
                    assembly.checked_index = None
                    moving.begin_move()
                    print('Placement cancelled. Keep only previously accepted parts; move base, then Space.')
                if moving is not None and key==32 and preparing:
                    try:
                        measured_frame,measured_pose,mask = capture_placement(camera,projector,tracker,K,dist,moving)
                        updated = moving.transform(measured_pose)
                        previous = assembly.transform
                        assembly.transform = updated
                        try:
                            candidate_scene = assembly.scene(height_mode)
                        except ValueError:
                            assembly.transform = previous
                            raise
                        step_before,step_pose = measured_frame.copy(),measured_pose
                        step_transform = assembly.transform.copy()
                        moving.arm()
                        preparing = False
                        scene = candidate_scene
                        print('BASELINE READY: add only the requested part now. V checks; Enter checks and advances.')
                    except ValueError as exc:
                        projector.black()
                        print(f'Not armed: {exc}')
                movable_ready = moving is None or (not preparing and moving.current is not None and not moving.invalidated)
                if key in (10,13,ord('v')) and (pose is None or not movable_ready):
                    print('Check paused: ' + status)
                if key in (10,13,ord('v'),ord('1'),ord('2')) and scene is not None and pose is not None and movable_ready:
                    try:
                        if key in (10,13,ord('v')):
                            if assembly.index > 0:
                                assembly.checked_index = None
                            measured_frame, measured_pose, mask = capture_placement(camera,projector,tracker,K,dist,moving)
                            if moving is not None:
                                assembly.transform = moving.transform(measured_pose)
                            passed = True
                            if assembly.index > 0:
                                assembly.checked_index = None
                                if assembly.names[assembly.index] not in config.PLACEMENT_PART_COLOURS:
                                    from perception.aruco import board_drift_px
                                    if board_drift_px(step_pose,measured_pose,K,dist)>config.MAX_BOARD_DRIFT_PX:
                                        raise ValueError('Uncoloured placement needs a stationary baseline')
                                passed,message,debug = assembly.validate_frames(step_before,measured_frame,measured_pose,K,
                                    exclude_quads=marker_quads(measured_pose,K,dist)+(moving.marker_quad() if moving else []),search_mask=mask,
                                    before_pose=step_pose,before_transform=step_transform)
                                print(message)
                                cv2.imshow('Placement comparison',debug)
                            if key in (10,13) and passed:
                                step_before,step_pose = measured_frame.copy(),measured_pose
                                step_transform = assembly.transform.copy()
                                assembly.advance()
                                print(assembly.message)
                                if assembly.complete:
                                    break
                                if moving is not None:
                                    moving.begin_move()
                                    preparing = True
                                    print('Between steps: reposition base if wanted, then Space BEFORE adding next part.')
                        elif assembly.index < 2:
                            height_mode = 'max' if key == ord('1') else 'body'
                        scene = assembly.scene(height_mode)
                    except ValueError as exc:
                        projector.black()
                        print(f'Not advanced: {exc}')
                if key == ord('b'):
                    projector.black()
                    scene = registration = None
                    assembly = None
                    if moving is not None:
                        moving.adapt_motion = False
                    baseline = None
                    preparing = False
                    if moving is not None:
                        moving.begin_move()
                        moving.marker_from_cad = None
                    gate.reset()
                    print('Reset: clear workspace and capture a new baseline with Space.')
                continue
            if polygon is not None:
                cv2.polylines(view, [polygon], True, (0, 255, 255), 2)
            if moving is not None and (moving.current is None or moving.invalidated):
                gate.reset()
                status = moving.reason if moving.current is None else 'Base moved: remove anchor and capture baseline again'
            elif pose is None:
                gate.reset()
                status = "Tracking unavailable - keep markers visible"
            elif baseline is None:
                status = "Clear board, then press Space"
            elif not gate.update(frame, search_mask=search_mask):
                status = "Settling - remove your hand"
            else:
                region = detect_change(baseline, frame,
                                       exclude_quads=marker_quads(pose, K, dist)+(moving.marker_quad() if moving else []),
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
                    if anchor and result == 'CORRECT SHAPE':
                        from perception.anchor import estimate_anchor, register_cad
                        try:
                            estimate = estimate_anchor(models[part_id], region, pose, K, scores[0][2])
                            registration = register_cad(cad, part_id, estimate, pose)
                            height_mode = 'body'
                            from assembly.manual_guidance import ManualAssembly
                            assembly = ManualAssembly(*registration,cad,heights)
                            if moving is not None:
                                moving.adapt_motion = True
                                moving.bind(pose,assembly.transform)
                                moving.arm()
                            scene = assembly.scene(height_mode)
                            print(assembly.message)
                            print(f"ANCHOR: center XY={estimate['xy_mm']} mm; yaw={estimate['yaw_deg']:.2f} deg; "
                                  f"unshifted overlap={estimate['overlap']:.3f}")
                            print('Green outline uses the inferred body deck. Enter confirms and '
                                  'shows the next part; B rejects.')
                        except ValueError as exc:
                            projector.black()
                            scene = registration = None
                            print(f'Anchor rejected: {exc}')
            if key == ord(" ") and pose is not None and (moving is None or moving.current is not None):
                polygon = workspace_polygon(pose, K)
                search_mask = np.zeros(frame.shape[:2], np.uint8)
                cv2.fillConvexPoly(search_mask, polygon, 255)
                baseline = frame.copy()
                if moving is not None:
                    moving.arm()
                gate.reset()
                print("Baseline captured")
    cv2.destroyAllWindows()
