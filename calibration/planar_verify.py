"""Read-only physical verification of saved H at the current stationary position."""
import hashlib
import json
import time
from types import SimpleNamespace
import cv2
import numpy as np
import config
from calibration.common import camera_signature, fixture_signature, load_camera
from calibration.planar import wait_ready
from calibration.measurement import measure_dot
from hardware.camera import Camera
from hardware.projector import Projector
from perception.aruco import BoardTracker, board_drift_px
from projection.planar import held_out_targets, board_to_pixel


def stable_reference(poses, K, dist):
    """Select a real observed medoid pose, never fit or alter H."""
    if len(poses) < 9 or any(p is None for p in poses):
        raise ValueError('Need nine reliable consecutive poses for verification.')
    expected=sorted(config.MARKER_CENTERS_MM)
    if any(sorted(p.visible_ids)!=expected for p in poses):
        raise ValueError(f'Reference window requires the complete configured marker set {expected}.')
    obj=poses[0].object_points
    pixels=np.array([cv2.projectPoints(obj,p.rvec,p.tvec,K,dist)[0].reshape(-1,2) for p in poses])
    median=np.median(pixels,axis=0)
    distances=np.max(np.linalg.norm(pixels-median,axis=2),axis=1)
    if max(distances)>config.MAX_BOARD_DRIFT_PX:
        raise ValueError('Current reference window is unstable; verification remains blank.')
    reference=poses[int(np.argmin(distances))]
    return reference,dict(jitter_median_px=float(np.median(distances)),max_deviation_px=float(max(distances)),
                          rvec=reference.rvec.tolist(),tvec=reference.tvec.tolist(),
                          object_points=obj.tolist(),marker_ids=reference.visible_ids)


def acquire_reference(camera, projector, tracker, report, timeout=30.):
    """Retry incomplete/unstable windows while retaining every sampled-frame diagnostic."""
    expected=set(config.MARKER_CENTERS_MM)
    frames=report.setdefault('reference_acquisition',[])
    report['reference_requirements']=dict(marker_ids=sorted(expected),consecutive_frames=9,timeout_s=timeout)
    poses=[]
    start=time.monotonic()
    projector.black()
    last_reason='No frames captured'
    while time.monotonic()-start < timeout:
        pose=tracker.estimate(camera.settled_frame())
        debug=tracker.last_diagnostics
        detected=list(debug.detected_ids)
        missing=sorted(expected-set(detected))
        complete=(pose is not None and sorted(pose.visible_ids)==sorted(expected))
        entry=dict(frame=len(frames)+1,elapsed_s=time.monotonic()-start,
                   detected_ids=detected,missing_ids=missing,known_ids=list(debug.known_ids),
                   pose_rms_px=debug.rms_error_px,pose_rejection=debug.rejection_reason)
        if not complete:
            poses.clear()
            last_reason=f'Missing markers {missing}' if missing else (debug.rejection_reason or 'Incomplete/duplicate marker set')
            entry['restart_reason']=last_reason
        else:
            poses.append(pose)
            poses=poses[-9:]
        entry['consecutive_valid_frames']=len(poses)
        result=None
        if len(poses)==9:
            try:
                result=stable_reference(poses,tracker.K,tracker.dist)
                entry['stable_window']=True
            except ValueError as exc:
                last_reason=str(exc)
                entry['restart_reason']=last_reason
                entry['stable_window']=False
                # Keep sliding the complete window so a transient pose outlier ages out.
        frames.append(entry)
        print('Reference acquisition: '+json.dumps(entry),flush=True)
        if result is not None and time.monotonic()-start < timeout:
            return result
    raise ValueError(f'Reference acquisition timed out after {timeout:g}s: need 9 consecutive complete, stable frames '
                     f'with IDs {sorted(expected)}. Last issue: {last_reason}. Per-frame evidence saved in verification report.')


def landing_checks(mapping, reference, camera, projector, tracker):
    measurements=[]
    try:
        for target in held_out_targets(mapping):
            # The local reference is ONLY a motion guard for these test flashes.
            # The saved homography is used verbatim to command every target.
            row,_,_=measure_dot(camera,projector,tracker,*board_to_pixel(mapping,target),reference=reference)
            error=float(np.linalg.norm(np.asarray(row['board_xyz'][:2])-target))
            measurements.append(dict(target_xy=target.tolist(),error_mm=error,measurement=row))
            print(f"Held-out {target.round(2)}: landing error {error:.3f} mm",flush=True)
    except ValueError as exc:
        measurements.append(dict(rejection=str(exc)))
        print('Verification stopped: '+str(exc))
    finally:
        projector.black()
    errors=[m['error_mm'] for m in measurements if 'error_mm' in m]
    return dict(measurements=measurements,passed=len(errors)==5 and max(errors)<=config.PLANAR_VALIDATION_MAX_MM,
                rms_mm=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                max_mm=max(errors) if errors else None,limit_mm=config.PLANAR_VALIDATION_MAX_MM)


def verify_current(dot_color=None):
    original=config.PLANAR_CALIBRATION.read_bytes()
    record=json.loads(original)
    config.DOT_COLOR=dot_color or record.get('dot_color','green')
    if config.DOT_COLOR not in ('green','magenta'):
        raise ValueError('Saved dot color must be green or magenta.')
    print(f'Verification dot color: {config.DOT_COLOR}')
    if (record['camera_signature']!=camera_signature() or record['fixture_signature']!=fixture_signature()
            or record['projector_size']!=list(config.PROJECTOR_SIZE)):
        raise ValueError('Calibration signatures mismatch; cannot verify against changed geometry/intrinsics.')
    tracker=BoardTracker(*load_camera(),max_rms_px=config.COLLECTOR_MAX_ARUCO_RMS_PX)
    saved=SimpleNamespace(**{k:np.asarray(v) for k,v in record['reference'].items()})
    report=dict(calibration_sha256=hashlib.sha256(original).hexdigest(),saved_status=record['status'],
                timestamp=time.time(),passed=False,dot_color=config.DOT_COLOR,
                note='Verification only. Local reference does not authorize guidance or replace saved reference.')
    config.DATA_DIR.mkdir(exist_ok=True)
    path=config.DATA_DIR / f'planar_verify_current_{time.time_ns()}.json'
    print('READ-ONLY CURRENT-POSITION TEST: only held-out dots; no guidance or reference acceptance.')
    try:
        with Projector() as projector, Camera() as camera:
            wait_ready(camera,tracker)
            projector.black()
            reference,window=acquire_reference(camera,projector,tracker,report)
            report['current_window']=window
            report['saved_reference_drift_px']=board_drift_px(saved,reference,tracker.K,tracker.dist)
            report['camera_properties']={name:float(camera.cap.get(prop)) for name,prop in
                [('exposure',cv2.CAP_PROP_EXPOSURE),('gain',cv2.CAP_PROP_GAIN),
                 ('autofocus',cv2.CAP_PROP_AUTOFOCUS),('focus',cv2.CAP_PROP_FOCUS),
                 ('auto_exposure',cv2.CAP_PROP_AUTO_EXPOSURE)]}
            print(f"Current stable window versus saved reference: {report['saved_reference_drift_px']:.3f} px")
            report.update(landing_checks(record['mapping'],reference,camera,projector,tracker))
    except (ValueError,RuntimeError,KeyboardInterrupt) as exc:
        report['rejection']=str(exc) or 'Interrupted'
        raise
    finally:
        report['calibration_bytes_unchanged']=config.PLANAR_CALIBRATION.read_bytes()==original
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f'Verification report: {path}')
    print(f"PASS={report['passed']}; RMS={report['rms_mm']} mm; max={report['max_mm']} mm; limit={report['limit_mm']} mm")
    print('Camera-derived landing errors share the ArUco model; visually check against physical marks too.')
    print('Pass: reference re-establishment can be reviewed separately; nothing accepted automatically.'
          if report['passed'] else 'Fail/incomplete: do not rebase the reference. Recalibrate the planar mapping.')
