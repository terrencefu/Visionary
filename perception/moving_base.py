"""Standalone ArUco-attached anchor tracking test. No projector or assembly steps."""
import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np
import config
from calibration.common import load_camera, run_cli
from hardware.camera import Camera, show_preview
from perception.aruco import BoardTracker, make_detector, board_drift_px
from perception.geometry import pose_matrix
from perception.anchor import estimate_anchor, plate_surface_heights
from perception.stl_matcher import load_models, verify, model_to_board
from perception.change_detector import StillnessGate, detect_change
from perception.pipeline import marker_quads
from perception.demo_change import workspace_polygon


@dataclass
class MarkerPose:
    matrix: np.ndarray
    corners: np.ndarray
    rms: float


def solve_marker(corners, size_mm, K, dist, max_rms):
    h = size_mm/2
    objects = np.array([[-h,-h,0],[h,-h,0],[h,h,0],[-h,h,0]],float)
    ok,rvec,tvec = cv2.solvePnP(objects,np.asarray(corners,float).reshape(4,2),K,dist,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError('Moving marker solvePnP failed')
    T = pose_matrix(rvec,tvec)
    depth = (objects@T[:3,:3].T+T[:3,3])[:,2]
    projected = cv2.projectPoints(objects,rvec,tvec,K,dist)[0].reshape(4,2)
    rms = float(np.sqrt(np.mean(np.sum((projected-corners)**2,axis=1))))
    if not np.isfinite(T).all() or np.any(depth<=0) or not np.isfinite(rms) or rms>max_rms:
        raise ValueError(f'Moving marker rejected: RMS={rms:.2f} px, limit={max_rms:.2f}')
    return MarkerPose(T,np.asarray(corners).reshape(4,2),rms)


def attach_anchor(marker_camera, board_camera, estimate, board_pose):
    """Freeze marker-to-anchor offset once; later only the marker is tracked."""
    anchor_board = np.eye(4)
    anchor_board[:3,:3] = model_to_board(np.eye(3),estimate['yaw_deg'],(0,0),board_pose).T
    anchor_board[:3,3] = np.r_[estimate['xy_mm'],0.]
    marker_board = np.linalg.inv(board_camera)@marker_camera
    # This first test assumes the sheet rests flat on the calibrated board.
    tilt = np.degrees(np.arccos(np.clip(abs(marker_board[2,2]),0,1)))
    if abs(marker_board[2,3])>5 or tilt>15:
        raise ValueError('For initial registration, lay the moving base flat on the board (within 5 mm / 15 deg)')
    return np.linalg.inv(marker_camera)@board_camera@anchor_board


def tracked_outline(marker_camera, anchor_marker, outline, K, dist):
    T = marker_camera@anchor_marker
    points = outline@T[:3,:3].T+T[:3,3]
    if not np.isfinite(points).all() or np.any(points[:,2]<=0):
        raise ValueError('Tracked anchor is behind camera or invalid')
    pixels = cv2.projectPoints(outline,cv2.Rodrigues(T[:3,:3])[0],T[:3,3],K,dist)[0].reshape(-1,2)
    return pixels,T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--marker-id',type=int,required=True)
    parser.add_argument('--marker-size',type=float,required=True,help='Black square side in mm, excluding white border')
    parser.add_argument('--cad',default='Fusion_output')
    parser.add_argument('--part-id',default='Plate 1x10 Silver')
    args = parser.parse_args()
    detector = make_detector()
    count = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,config.ARUCO_DICTIONARY)).bytesList.shape[0]
    if args.marker_id in config.MARKER_CENTERS_MM or not 0<=args.marker_id<count:
        parser.error('Use a marker ID in the configured dictionary, different from fixed-board IDs 0-3')
    if not np.isfinite(args.marker_size) or args.marker_size<=0:
        parser.error('--marker-size must be a positive finite mm measurement')
    K,dist = load_camera()
    board_tracker = BoardTracker(K,dist)
    models = load_models(args.cad)
    if args.part_id not in models:
        parser.error(f'Unknown component: {args.part_id}')
    mesh = models[args.part_id]
    lo,hi = mesh.min(axis=(0,1)),mesh.max(axis=(0,1))
    height = plate_surface_heights(mesh)['body']
    outline = np.array([[lo[0],lo[1],height],[hi[0],lo[1],height],
                        [hi[0],hi[1],height],[lo[0],hi[1],height]])
    baseline = baseline_pose = baseline_marker = anchor_marker = None
    gate = StillnessGate()
    last_status = None
    last_report = 0.
    print('Moving-base test: camera overlay only; no projector, servo, or assembly advancement.')
    print('Use a stiff, flat base. Keep its marker and the fixed board markers visible for registration.')
    print('Space: baseline with marker/base present but no anchor. Attach plate without moving base; V: identify and bind.')
    print('After binding: slide/rotate the whole base; green outline should follow the attached plate.')
    print('R: reset. Q/Esc: exit. Marker loss hides overlay; reacquisition resumes the SAME binding.')
    with Camera() as camera:
        while True:
            raw = camera.read()
            view = raw.copy()
            quads,ids,_ = detector.detectMarkers(raw)
            marker = None
            marker_error = f'Marker {args.marker_id} not visible'
            if ids is not None:
                indices = np.flatnonzero(ids.ravel()==args.marker_id)
                if len(indices)==1:
                    try:
                        marker = solve_marker(quads[indices[0]].reshape(4,2),args.marker_size,K,dist,config.MAX_ARUCO_RMS_PX)
                    except (ValueError,cv2.error) as exc:
                        marker_error = str(exc)
                elif len(indices)>1:
                    marker_error = 'Duplicate moving-marker ID visible; remove duplicate'
                cv2.aruco.drawDetectedMarkers(view,quads,ids)
            region = None
            board = None
            if anchor_marker is not None:
                status = 'TRACKING attached anchor' if marker is not None else marker_error+' - overlay hidden'
                if marker is not None:
                    try:
                        pixels,T = tracked_outline(marker.matrix,anchor_marker,outline,K,dist)
                        if np.isfinite(pixels).all() and np.max(np.abs(pixels))<1e6:
                            cv2.polylines(view,[np.rint(pixels).astype(np.int32)],True,(0,255,0),2)
                            cv2.drawFrameAxes(view,K,dist,cv2.Rodrigues(T[:3,:3])[0],T[:3,3],15)
                        if time.monotonic()-last_report>1:
                            print(f'TRACKING: anchor origin camera XYZ={np.round(T[:3,3],1)} mm; marker RMS={marker.rms:.2f} px')
                            last_report = time.monotonic()
                    except ValueError as exc:
                        status = str(exc)
            else:
                board = board_tracker.estimate(raw)
                frame = cv2.undistort(raw,K,dist)
                status = 'Place marked base flat; Space captures baseline'
                if board is None or marker is None:
                    gate.reset()
                    status = marker_error if marker is None else 'Need reliable fixed-board pose to register'
                elif baseline is not None:
                    mask = np.zeros(frame.shape[:2],np.uint8)
                    cv2.fillConvexPoly(mask,workspace_polygon(board,K),255)
                    reference_pixels = cv2.projectPoints(np.array([[-args.marker_size/2,-args.marker_size/2,0],
                        [args.marker_size/2,-args.marker_size/2,0],[args.marker_size/2,args.marker_size/2,0],
                        [-args.marker_size/2,args.marker_size/2,0]],float),
                        cv2.Rodrigues(baseline_marker[:3,:3])[0],baseline_marker[:3,3],K,dist)[0].reshape(4,2)
                    if board_drift_px(baseline_pose,board,K,dist)>config.MAX_BOARD_DRIFT_PX or np.max(np.linalg.norm(reference_pixels-marker.corners,axis=1))>config.MAX_BOARD_DRIFT_PX:
                        gate.reset()
                        status = 'Base/rig moved before identification: remove anchor and capture baseline again'
                    elif not gate.update(frame,search_mask=mask):
                        status = 'Settling - remove hand'
                    else:
                        moving_quad = cv2.undistortPoints(marker.corners.reshape(-1,1,2),K,dist,P=K).reshape(4,2)
                        region = detect_change(baseline,frame,exclude_quads=marker_quads(board,K,dist)+[moving_quad],search_mask=mask)
                        status = 'Anchor candidate found - V to identify/bind' if region is not None else 'Waiting for anchor placement'
            if status != last_status:
                print(status)
                last_status = status
            show_preview(view,status,name='Moving base anchor test')
            key = cv2.waitKey(1)&0xff
            if key in (27,ord('q')):
                break
            if key == ord('r'):
                baseline = baseline_pose = baseline_marker = anchor_marker = None
                gate.reset()
            elif key == 32 and anchor_marker is None and marker is not None and board is not None:
                baseline,baseline_pose,baseline_marker = frame.copy(),board,marker.matrix.copy()
                gate.reset()
                print('Baseline captured; attach anchor without shifting base')
            elif key == ord('v') and region is not None and marker is not None and board is not None:
                try:
                    result,scores,debug = verify(models,args.part_id,region,board,K)
                    print(result,scores)
                    cv2.imshow('Anchor STL comparison',debug)
                    if result == 'CORRECT SHAPE':
                        estimate = estimate_anchor(mesh,region,board,K,scores[0][2])
                        anchor_marker = attach_anchor(marker.matrix,pose_matrix(board.rvec,board.tvec),estimate,board)
                        print('BOUND: move base and anchor together. Marker-to-anchor offset is now fixed in memory.')
                except ValueError as exc:
                    print(f'Not bound: {exc}')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_cli(main)
