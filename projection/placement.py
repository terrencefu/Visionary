"""Standalone stationary LEGO footprint guidance; no assembly or servo control."""
import argparse
import json
import time
from types import SimpleNamespace
import cv2
import numpy as np
import config
from calibration.common import camera_signature, fixture_signature, load_camera, run_cli
from calibration.planar import wait_ready
from projection.stationary import StationaryMonitor
from hardware.camera import Camera, show_preview
from hardware.projector import Projector
from perception.aruco import BoardTracker, board_drift_px
from projection.planar import board_to_pixel, contains, transform


def footprint_corners(x, y, width, height, rotation):
    if not np.isfinite([x,y,width,height,rotation]).all() or width <= 0 or height <= 0:
        raise ValueError('Center/rotation must be finite and footprint dimensions positive.')
    angle = np.radians(rotation % 360)
    R = np.array([[np.cos(angle),-np.sin(angle)], [np.sin(angle),np.cos(angle)]])
    offsets = np.array([[-width/2,-height/2],[width/2,-height/2],
                        [width/2,height/2],[-width/2,height/2]])
    return offsets @ R.T + [x,y]


def placement_canvas(mapping, part_id, x, y, width, height, rotation):
    corners = footprint_corners(x,y,width,height,rotation)
    # Convex workspace and inlier hull: every corner inside implies all edges inside.
    uv = np.array([board_to_pixel(mapping,p) for p in corners])
    H = np.asarray(mapping['H'],float)
    denominators = np.c_[corners,np.ones(4)] @ H[2]
    if np.min(denominators) < 0 < np.max(denominators):
        raise ValueError('Footprint crosses the homography horizon.')
    w,h = config.PROJECTOR_SIZE
    frame = np.zeros((h,w,3),np.uint8)
    color = (0,255,0)
    cv2.polylines(frame,[np.rint(uv).astype(np.int32)],True,color,2,cv2.LINE_AA)
    radius = min(2.,width/4,height/4)
    for a,b in [((x-radius,y),(x+radius,y)),((x,y-radius),(x,y+radius))]:
        p,q = [tuple(np.rint(board_to_pixel(mapping,c)).astype(int)) for c in (a,b)]
        cv2.line(frame,p,q,color,2,cv2.LINE_AA)
    # Place readable screen text beside the footprint, entirely inside support.
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX,.5,1
    (tw,th), baseline = cv2.getTextSize(part_id,font,scale,thickness)
    lo,hi = np.floor(uv.min(axis=0)).astype(int), np.ceil(uv.max(axis=0)).astype(int)
    choices = [(hi[0]+8,lo[1]+th),(lo[0]-tw-8,lo[1]+th),
               (lo[0],lo[1]-baseline-8),(lo[0],hi[1]+th+8)]
    for tx,ty in choices:
        box = np.array([[tx-2,ty-th-2],[tx+tw+2,ty-th-2],
                        [tx+tw+2,ty+baseline+2],[tx-2,ty+baseline+2]])
        if (np.all(box >= 0) and np.all(box < [w,h]) and
                all(contains(mapping,p) for p in transform(np.linalg.inv(H),box))):
            cv2.putText(frame,part_id,(int(tx),int(ty)),font,scale,color,thickness,cv2.LINE_AA)
            break
    else:
        raise ValueError('No room beside footprint for the part label within calibrated workspace.')
    return frame,corners,uv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part-id',required=True)
    parser.add_argument('--center',nargs=2,type=float,required=True,metavar=('X_MM','Y_MM'))
    parser.add_argument('--width',type=float,required=True,help='Footprint width in mm along local X.')
    parser.add_argument('--height',type=float,required=True,help='Footprint height in mm along local Y.')
    parser.add_argument('--rotation',type=float,default=0.,help='Degrees: positive turns +X toward +Y.')
    parser.add_argument('--stationary-ready',required=True,action='store_true')
    parser.add_argument('--movement-px',type=float,default=config.MAX_BOARD_DRIFT_PX)
    parser.add_argument('--movement-window',type=int,default=9)
    parser.add_argument('--movement-persist',type=float,default=.5,help='Seconds of filtered offset before stopping.')
    args = parser.parse_args()
    if not args.part_id.strip() or not args.part_id.isascii() or not args.part_id.isprintable():
        raise ValueError('Use a nonempty printable ASCII part ID for the projector label.')
    record = json.loads(config.PLANAR_CALIBRATION.read_text(encoding='utf-8'))
    if (record['camera_signature'] != camera_signature() or record['fixture_signature'] != fixture_signature()
            or record['projector_size'] != list(config.PROJECTOR_SIZE) or record['status'].startswith('STALE')):
        raise ValueError('Saved planar calibration is stale/incompatible; recalibrate first.')
    frame,corners,uv = placement_canvas(record['mapping'],args.part_id,*args.center,
                                       args.width,args.height,args.rotation)
    print(f'Part {args.part_id}; board corners mm:\n{corners}\nProjector corners px:\n{uv}')
    print('Stationary planar guidance only. Esc/Q exits. No calibration writes or servo movement.')
    print('If the projector moves relative to the board, stop and recalibrate.')
    reference = SimpleNamespace(**{k:np.asarray(v) for k,v in record['reference'].items()})
    tracker = BoardTracker(*load_camera(),max_rms_px=config.COLLECTOR_MAX_ARUCO_RMS_PX)
    guard = StationaryMonitor(reference,tracker.K,tracker.dist,args.movement_px,
                              args.movement_window,args.movement_persist)
    config.DATA_DIR.mkdir(exist_ok=True)
    log_path=config.DATA_DIR / ('placement_motion_' + str(time.time_ns()) + '.jsonl')
    print('Reference board->camera rvec (radians):',reference.rvec.ravel(),
          'tvec (mm):',reference.tvec.ravel())
    print(f'Motion log: {log_path}; limit={args.movement_px}px, window={args.movement_window}, persistence={args.movement_persist}s')
    last_log=0.
    with log_path.open('w',encoding='utf-8') as log, Projector() as projector, Camera() as camera:
        log.write(json.dumps(dict(reference=record['reference'],limit_px=args.movement_px,
                                  window=args.movement_window,persist_s=args.movement_persist))+'\n')
        wait_ready(camera,tracker)
        while True:
            raw = camera.read()
            pose = tracker.estimate(raw)
            now=time.monotonic()
            info=guard.update(pose,now)
            debug=tracker.last_diagnostics
            info.update(time=now,marker_ids=debug.known_ids,marker_count=debug.used_marker_count,
                        pose_rms_px=debug.rms_error_px,rejection=debug.rejection_reason)
            log.write(json.dumps(info)+'\n')
            if now-last_log>=.5 or info['stop']:
                print(json.dumps(info),flush=True)
                log.flush()
                last_log=now
            if info['stop']:
                projector.black()
                raise ValueError(info['status']+'; projection stopped. Inspect motion log; calibration unchanged.')
            ready=info['ready']
            if ready:
                cv2.imshow(projector.name,frame)
            else:
                projector.black()
            lines=[info['status'],f"IDs {debug.known_ids}; pose RMS {debug.rms_error_px}",
                   f"Raw drift {info['raw_drift_px']}; filtered {info['filtered_drift_px']}; limit {args.movement_px} px",
                   f"Window jitter {info['jitter_px']}; baseline drift {info['baseline_drift_px']}"]
            show_preview(raw,f'{args.part_id} | '+info['status'],debug_lines=lines)
            if cv2.waitKey(1) & 0xff in (27,ord('q')):
                return


if __name__ == '__main__':
    run_cli(main)
