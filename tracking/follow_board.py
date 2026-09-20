"""Follow a selected ArUco board with the existing immediate-command firmware."""
import argparse
from contextlib import ExitStack
import time
import cv2
import numpy as np
import config
from calibration.common import load_camera, run_cli
from hardware.camera import Camera, show_preview
from hardware.servo import ServoLink
from perception.aruco import BoardTracker
from tracking.controller import FollowController, board_center_pixel


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', help='USB serial port, e.g. COM5 or /dev/cu.usbmodem...')
    parser.add_argument('--live', action='store_true', help='Enable serial control; default is preview only.')
    parser.add_argument('--pan-sign', type=int, choices=(-1,1))
    parser.add_argument('--tilt-sign', type=int, choices=(-1,1))
    parser.add_argument('--pan-only', action='store_true')
    parser.add_argument('--gain', type=float, default=.04)
    parser.add_argument('--deadband', type=float, default=25.)
    parser.add_argument('--max-step', type=int, default=12)
    parser.add_argument('--interval', type=float, default=.15)
    parser.add_argument('--pan-limits', type=int, nargs=2, default=(400,2700))
    parser.add_argument('--tilt-limits', type=int, nargs=2, default=(700,1500))
    parser.add_argument('--aim', type=float, nargs=2, help='Desired board center in RAW camera pixels.')
    parser.add_argument('--marker-ids', type=int, nargs=4,
                        help='IDs at the configured 0,1,2,3 positions; board must have the same measured geometry.')
    parser.add_argument('--project', action='store_true', help='Render live 3D board overlay after centering and settling.')
    args = parser.parse_args(argv)
    if args.live and (not args.port or args.pan_sign is None or (not args.pan_only and args.tilt_sign is None)):
        parser.error('--live requires --port, --pan-sign and (unless --pan-only) --tilt-sign.')
    if args.marker_ids and (len(set(args.marker_ids)) != 4 or any(i < 0 or i >= 50 for i in args.marker_ids)):
        parser.error('Use four distinct marker IDs in 0..49.')
    return args


def selected_centers(ids):
    if ids is None:
        return None
    if set(config.MARKER_CENTERS_MM) != {0,1,2,3}:
        raise ValueError('--marker-ids requires configured reference positions 0,1,2,3.')
    return {new: config.MARKER_CENTERS_MM[old] for old, new in enumerate(ids)}


def main():
    args = parse_args()
    config.validate_fixture()
    K, dist = load_camera()
    tracker = BoardTracker(K, dist, marker_centers=selected_centers(args.marker_ids))
    aim = np.array(args.aim if args.aim is not None else np.array(config.CAMERA_SIZE)/2, dtype=float)
    if not np.isfinite(aim).all() or np.any(aim < 0) or np.any(aim >= config.CAMERA_SIZE):
        raise ValueError('--aim must be inside the RAW camera image.')
    controller = FollowController(pan_sign=args.pan_sign or 1, tilt_sign=args.tilt_sign or 1,
                                  gain=args.gain, deadband=args.deadband, max_step=args.max_step,
                                  interval=args.interval, pan_limits=args.pan_limits,
                                  tilt_limits=args.tilt_limits, pan_only=args.pan_only,
                                  pan=int(np.clip(1500, *args.pan_limits)),
                                  tilt=int(np.clip(1000, *args.tilt_limits)))
    print('Space: start/pause. Q/Esc: quit and hold. O: disable outputs (support head).')
    print('Default preview sends no commands. Signs refer to RAW pixels, not the rotated preview.')
    print('On first start the servos enable at their Arduino-reported commanded positions; these are not measured angles.')
    with ExitStack() as stack:
        projector = None
        if args.project:
            from hardware.projector import Projector
            from projection.guidance import board_scene, render_scene
            from projection.world import load_projector
            calibration = load_projector()
            scene = board_scene()
            projector = stack.enter_context(Projector())
        camera = stack.enter_context(Camera())
        link = stack.enter_context(ServoLink(args.port)) if args.live else None
        if link:
            controller.pan, controller.tilt = link.positions
            controller.__post_init__()  # validate reported pulses against requested limits
        # Use imshow directly here; the projector helper also consumes a key event.
        blank = np.zeros((config.PROJECTOR_SIZE[1], config.PROJECTOR_SIZE[0], 3), np.uint8)
        def blank_projector():
            cv2.imshow(projector.name, blank)

        active = False
        centered_since = None
        while True:
            raw = camera.read()
            pose = tracker.estimate(raw)
            target = board_center_pixel(pose, K, dist, config.BOARD_BOUNDS_MM) if pose is not None else None
            if target is not None and (np.any(target < 0) or np.any(target >= config.CAMERA_SIZE)):
                target = None
            now = time.monotonic()
            commands = controller.update(target, aim, now) if active else []
            if commands and projector:
                blank_projector()  # never leave a stale overlay lit while moving
            for axis, pulse in commands:
                if link:
                    link.move(axis, pulse)
                print(f'{"SEND" if link else "PREVIEW"} {axis} {pulse}')
            centered = active and target is not None and controller.status == 'CENTERED'
            if not centered:
                centered_since = None
            elif centered_since is None:
                centered_since = now
            projection_status = ''
            if projector:
                if centered_since is not None and now - centered_since >= config.SETTLE_SECONDS:
                    try:
                        cv2.imshow(projector.name, render_scene(scene, pose, calibration))
                        projection_status = ' | overlay active'
                    except ValueError as exc:
                        blank_projector()
                        projection_status = ' | ' + str(exc)
                else:
                    blank_projector()
            view = raw.copy()
            cv2.drawMarker(view, tuple(aim.astype(int)), (255,255,0), cv2.MARKER_CROSS, 35, 2)
            if target is not None:
                cv2.circle(view, tuple(np.rint(target).astype(int)), 10, (0,255,0), 2)
                cv2.line(view, tuple(aim.astype(int)), tuple(np.rint(target).astype(int)), (0,255,0), 2)
            status = controller.status if active else 'PAUSED - Space to start'
            if target is None:
                status += ' | ' + (tracker.last_diagnostics.rejection_reason or 'Board center outside camera')
            show_preview(view, f'{"LIVE" if link else "PREVIEW"} {status} | P {controller.pan} T {controller.tilt}' + projection_status)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord('q')):
                break
            if key == ord('o'):
                active = False
                controller.reset()
                if link:
                    link.off()
            if key == 32:
                if active:
                    active = False
                    controller.reset()
                elif target is not None:
                    if link:
                        link.move('P', controller.pan)
                        if not args.pan_only:
                            link.move('T', controller.tilt)
                    controller.reset()
                    active = True


if __name__ == '__main__':
    run_cli(main)
