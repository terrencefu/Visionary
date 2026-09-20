"""Optional stationary planar calibration, physical validation and guidance."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import config
from calibration.common import camera_signature, fixture_signature, load_camera, run_cli
from calibration.measurement import measure_dot
from hardware.camera import Camera, show_preview
from hardware.projector import Projector
from perception.aruco import BoardTracker, board_drift_px
from projection.planar import fit_mapping, board_to_pixel, warp_overlay, held_out_targets


def write_record(record):
    config.PLANAR_CALIBRATION.write_text(json.dumps(record, indent=2), encoding='utf-8')


def invalidate(record, reason):
    record['status'] = 'STALE - RECALIBRATE'
    record['invalidated_reason'] = reason
    write_record(record)
    raise ValueError(reason + '; recalibrate planar mapping at the current position.')


def check_stationary(record, reference, pose, tracker):
    if pose is None:
        raise ValueError('No reliable current ArUco pose; cannot verify stationary mapping.')
    drift = board_drift_px(reference, pose, tracker.K, tracker.dist)
    if not np.isfinite(drift) or drift > config.MAX_BOARD_DRIFT_PX:
        invalidate(record, f'Estimated board/rig drift {drift:.2f} px exceeds {config.MAX_BOARD_DRIFT_PX} px')


def wait_ready(camera, tracker):
    while True:
        raw = camera.read()
        pose = tracker.estimate(raw)
        show_preview(raw, f"PLANAR | {'READY' if pose else 'NOT READY'} | Space: start | Esc: exit")
        key = cv2.waitKey(1) & 0xff
        if key in (27, ord('q')):
            raise KeyboardInterrupt
        if key == 32 and pose is not None:
            return pose


def validate(record, reference, camera, projector, tracker):
    # Never use these observations to update H.
    measurements = []
    for xy in held_out_targets(record['mapping']):
        try:
            check_stationary(record, reference, tracker.estimate(camera.read()), tracker)
            row, view, _ = measure_dot(camera, projector, tracker,
                                       *board_to_pixel(record['mapping'], xy), reference=reference)
            observed = np.array(row['board_xyz'][:2])
            error = float(np.linalg.norm(observed-xy))
            measurements.append(dict(target_xy=xy.tolist(), observed_xy=observed.tolist(), error_mm=error))
            print(f'Held-out {xy.round(2)} -> {observed.round(2)} mm; landing error {error:.3f} mm')
            show_preview(view, f'PLANAR held-out error {error:.2f} mm')
        except ValueError as exc:
            measurements.append(dict(target_xy=xy.tolist(), rejection=str(exc)))
            print(f'Held-out target rejected: {exc}')
            if record['status'].startswith('STALE'):
                break
    errors = [m['error_mm'] for m in measurements if 'error_mm' in m]
    passed = (len(errors) == 5 and max(errors) <= config.PLANAR_VALIDATION_MAX_MM
              and not record['status'].startswith('STALE'))
    result = dict(timestamp=datetime.now(timezone.utc).isoformat(), measurements=measurements,
                  passed=passed, max_allowed_mm=config.PLANAR_VALIDATION_MAX_MM,
                  rms_mm=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                  max_mm=max(errors) if errors else None)
    record.setdefault('validation_history', []).append(result)
    if not record['status'].startswith('STALE'):
        record['status'] = 'VALIDATED AT RECORDED POSITION' if passed else 'UNVALIDATED'
    write_record(record)
    print(f"Held-out validation: {len(errors)}/5 measured; RMS={result['rms_mm']} mm; max={result['max_mm']} mm; pass={passed}")
    print('Errors use the current camera/ArUco reconstruction; they are not independent tape-measure accuracy.')
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['calibrate', 'validate', 'render', 'validate-board', 'verify-current'])
    parser.add_argument('--stationary-ready', required=True, action='store_true',
                        help='Confirm unchanged rigid mount, board and projector settings; recalibrate after any movement.')
    parser.add_argument('--target', nargs=2, type=float, metavar=('X_MM', 'Y_MM'))
    parser.add_argument('--outline', type=Path, help='JSON list of board XY vertices for a closed outline.')
    parser.add_argument('--overlay', type=Path, help='Image spanning BOARD_BOUNDS_MM, +X right and +Y down.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dot-color', choices=['green','magenta'], default=None)
    args = parser.parse_args()
    if sum(x is not None for x in (args.target, args.outline, args.overlay)) > 1:
        raise ValueError('Choose only one of --target, --outline, --overlay.')
    config.validate_fixture()
    if args.action == 'verify-current':
        from calibration.planar_verify import verify_current
        verify_current(dot_color=args.dot_color)
        return
    tracker = BoardTracker(*load_camera(), max_rms_px=config.COLLECTOR_MAX_ARUCO_RMS_PX)
    record = None
    if args.action == 'calibrate':
        if config.PLANAR_CALIBRATION.exists() and not args.overwrite:
            raise ValueError('Planar calibration exists; use --overwrite after moving the rig.')
    else:
        record = json.loads(config.PLANAR_CALIBRATION.read_text(encoding='utf-8'))
        if (record['camera_signature'] != camera_signature() or record['fixture_signature'] != fixture_signature()
                or record['projector_size'] != list(config.PROJECTOR_SIZE)
                or record['status'].startswith('STALE')):
            raise ValueError('Planar calibration is stale/incompatible; recalibrate.')
    print('PLANAR: only valid while projector, camera, board and projector settings remain unchanged.')
    config.DOT_COLOR = args.dot_color or (record.get('dot_color','green') if record else config.DOT_COLOR)
    print(f'Calibration dot color: {config.DOT_COLOR}')
    print('Visually verify all calibration dots land on flat cardboard. Move anything -> recalibrate.')
    with Projector() as projector, Camera() as camera:
        current = wait_ready(camera, tracker)
        if args.action == 'calibrate':
            rows = []
            for v in np.rint(np.linspace(*config.GRID_V_RANGE_PX, config.GRID_ROWS)).astype(int):
                for u in np.rint(np.linspace(*config.GRID_U_RANGE_PX, config.GRID_COLS)).astype(int):
                    try:
                        row, view, _ = measure_dot(camera, projector, tracker, u, v, reference=current)
                        rows.append(row)
                        print(f'Accepted ({u},{v}) -> {np.round(row["board_xyz"][:2], 2)} mm')
                        show_preview(view, f'PLANAR accepted {len(rows)}')
                    except ValueError as exc:
                        print(f'Skipped ({u},{v}): {exc}')
            # Retain measurements even when the robust fit rejects their coverage.
            config.DATA_DIR.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
            (config.DATA_DIR / f'planar_samples_{stamp}.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            mapping = fit_mapping([r['board_xyz'][:2] for r in rows], [r['projector_uv'] for r in rows])
            record = dict(mapping=mapping, rows=rows, status='UNVALIDATED', dot_color=config.DOT_COLOR,
                          timestamp=stamp, camera_signature=camera_signature(), fixture_signature=fixture_signature(),
                          projector_size=list(config.PROJECTOR_SIZE),
                          reference=dict(rvec=current.rvec.tolist(), tvec=current.tvec.tolist(),
                                         object_points=current.object_points.tolist()))
            print(f"Candidate planar H: {sum(mapping['inliers'])}/{len(rows)} inliers; RMS {mapping['rms_px']:.3f} px; coverage {mapping['coverage']:.0%}")
            if sum(mapping['inliers']) < 8:
                print('Only 4-7 inliers: prefer 8-12+ distributed observations.')
            from calibration.planar_verify import landing_checks
            checks=landing_checks(mapping,current,camera,projector,tracker)
            record['validation_history']=[checks]
            (config.DATA_DIR / f'planar_candidate_{stamp}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
            if not checks['passed']:
                raise ValueError('Candidate failed held-out validation; saved homography was not overwritten.')
            record['status']='VALIDATED AT RECORDED POSITION'
            if config.PLANAR_CALIBRATION.exists():
                (config.DATA_DIR / f'planar_previous_{stamp}.json').write_bytes(config.PLANAR_CALIBRATION.read_bytes())
            write_record(record)
            print(f'Saved validated planar calibration using {config.DOT_COLOR} dots.')
            return
        reference = SimpleNamespace(**{k: np.asarray(v) for k, v in record['reference'].items()})
        if args.action == 'validate-board':
            from calibration.planar_board import validate_board
            validate_board(record, reference, camera, projector, tracker)
            return  # Read-only: never run dot validation or write calibration status.
        check_stationary(record, reference, current, tracker)
        passed = validate(record, reference, camera, projector, tracker)
        if args.action != 'render':
            return
        if not passed:
            raise ValueError('Held-out validation failed; guidance not rendered. Inspect/recalibrate.')
        frame = np.zeros((projector.height, projector.width, 3), np.uint8)
        if args.overlay:
            overlay = cv2.imread(str(args.overlay))
            if overlay is None:
                raise ValueError('Cannot read overlay image.')
            frame = warp_overlay(record['mapping'], overlay)
        elif args.outline:
            points = json.loads(args.outline.read_text(encoding='utf-8'))
            if len(points) < 3:
                raise ValueError('Outline requires at least three XY vertices.')
            uv = np.array([board_to_pixel(record['mapping'], p) for p in points])
            cv2.polylines(frame, [np.rint(uv).astype(np.int32)], True, (0,255,0), 2)
        else:
            xy = args.target or np.mean(record['mapping']['hull'], axis=0).tolist()
            uv = board_to_pixel(record['mapping'], xy)
            cv2.circle(frame, tuple(np.rint(uv).astype(int)), 5, (0,255,0), -1)
        print('Guidance active. Esc/Q exits. Movement invalidates this calibration.')
        while True:
            raw = camera.read()  # Detection always uses RAW, never the rotated preview.
            try:
                check_stationary(record, reference, tracker.estimate(raw), tracker)
            except ValueError:
                projector.black()
                raise
            cv2.imshow(projector.name, frame)
            show_preview(raw, 'PLANAR guidance | stationary only | Esc: exit')
            if cv2.waitKey(1) & 0xff in (27, ord('q')):
                break


if __name__ == '__main__':
    run_cli(main)
