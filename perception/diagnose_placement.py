"""Dump everything one placement check saw, so a low score can be explained.

    python -m perception.diagnose_placement --cad Fusion_output \
        --part-id "389423 Bright Blue Technic Brick 1 x 6 with Holes" --base-z 19.2

Space captures the baseline, then place the part and press Space again. Every
intermediate image is written to data/diagnose_<timestamp>/ with the scores, so
the failure can be read off the pictures instead of guessed at. Q/Esc exits.
This only observes: no calibration, projector, or saved state is touched.
"""
import argparse
import datetime as _dt
import json

import cv2
import numpy as np

import config
from calibration.common import load_camera
from hardware.camera import Camera
from perception.aruco import BoardTracker
from perception.change_detector import _changed_mask, detect_change
from perception.pipeline import marker_quads


def _stats(name, mask):
    count = int(np.count_nonzero(mask))
    n, _, boxes, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    sizes = sorted((int(boxes[i, cv2.CC_STAT_AREA]) for i in range(1, n)), reverse=True)
    return {'name': name, 'pixels': count, 'fragments': len(sizes),
            'fragment_sizes': sizes[:8]}


def run(cad, part_id, base_z=0.0, camera_index=None):
    if camera_index is not None:
        config.CAMERA_INDEX = camera_index
    K, dist = load_camera()
    tracker = BoardTracker(K, dist)
    from perception.stl_matcher import load_models, verify
    models = load_models(cad)
    if part_id not in models:
        raise ValueError(f'Choose --part-id from: {list(models)}')

    out = config.DATA_DIR / ('diagnose_' + _dt.datetime.now().strftime('%Y%m%dT%H%M%S'))
    baseline = None
    report = {'part_id': part_id, 'base_z': base_z}
    print('Clear the board and press Space for the baseline. Q/Esc exits.')
    with Camera() as camera:
        while True:
            raw = camera.read()
            pose = tracker.estimate(raw)
            view = cv2.resize(raw, (960, 540))
            cv2.putText(view, 'baseline: ' + ('SET' if baseline is not None else 'none'),
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if baseline is not None else (0, 200, 255), 2)
            cv2.putText(view, 'board pose: ' + ('ok' if pose else 'LOST'), (12, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if pose else (0, 0, 255), 2)
            if config.PREVIEW_ROTATE_180:
                view = cv2.rotate(view, cv2.ROTATE_180)
            cv2.imshow('diagnose (Space=capture, Q=quit)', view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key != 32:
                continue
            if pose is None:
                print('Board pose lost; not capturing.')
                continue

            undistorted = cv2.undistort(raw, K, dist)
            if baseline is None:
                baseline = undistorted
                print('Baseline captured. Place the part, remove your hand, press Space.')
                continue

            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / '1_before.png'), baseline)
            cv2.imwrite(str(out / '2_after.png'), undistorted)
            cv2.imwrite(str(out / '3_absdiff.png'),
                        cv2.absdiff(baseline, undistorted).max(axis=2))
            quads = marker_quads(pose, K, dist)
            raw_mask = _changed_mask(baseline, undistorted)
            cv2.imwrite(str(out / '4_threshold_mask.png'), raw_mask)
            report['threshold_mask'] = _stats('threshold_mask', raw_mask)

            region = detect_change(baseline, undistorted, exclude_quads=quads)
            if region is None:
                report['region'] = None
                print('No change region found. See', out)
            else:
                cv2.imwrite(str(out / '5_region_mask.png'), region.mask)
                report['region'] = _stats('region_mask', region.mask)
                report['region']['bbox'] = list(map(int, region.bbox))
                report['region']['kept_fraction_of_threshold'] = round(
                    float(np.count_nonzero(region.mask)) /
                    max(1, np.count_nonzero(raw_mask)), 3)
                status, scores, debug = verify(models, part_id, region, pose, K,
                                               base_z=base_z)
                cv2.imwrite(str(out / '6_overlap.png'), debug)
                report['status'] = status
                report['scores'] = [{'part': n, 'overlap': round(float(s), 4),
                                     'yaw_deg': round(float(a), 1)} for n, s, a in scores]
                print(f'\n{status}')
                for row in report['scores']:
                    print('   %-46s overlap=%.3f yaw=%.0f'
                          % (row['part'][:44], row['overlap'], row['yaw_deg']))
                print('  kept %.0f%% of the thresholded change; %d fragment(s)'
                      % (100*report['region']['kept_fraction_of_threshold'],
                         report['region']['fragments']))
            (out / 'report.json').write_text(json.dumps(report, indent=2))
            print('Wrote', out)
            baseline = None
            print('\nBaseline cleared. Space to capture a new one.')
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cad', required=True)
    parser.add_argument('--part-id', required=True)
    parser.add_argument('--base-z', type=float, default=0.0,
                        help='Support height in mm for a stacked part (0 = on the board).')
    parser.add_argument('--camera-index', type=int, default=None)
    args = parser.parse_args()
    run(args.cad, args.part_id, args.base_z, args.camera_index)


if __name__ == '__main__':
    main()
