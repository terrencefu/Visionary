"""Read-only visual alignment test using the saved planar homography."""
import time
import cv2
import numpy as np
import config
from hardware.camera import show_preview
from perception.aruco import board_drift_px
from projection.planar import transform


def alignment_canvas(mapping):
    width, height = config.PROJECTOR_SIZE
    canvas = np.zeros((height, width, 3), np.uint8)
    H = np.asarray(mapping['H'], float)
    centers = config.MARKER_CENTERS_MM
    marker_color, bounds_color = (0, 0, 255), (0, 255, 0)

    def pixels(points):
        # Intentional extrapolation for this visual test only: marker centers
        # lie outside the calibration hull. Never change H or clip to that hull.
        points = np.asarray(points, float)
        denominators = np.c_[points, np.ones(len(points))] @ H[2]
        if np.any(denominators > 0) and np.any(denominators < 0):
            raise ValueError('Alignment geometry crosses the homography horizon.')
        uv = transform(H, points)
        if np.max(np.abs(uv)) > 1e7:
            raise ValueError('Alignment projects too far outside the projector canvas.')
        return np.rint(uv).astype(np.int32)

    def segment(a, b, color):
        p, q = pixels([a, b])
        cv2.line(canvas, tuple(p), tuple(q), color, 3, cv2.LINE_AA)

    def outline(points, color):
        pixels(points)  # Also reject a horizon crossing anywhere in the polygon.
        for a, b in zip(points, points[1:] + points[:1]):
            segment(a, b, color)

    outline([centers[i] for i in (0, 1, 3, 2)], marker_color)
    for i in (0, 1, 2, 3):
        x, y = centers[i]
        segment((x-3, y), (x+3, y), marker_color)
        segment((x, y-3), (x, y+3), marker_color)
        anchor = pixels([(x+4, y+4)])[0]
        cv2.putText(canvas, f'ID {i}', tuple(anchor), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, marker_color, 2, cv2.LINE_AA)
    x0, y0, x1, y1 = config.BOARD_BOUNDS_MM
    outline([(x0,y0), (x1,y0), (x1,y1), (x0,y1)], bounds_color)
    # Legend is projector UI text, not board geometry. All geometry above uses H.
    cv2.putText(canvas, 'Marker-center rectangle (not cardboard edge)', (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, .6, marker_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, 'Usable workspace', (20,55), cv2.FONT_HERSHEY_SIMPLEX,
                .6, bounds_color, 2, cv2.LINE_AA)
    return canvas


class RecoveryGuard:
    """Blank on uncertain tracking; allow two seconds to recover, confirm drift."""
    def __init__(self):
        self.missing_since = None
        self.drift_frames = 0

    def update(self, drift, now):
        if drift is None or not np.isfinite(drift):
            self.drift_frames = 0
            if self.missing_since is None:
                self.missing_since = now
            if now-self.missing_since >= 2.0:
                raise ValueError('ArUco tracking did not recover within 2 seconds; projection stopped.')
            return False
        self.missing_since = None
        if drift > config.MAX_BOARD_DRIFT_PX:
            self.drift_frames += 1
            if self.drift_frames >= 3:
                raise ValueError(f'Movement confirmed across 3 poses ({drift:.2f} px drift); stop and recalibrate.')
            return False
        self.drift_frames = 0
        return True


def validate_board(record, reference, camera, projector, tracker):
    frame = alignment_canvas(record['mapping'])
    guard = RecoveryGuard()
    print('VISUAL ONLY: red marker-center rectangle; green usable workspace. Not cardboard edges.')
    print('Marker centers may extrapolate beyond fitted samples; off-screen segments are clipped, never scaled.')
    print('Tracking loss blanks projection for up to 2 seconds. Esc/Q exits. Calibration is read-only.')
    while True:
        raw = camera.read()
        pose = tracker.estimate(raw)
        drift = None if pose is None else board_drift_px(reference, pose, tracker.K, tracker.dist)
        try:
            ready = guard.update(drift, time.monotonic())
        except ValueError:
            projector.black()
            raise
        if ready:
            cv2.imshow(projector.name, frame)
        else:
            projector.black()
        show_preview(raw, 'Marker-center rectangle | ' + ('ALIGNMENT TEST' if ready else 'Tracking recovery / checking movement'))
        if cv2.waitKey(1) & 0xff in (27, ord('q')):
            return
