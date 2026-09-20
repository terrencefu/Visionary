"""Small immediate pulse corrections from RAW image error; no motion ramp."""
from dataclasses import dataclass
import numpy as np
import cv2


def board_center_pixel(pose, K, dist, bounds):
    x0, y0, x1, y1 = bounds
    point = np.array([[(x0+x1)/2, (y0+y1)/2, 0.]], dtype=float)
    camera_point = pose.R @ point[0] + pose.tvec.reshape(3)
    if not np.isfinite(camera_point).all() or camera_point[2] <= 0:
        return None
    pixel = cv2.projectPoints(point, pose.rvec, pose.tvec, K, dist)[0].reshape(2)
    return pixel if np.isfinite(pixel).all() else None


@dataclass
class FollowController:
    # Sign means: positive RAW image error requires increasing (+1) or
    # decreasing (-1) that axis pulse. Determine on the actual mounted rig.
    pan_sign: int = 1
    tilt_sign: int = 1
    gain: float = 0.04  # microseconds per pixel per update
    deadband: float = 25.
    max_step: int = 12
    interval: float = .15
    acquire_frames: int = 3
    pan_limits: tuple = (400, 2700)
    tilt_limits: tuple = (700, 1500)
    pan: int = 1500
    tilt: int = 1000
    pan_only: bool = False

    def __post_init__(self):
        values = [self.gain, self.deadband, self.max_step, self.interval, self.acquire_frames,
                  *self.pan_limits, *self.tilt_limits, self.pan, self.tilt]
        if (not np.isfinite(values).all() or min(values[:5]) <= 0
                or self.pan_sign not in (-1, 1) or self.tilt_sign not in (-1, 1)):
            raise ValueError('Invalid tracking parameters.')
        for limits, firmware, value in [(self.pan_limits, (400,2700), self.pan),
                                         (self.tilt_limits, (700,1500), self.tilt)]:
            if not firmware[0] <= limits[0] < limits[1] <= firmware[1] or not limits[0] <= value <= limits[1]:
                raise ValueError('Start pulses and limits must fit the Arduino ranges.')
        self.good_frames = 0
        self.last_command = -float('inf')
        self.status = 'PAUSED'

    def reset(self):
        self.good_frames = 0
        self.status = 'PAUSED'

    def update(self, target, aim, now):
        if target is None or not np.isfinite(target).all():
            self.good_frames = 0
            self.status = 'BOARD LOST - HOLD'
            return []
        self.good_frames += 1
        if self.good_frames < self.acquire_frames:
            self.status = 'ACQUIRING'
            return []
        if now - self.last_command < self.interval:
            self.status = 'WAITING FOR NEXT FRAME'
            return []
        error = np.asarray(target) - np.asarray(aim)
        commands = []
        limited = False
        for axis, e, sign, limits in [('P', error[0], self.pan_sign, self.pan_limits),
                                       ('T', error[1], self.tilt_sign, self.tilt_limits)]:
            if (axis == 'T' and self.pan_only) or abs(e) <= self.deadband:
                continue
            # Deadband removed from error prevents a jump at its boundary.
            delta = sign * np.sign(e) * min(self.max_step, max(1, round(self.gain*(abs(e)-self.deadband))))
            current = self.pan if axis == 'P' else self.tilt
            requested = current + int(delta)
            pulse = int(np.clip(requested, *limits))
            limited |= pulse != requested
            if pulse != current:
                commands.append((axis, pulse))
                if axis == 'P':
                    self.pan = pulse
                else:
                    self.tilt = pulse
        if commands:
            self.last_command = now
        self.status = 'LIMIT REACHED' if limited else ('CORRECTING' if commands else 'CENTERED')
        return commands
