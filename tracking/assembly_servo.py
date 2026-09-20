"""Optional head centring using the assembly program's existing camera frames."""
import time
import numpy as np
import config
from tracking.controller import FollowController, board_center_pixel


def add_servo_arguments(parser):
    parser.add_argument('--servo-port', help='Enable optional USB head centring (e.g. COM5). F starts/pauses.')
    parser.add_argument('--servo-pan-sign', type=int, choices=(-1,1))
    parser.add_argument('--servo-tilt-sign', type=int, choices=(-1,1))
    parser.add_argument('--servo-pan-only', action='store_true')
    parser.add_argument('--servo-step', type=int, default=2, help='Maximum correction in us; initial test default 2.')
    parser.add_argument('--servo-interval', type=float, default=.3)
    parser.add_argument('--servo-aim', type=float, nargs=2, help='Aim point in RAW camera pixels.')
    parser.add_argument('--servo-pan-limits', type=int, nargs=2, default=(400,2700))
    parser.add_argument('--servo-tilt-limits', type=int, nargs=2, default=(700,1500))


def servo_options(parser, args):
    if not args.servo_port:
        return None
    if args.servo_pan_sign is None or (not args.servo_pan_only and args.servo_tilt_sign is None):
        parser.error('--servo-port requires --servo-pan-sign and --servo-tilt-sign (unless --servo-pan-only).')
    return dict(port=args.servo_port, pan_sign=args.servo_pan_sign,
                tilt_sign=args.servo_tilt_sign or 1, pan_only=args.servo_pan_only,
                max_step=args.servo_step, interval=args.servo_interval,
                aim=args.servo_aim, pan_limits=args.servo_pan_limits, tilt_limits=args.servo_tilt_limits)


class AssemblyServo:
    def __init__(self, link, K, dist, *, aim=None, clock=time.monotonic, **options):
        self.link, self.K, self.dist, self.clock = link, K, dist, clock
        self.aim = np.asarray(aim if aim is not None else np.array(config.CAMERA_SIZE)/2, float)
        if self.aim.shape != (2,) or not np.isfinite(self.aim).all() or np.any(self.aim<0) or np.any(self.aim>=config.CAMERA_SIZE):
            raise ValueError('Servo aim must be inside the RAW camera image.')
        self.control = FollowController(pan=link.positions[0], tilt=link.positions[1], **options)
        self.active = False
        self.centered_since = None
        self.hold_until = -float('inf')
        self.status = 'Servos paused; F to centre before baseline / during move phase'

    def toggle(self, allowed, pose, before_move):
        if self.active:
            self.active = False
            self.control.reset()
            self.centered_since = None
            self.status = 'Servos paused - holding position'
            return
        if not allowed or pose is None:
            self.status = 'Cannot start: need board pose and a phase without a placement baseline'
            return
        before_move()
        # First enable uses reported commands, never a fabricated current angle.
        self.link.move('P', self.control.pan)
        if not self.control.pan_only:
            self.link.move('T', self.control.tilt)
        self.hold_until = self.clock() + config.SETTLE_SECONDS
        self.active = True
        self.control.reset()
        self.centered_since = None
        self.status = 'Servo centring started'

    def update(self, pose, allowed, before_move):
        """False blocks capture/check/advance until a fresh centred window exists."""
        if not self.active:
            ready = self.clock() >= self.hold_until
            if not ready:
                self.status = 'Paused - waiting for last correction to settle'
            return ready
        if not allowed:
            self.control.reset()
            self.centered_since = None
            self.status = 'Servos holding for placement; move phase permits centring'
            return True
        target = board_center_pixel(pose, self.K, self.dist, config.BOARD_BOUNDS_MM) if pose is not None else None
        if target is not None and (np.any(target<0) or np.any(target>=config.CAMERA_SIZE)):
            target = None
        now = self.clock()
        commands = self.control.update(target, self.aim, now)
        if commands:
            before_move()
            for axis, pulse in commands:
                self.link.move(axis, pulse)
            self.hold_until = self.clock() + config.SETTLE_SECONDS
        if self.control.status != 'CENTERED':
            self.centered_since = None
        elif self.centered_since is None:
            self.centered_since = now
        ready = (self.centered_since is not None and now-self.centered_since >= config.SETTLE_SECONDS
                 and now >= self.hold_until)
        self.status = self.control.status + (' - ready for Space' if ready else ' - wait; F pauses')
        return ready
