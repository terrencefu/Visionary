"""Sequential assembly guidance: one step at a time, advance only when correct.

The state machine owns the two pieces of temporal knowledge that make the
perception problem tractable:

  * the BASELINE frame, captured with the workspace as it was before this step,
    so the change detector only ever sees the one new part, and
  * the STILLNESS gate, so nothing is measured while the user's hand is still
    in shot.

Feed it every camera frame. It returns what the projector should show.
"""
from dataclasses import dataclass, field
from enum import Enum

import cv2

from perception import part_catalog
from perception.change_detector import StillnessGate
from perception.pipeline import detect_and_validate
from perception.validator import TOLERANCE_DEG, TOLERANCE_MM, correction_text


class Phase(Enum):
    WAITING = "waiting"              # settled, nothing new placed
    SETTLING = "settling"            # scene still moving; hand probably in shot
    CORRECTING = "correcting"        # right part, wrong pose
    WRONG_PART = "wrong_part"        # a different part was placed
    UNCLEAR = "unclear"              # something is there but not recognised
    STEP_COMPLETE = "step_complete"  # correct; the step has advanced
    COMPLETE = "complete"            # no steps left


@dataclass
class Step:
    part_id: str
    target_pose: dict                # {"x_mm", "y_mm", "theta_deg"} in board mm
    tol_mm: float | None = None
    tol_deg: float | None = None
    z_mm: float = 0.0                # height of the plane this part rests on
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        """Build from the CAD side's step dict. Extra keys are kept, not dropped."""
        known = {"part_id", "target_pose", "tol_mm", "tol_deg", "z_mm", "assembly_height"}
        target = dict(data["target_pose"])
        # The CAD side calls it assembly_height; a target pose may also carry z_mm.
        z = data.get("z_mm", data.get("assembly_height", target.get("z_mm", 0.0)))
        return cls(part_id=data["part_id"], target_pose=target,
                   tol_mm=data.get("tol_mm"), tol_deg=data.get("tol_deg"),
                   z_mm=float(z),
                   extra={k: v for k, v in data.items() if k not in known})


@dataclass
class StepUpdate:
    phase: Phase
    step_index: int
    step: Step | None
    guidance: str                    # one line for the projector to render
    result: object = None            # PerceptionResult, when one was computed


class AssemblyState:
    def __init__(self, steps, tol_mm=TOLERANCE_MM, tol_deg=TOLERANCE_DEG,
                 matcher=None, still_frames=None):
        steps = [s if isinstance(s, Step) else Step.from_dict(s) for s in steps]
        if not steps:
            raise ValueError("An assembly needs at least one step.")
        self.steps = steps
        self.index = 0
        self.tol_mm, self.tol_deg = tol_mm, tol_deg
        self.matcher = matcher
        self.baseline = None
        self._baseline_undistorted = None
        self.gate = StillnessGate(still_frames)

    # --- step bookkeeping -----------------------------------------------------

    def current_step(self):
        return None if self.is_complete else self.steps[self.index]

    @property
    def is_complete(self):
        return self.index >= len(self.steps)

    def advance(self):
        if not self.is_complete:
            self.index += 1

    def set_baseline(self, frame):
        """Capture the workspace as it is before this step's part is placed.

        Call this at SHOW_NEXT_STEP, before the user reaches in -- and with the
        projector showing whatever it will show during the step, since its
        light contaminates the camera image.
        """
        self.baseline = frame.copy()
        self._baseline_undistorted = None
        self.gate.reset()

    # --- per-frame ------------------------------------------------------------

    def update(self, frame, board_pose, camera_matrix, dist_coeffs):
        """Feed one camera frame; get back what to project."""
        if self.is_complete:
            return StepUpdate(Phase.COMPLETE, self.index, None, "Assembly complete")

        step = self.current_step()
        if self.baseline is None:
            self.set_baseline(frame)

        if not self.gate.update(frame):
            return StepUpdate(Phase.SETTLING, self.index, step, "Hold still...")

        # Undistort once per baseline, not once per frame: at 1080p that is
        # ~17 ms of pure waste on an image that has not changed.
        if self._baseline_undistorted is None:
            self._baseline_undistorted = cv2.undistort(self.baseline, camera_matrix, dist_coeffs)

        result = detect_and_validate(
            self._baseline_undistorted, cv2.undistort(frame, camera_matrix, dist_coeffs),
            step.part_id, board_pose, camera_matrix, dist_coeffs, already_undistorted=True,
            expected_pose=step.target_pose, matcher=self.matcher, z_mm=step.z_mm,
            tol_mm=step.tol_mm if step.tol_mm is not None else self.tol_mm,
            tol_deg=step.tol_deg if step.tol_deg is not None else self.tol_deg,
            symmetry_deg=getattr(part_catalog.CATALOG.get(step.part_id), "symmetry_deg", 360))

        return self._interpret(result, frame, step)

    def _interpret(self, result, frame, step):
        if not result.detected:
            if result.status == "no_change":
                return StepUpdate(Phase.WAITING, self.index, step,
                                  f"Place {step.part_id}", result)
            return StepUpdate(Phase.UNCLEAR, self.index, step, result.message, result)

        if result.status == "wrong_part":
            return StepUpdate(Phase.WRONG_PART, self.index, step,
                              f"Wrong part: that looks like {result.part_id}. "
                              f"Expected {step.part_id}.", result)

        if result.error is not None and result.error.correct:
            completed = self.index
            # The placed part belongs to the workspace now, or it would
            # re-trigger on every following step.
            self.set_baseline(frame)
            self.advance()
            phase = Phase.COMPLETE if self.is_complete else Phase.STEP_COMPLETE
            return StepUpdate(phase, completed, step,
                              f"{step.part_id} placed correctly", result)

        return StepUpdate(Phase.CORRECTING, self.index, step,
                          correction_text(result.error), result)
