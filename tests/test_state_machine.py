"""Sequential assembly: one step at a time, advancing only on a correct placement."""
import unittest

import cv2
import numpy as np

from assembly.state_machine import AssemblyState, Phase, Step
from perception import part_catalog
from tests.synthetic import blank_scene, render_part_on_board

K = np.array([[1500.0, 0, 960.0], [0, 1490.0, 540.0], [0, 0, 1.0]])
DIST = np.array([0.05, -0.02, 0.001, -0.001, 0.0])
SHAPE = (1080, 1920)
BGR = {"red": (40, 40, 210), "blue": (200, 70, 30)}


class FakeBoardPose:
    def __init__(self, rvec, tvec):
        self.rvec, self.tvec, self.image_points = rvec, tvec, None
        self.reprojection_error, self.visible_ids = 0.4, [0, 1, 2, 3]


def overhead_camera(look_at=(125.0, 95.0)):
    rvec = np.array([0.02, -0.03, np.pi])
    R = cv2.Rodrigues(rvec)[0]
    return FakeBoardPose(rvec, np.array([0.0, 0.0, 600.0]) - R @ np.array([*look_at, 0.0]))


BOARD = overhead_camera()
STEPS = [Step("red_l_plate", {"x_mm": 125.0, "y_mm": 95.0, "theta_deg": 0.0}),
         Step("blue_2x4", {"x_mm": 60.0, "y_mm": 140.0, "theta_deg": 90.0})]


def scene(placements=()):
    frame = blank_scene(SHAPE)
    for part_id, x, y, theta, colour in placements:
        render_part_on_board(frame, part_catalog.get(part_id).outline_mm, x, y, theta,
                             BOARD.rvec, BOARD.tvec, K, DIST, BGR[colour])
    return frame


def settle(machine, frame, times=8):
    """Feed the same frame until the stillness gate is satisfied, and return the
    first decisive update. Completing a step resets the gate, so simply taking
    the last update would read SETTLING again."""
    update = None
    for _ in range(times):
        update = machine.update(frame, BOARD, K, DIST)
        if update.phase is not Phase.SETTLING:
            return update
    return update


class StepSequencingTests(unittest.TestCase):
    def setUp(self):
        self.machine = AssemblyState(STEPS, tol_mm=3.0, tol_deg=8.0)

    def test_starts_on_the_first_step(self):
        self.assertEqual(self.machine.current_step().part_id, "red_l_plate")
        self.assertEqual(self.machine.index, 0)
        self.assertFalse(self.machine.is_complete)

    def test_advance_moves_to_the_next_step(self):
        self.machine.advance()
        self.assertEqual(self.machine.current_step().part_id, "blue_2x4")

    def test_advancing_past_the_last_step_completes_the_assembly(self):
        self.machine.advance()
        self.machine.advance()
        self.assertTrue(self.machine.is_complete)
        self.assertIsNone(self.machine.current_step())

    def test_a_correct_placement_advances_the_step(self):
        empty = scene()
        self.machine.set_baseline(empty)
        settle(self.machine, empty)

        placed = scene([("red_l_plate", 125.0, 95.0, 0.0, "red")])
        update = settle(self.machine, placed)

        self.assertEqual(update.phase, Phase.STEP_COMPLETE)
        self.assertEqual(self.machine.index, 1)

    def test_a_misplacement_holds_the_step_and_returns_a_correction(self):
        empty = scene()
        self.machine.set_baseline(empty)
        settle(self.machine, empty)

        placed = scene([("red_l_plate", 110.0, 95.0, 0.0, "red")])
        update = settle(self.machine, placed)

        self.assertEqual(update.phase, Phase.CORRECTING)
        self.assertEqual(self.machine.index, 0, "must stay on the same step")
        self.assertAlmostEqual(update.result.error.dx_mm, 15.0, delta=2.0)
        self.assertIn("+X", update.guidance)

    def test_a_correction_after_a_misplacement_advances(self):
        empty = scene()
        self.machine.set_baseline(empty)
        settle(self.machine, empty)
        settle(self.machine, scene([("red_l_plate", 110.0, 95.0, 0.0, "red")]))

        update = settle(self.machine, scene([("red_l_plate", 125.0, 95.0, 0.0, "red")]))

        self.assertEqual(update.phase, Phase.STEP_COMPLETE)
        self.assertEqual(self.machine.index, 1)

    def test_the_wrong_part_holds_the_step(self):
        empty = scene()
        self.machine.set_baseline(empty)
        settle(self.machine, empty)

        update = settle(self.machine, scene([("blue_2x4", 125.0, 95.0, 0.0, "blue")]))

        self.assertEqual(update.phase, Phase.WRONG_PART)
        self.assertEqual(self.machine.index, 0)
        self.assertIn("blue_2x4", update.guidance)


class HandAndMotionTests(unittest.TestCase):
    def setUp(self):
        self.machine = AssemblyState(STEPS, tol_mm=3.0, tol_deg=8.0)
        self.empty = scene()
        self.machine.set_baseline(self.empty)
        settle(self.machine, self.empty)

    def test_nothing_is_judged_while_the_scene_is_moving(self):
        """The user's hand is still in shot: no validation may happen yet."""
        moving = scene([("red_l_plate", 125.0, 95.0, 0.0, "red")])
        update = self.machine.update(moving, BOARD, K, DIST)

        self.assertEqual(update.phase, Phase.SETTLING)
        self.assertIsNone(update.result)
        self.assertEqual(self.machine.index, 0)

    def test_an_empty_settled_scene_just_waits(self):
        update = settle(self.machine, self.empty)

        self.assertEqual(update.phase, Phase.WAITING)
        self.assertEqual(self.machine.index, 0)

    def test_the_baseline_advances_with_the_step(self):
        """After a step completes, the placed part must become part of the
        baseline, or it re-triggers the next step forever."""
        settle(self.machine, scene([("red_l_plate", 125.0, 95.0, 0.0, "red")]))

        update = settle(self.machine, scene([("red_l_plate", 125.0, 95.0, 0.0, "red")]))

        self.assertEqual(self.machine.index, 1)
        self.assertEqual(update.phase, Phase.WAITING)


class StepDefinitionTests(unittest.TestCase):
    def test_a_step_can_be_built_from_a_plain_dict(self):
        step = Step.from_dict({"part_id": "blue_2x4",
                               "target_pose": {"x_mm": 1.0, "y_mm": 2.0, "theta_deg": 3.0}})
        self.assertEqual(step.part_id, "blue_2x4")
        self.assertEqual(step.target_pose["y_mm"], 2.0)

    def test_an_empty_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            AssemblyState([])


if __name__ == "__main__":
    unittest.main()
