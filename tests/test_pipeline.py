"""Hardware-free checks for coordinate conventions and calibration failure modes."""
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import config
from calibration.dataset import rotation_difference_deg
from calibration.solve_projector_calibration import solve, quality_failures
from perception.geometry import camera_pixel_to_board, pose_matrix
from perception.green_dot import detect_green_dot
from perception.aruco import BoardTracker, marker_corners
from projection.world import board_point_to_projector


class GeometryTests(unittest.TestCase):
    K = np.array([[1500., 0, 960], [0, 1490., 540], [0, 0, 1.]])
    dist = np.array([0.05, -0.02, 0.001, -0.001, 0.0])

    def test_raw_upside_down_camera_round_trip(self):
        r = np.array([0.2, -0.1, 3.0])
        t = np.array([70., 80., 950.])
        expected = np.array([145., 95., 0.])
        pixel = cv2.projectPoints(expected.reshape(1, 3), r, t, self.K, self.dist)[0].ravel()
        actual = camera_pixel_to_board(*pixel, r, t, self.K, self.dist)
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_intersection_rejects_parallel_and_behind(self):
        with self.assertRaises(ValueError):
            camera_pixel_to_board(960, 540, np.array([0., np.pi/2, 0.]), np.array([0., 0., 500.]), self.K, np.zeros(5))
        with self.assertRaises(ValueError):
            camera_pixel_to_board(960, 540, np.zeros(3), np.array([0., 0., -500.]), self.K, np.zeros(5))

    def test_board_to_projector_composition(self):
        rb, tb = np.array([0.1, -0.2, 0.3]), np.array([10., -30., 800.])
        Tpc = pose_matrix([0.01, -0.03, 0.02], [100., 5., 20.])
        point = np.array([130., 70., 0.])
        actual = board_point_to_projector(point, rb, tb, self.K, self.dist, Tpc)
        camera_point = (pose_matrix(rb, tb) @ np.r_[point, 1])[:3]
        projector_point = (Tpc @ np.r_[camera_point, 1])[:3]
        expected = cv2.projectPoints(projector_point.reshape(1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)[0].ravel()
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_multimarker_pose_and_unknown_id(self):
        centers = {0: (0, 0), 1: (320, 0), 2: (0, 240), 3: (320, 240)}
        with patch.multiple(config, MARKER_SIZE_MM=30., MARKER_CENTERS_MM=centers, BOARD_BOUNDS_MM=(-30, -30, 350, 270)):
            tracker = BoardTracker(self.K, self.dist)
            expected = pose_matrix([0.2, -0.1, 0.1], [-100., -100., 1000.])
            rvec = cv2.Rodrigues(expected[:3, :3])[0]
            corners = [cv2.projectPoints(marker_corners(i), rvec, expected[:3, 3], self.K, self.dist)[0].reshape(1, 4, 2).astype(np.float32) for i in centers]
            class FakeDetector:
                def detectMarkers(self, raw):
                    return corners + [corners[0]], np.array([[0], [1], [2], [3], [49]]), []
            tracker.detector = FakeDetector()
            pose = tracker.estimate(np.zeros((1080, 1920, 3), np.uint8))
            self.assertIsNotNone(pose)
            self.assertEqual(pose.visible_ids, [0, 1, 2, 3])
            np.testing.assert_allclose(pose_matrix(pose.rvec, pose.tvec), expected, atol=1e-3)

    def test_actual_aruco_detector_handles_upside_down_raw(self):
        from perception.aruco import make_detector
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        raw = np.full((500, 500), 255, np.uint8)
        raw[100:300, 150:350] = cv2.aruco.generateImageMarker(dictionary, 3, 200)
        raw = cv2.rotate(raw, cv2.ROTATE_180)
        _, ids, _ = make_detector().detectMarkers(raw)
        self.assertEqual(ids.ravel().tolist(), [3])


class DotTests(unittest.TestCase):
    def test_green_dot_ignores_white_exposure_change_and_static_green(self):
        background = np.zeros((300, 400, 3), np.uint8)
        cv2.circle(background, (60, 60), 15, (0, 200, 0), -1)
        lit = background.copy()
        cv2.rectangle(lit, (200, 0), (399, 299), (100, 100, 100), -1)
        cv2.circle(lit, (125, 170), 20, (0, 230, 0), -1)
        dot, _ = detect_green_dot(background, lit)
        self.assertIsNotNone(dot)
        np.testing.assert_allclose([dot.u, dot.v], [125, 170], atol=0.1)

    def test_rejects_ambiguous_blobs(self):
        background = np.zeros((300, 400, 3), np.uint8)
        lit = background.copy()
        for center in [(80, 80), (200, 200)]:
            cv2.circle(lit, center, 20, (0, 255, 0), -1)
        self.assertIsNone(detect_green_dot(background, lit)[0])


def synthetic_poses(nonrigid=False):
    Kp = np.array([[1250., 0, 960.], [0, 1270., 540.], [0, 0, 1.]])
    Tpc = pose_matrix([0.02, -0.08, 0.01], [-110., 8., 25.])
    objects = np.array([[x, y, 0] for y in np.linspace(-150, 150, 5)
                        for x in np.linspace(-220, 220, 6)], np.float32)
    poses = []
    for i, (rx, ry, z) in enumerate([(-.3, -.2, 700), (.3, .1, 950),
                                    (.1, -.3, 850), (-.2, .3, 1000),
                                    (.35, -.25, 780), (-.25, .25, 900),
                                    (.2, .3, 730), (-.3, -.1, 1100)]):
        Tcb = pose_matrix([rx, ry, .04*i], [i*8.-30, i*4.-20, z])
        transform = Tpc.copy()
        if nonrigid:
            transform[0, 3] += (-1 if i % 2 else 1)*35
        Tpb = transform @ Tcb
        uv = cv2.projectPoints(objects, cv2.Rodrigues(Tpb[:3, :3])[0], Tpb[:3, 3], Kp, np.zeros(5))[0].reshape(-1, 2)
        poses.append(dict(name=str(i), objects=objects.copy(), projector_uv=uv.astype(np.float32),
                          transforms=[Tcb.copy() for _ in objects]))
    return poses, Kp, Tpc


class SolverTests(unittest.TestCase):
    def test_recovers_known_rigid_calibration(self):
        poses, Kp, Tpc = synthetic_poses()
        result = solve(poses)
        self.assertEqual(quality_failures(result), [])
        self.assertLess(result["fixed_rms"], 0.01)
        np.testing.assert_allclose(result["camera_matrix"], Kp, atol=0.05)
        np.testing.assert_allclose(result["T_projector_camera"][:3, 3], Tpc[:3, 3], atol=0.1)
        self.assertLess(rotation_difference_deg(result["T_projector_camera"][:3, :3], Tpc[:3, :3]), 0.01)

    def test_rejects_nonrigid_calibration(self):
        poses, _, _ = synthetic_poses(nonrigid=True)
        self.assertTrue(quality_failures(solve(poses)))

    def test_rejects_insufficient_diversity(self):
        poses, _, _ = synthetic_poses()
        with self.assertRaisesRegex(ValueError, "diversity"):
            solve([poses[0]]*6)


if __name__ == "__main__":
    unittest.main()
