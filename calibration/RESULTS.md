# Calibration and fixture diagnostic results

Captured on 2026-09-19. These results belong to the current webcam and fixture; they are not a generic camera calibration.

## Camera intrinsics

- File: `camera_calibration_1080p.npz` at the repository root.
- Camera: external webcam AC310, currently DirectShow index 0.
- Raw resolution: 1920 x 1080; preview rotation is not applied to calibration inputs.
- Chessboard: 9 x 6 inner corners, configured 25 mm square size.
- Accepted calibration views: 30.
- Calibration RMS: **0.443699 px**.
- Archive includes the camera matrix, distortion coefficients, image size, per-view residuals, and calibration point observations.

## Remeasured fixture

User-confirmed black-square side: 30 mm. Centers in mm:

| ID | X | Y |
|---|---:|---:|
| 0 | 0 | 0 |
| 1 | 493 | 0 |
| 2 | 0 | 305 |
| 3 | 493 | 305 |

The production pose acceptance threshold remains **2.0 px RMS**. Board bounds have not been configured.

## Saved failing-frame diagnostics

Numeric reports are committed at `data/aruco_debug/report.json` and `data/aruco_pose_comparison/report.json`. Raw/annotated camera images remain local and are not included in this commit. Paths recorded inside the reports refer to the original development machine.

All four IDs were detected and the individual pose solvers succeeded:

| ID | Individual RMS (px) | Residual using the 0/1/3 pose (px RMS) |
|---|---:|---:|
| 0 | 0.134 | 5.831 |
| 1 | 0.234 | 4.352 |
| 2 | 1.997 | 171.811 (held out) |
| 3 | 1.985 | 6.213 |

Full-board RMS: **34.977 px**. Diagnostic 0/1/3-only RMS: **5.524 px**. Both combined fits fail the unchanged 2 px gate. Individual fits for markers 2 and 3 only narrowly pass it in this frame.

This supports an inconsistency in the shared board geometry/pose fit and/or lens model. Excluding marker 2 improves the fit but does not resolve it. No gross quarter-turn corner-order mismatch was indicated. The data does not prove cardboard lift is the sole cause, and low individual-marker residuals do not independently validate intrinsics or absolute pose.

Run `python main.py aruco --diagnose-poses` to capture a current comparison. This diagnostic does not replace the production board pose or change calibration, dimensions, or thresholds.

## Validation status

The code passed 116 tests, including physical-rotation/canonical-corner tests, synthetic nonplanarity diagnostics, and preservation of the unrotated debug image. Projector calibration and physical world-lock validation are still pending; the failing fixture reports must not be treated as successful system calibration.
