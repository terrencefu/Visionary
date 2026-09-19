# Stationary planar guidance

## Read-only board alignment overlay

```powershell
python main.py planar validate-board --stationary-ready
```

Press Space when READY. Red connects marker centers in order 0–1–3–2 with
labelled 3 mm half-width crosshairs; green outlines `BOARD_BOUNDS_MM`. The projected
legend says "Marker-center rectangle (not cardboard edge)"; the measured 1 mm
asymmetry is preserved. These are not the cardboard's outer corners.
Every geometric endpoint uses the saved homography, with no refitting or updates
to the calibration file. Marker locations intentionally extrapolate beyond the
calibration hull for this visual test only. Off-screen portions are clipped by
the full-resolution projector canvas, not resized to fit.

Temporary tracking loss blanks the projection and allows two seconds to recover.
Three consecutive reliable poses exceeding the existing movement limit stop the
test; recalibrate before reuse. Escape/Q exits. This command neither runs nor
changes the separate physical dot-validation workflow, and does not certify
physical accuracy or detect independent projector movement.

This optional path does not load the 3D projector calibration. It uses the existing
camera intrinsics, RAW ArUco detection, green-dot measurement, projector display,
grid pixel ranges and usable board bounds. Preview rotation affects display only.

Keep the board, camera and projector stationary. Visually confirm the calibration
grid lands on flat cardboard. The entire projector image need not fit on the board.

```powershell
python main.py planar calibrate --stationary-ready
python main.py planar validate --stationary-ready
python main.py planar render --stationary-ready --target 150 100
python main.py planar render --stationary-ready --outline outline.json
python main.py planar render --stationary-ready --overlay guidance.png
```

Press Space when the board is READY to start each command. Escape/Q exits.
After any movement or projector zoom/keystone/resolution change, recalibrate:

```powershell
python main.py planar calibrate --stationary-ready --overwrite
```

An outline file contains a closed polygon's board XY vertices in millimetres, e.g.
`[[100,80],[180,80],[180,130],[100,130]]`. Vertices must lie within calibrated support.
An overlay image spans `BOARD_BOUNDS_MM`: its top-left is minimum X/Y, +X is right,
+Y is down, and its bottom-right is maximum X/Y. Black means no projected light.
Overlay content is clipped to the inlier convex hull. The Python API is in
`projection/planar.py`: `board_to_pixel`, `transform`, and `warp_overlay`.

The collector preserves the existing dot contour, marker overlap, board bounds,
pose-quality and movement checks. RANSAC estimates board-mm to projector-pixel H;
all measurements and inlier/residual diagnostics are retained. At least four
inliers must cover 20% of the usable board area and span 40% of each axis. Prefer
8–12 or more; four points cannot independently establish outlier robustness.
Targets outside the accepted inlier hull are rejected rather than extrapolated.

Calibration automatically measures five held-out targets, at least 5 mm from all
fitting observations, and reports each landing error plus RMS and maximum in mm.
These observations never update H. Rendering repeats this validation and requires
all five errors <= `PLANAR_VALIDATION_MAX_MM` (default 5 mm). This is a separate
planar setting; the existing 3D accuracy gates are unchanged. Errors are measured
through the same camera/ArUco model, so they do not independently validate that model.

`planar_calibration.json` contains H, source rows, pose reference, signatures,
inlier hull, pixel residuals, status and physical validation history. Raw accepted
correspondences are also saved under `data/planar_samples_*.json`, even if fitting
fails. This is separate from `projector_calibration.npz` and its diagnostics.

During rendering, missing/unreliable tracking blanks output; estimated movement
above the existing drift limit marks the mapping stale and requires recalibration.
ArUco cannot detect a projector moving independently of the camera. The explicit
stationary confirmation and pre-render physical checks do not replace a rigid
mount: if anything moves during guidance, stop and recalibrate. A homography also
cannot correct non-flat cardboard, inaccurate ArUco geometry or all lens distortion.
