# CAD-guided assembly copilot

Hackathon MVP: establish a measured ArUco world frame, calibrate a rigid camera/projector pair, then independently measure where projected guidance lands. Fusion, part perception, and servo control are later integrations.

## Current setup

- Python 3.11, OpenCV 4 with contrib/ArUco, NumPy, screeninfo.
- Camera index **0**, raw **1920Ã—1080**. The camera is upside down. Geometry always uses raw frames; only the annotated human preview rotates 180Â°.
- Projector: Windows **Extend**, **1920Ã—1080**, non-primary monitor. Keep Windows scaling at 100% on the projector and verify the smoke test fills the display.
- Marker dictionary **DICT_4X4_50**, black-square size **30 mm** (confirmed). All markers have the same physical orientation.
- Marker centers in `config.py` follow the user's remeasurement: 493 mm top/bottom and 305 mm left/right. IDs 0â€“3 are `(0,0)`, `(493,0)`, `(0,305)`, `(493,305)` mm respectively. These explicitly supplied coordinates supersede the earlier approximate diagonal reconstruction. Marker 0 is the origin; +X points toward marker 1 and +Y toward the lower row. The user separately confirmed the 30 mm black-square side, excluding the white border.
- **Flat-board bounds still need measuring.** Set `BOARD_BOUNDS_MM` before projector collection or validation. ArUco pose testing can run without these bounds. Old fixture dimensions and old calibration numbers are not included.

## Environment

A local fresh environment can be created from the repository root:

```powershell
conda env create --file environment.yml
conda activate hackthenorth
```

The project uses the named environment `hackthenorth`. On the original development machine it lives at `C:\Users\terre\miniconda3\envs\hackthenorth`; the previous `.conda-env` in the parent folder is retained as a separate environment.

If the environment already exists, activate it. If Conda is not initialized in your shell, open Anaconda Prompt and run `conda activate hackthenorth`. Alternatively, on the original development machine, call `& "$env:USERPROFILE\miniconda3\envs\hackthenorth\python.exe"` directly instead of `python` below. Do not install another OpenCV package alongside `opencv-contrib-python`.

## Run in this order

Run commands from the repository root. Every step is independent; stop and fix a failed smoke test before proceeding. Press **Esc** to exit preview tools.

1. **Webcam:** `python main.py webcam`. A bounded capture check is `python main.py webcam --frames 10 --no-preview`. A resolution mismatch fails instead of silently resizing.
2. **Camera intrinsics:** `python main.py camera-calibration --square-mm 25`. Measure the actual printed square; 25 mm is only the starting value. Use a **9Ã—6 inner-corner** chessboard. Space captures; vary tilt, distance, and coverage across the image. Aim for 30 sharp views; C solves with at least 15. Saves `camera_calibration_1080p.npz` with `camera_matrix`, `dist_coeffs`, image size, RMS, and per-view errors. Review errors before proceeding; replacing a calibration requires `--overwrite`.
3. **ArUco detection:** `python main.py aruco --detect-only`. No fixture measurements or intrinsics needed for this mode.
4. **Board pose:** check `MARKER_CENTERS_MM` in `config.py`, then run `python main.py aruco` after camera intrinsic calibration. Needs at least 3 configured markers and a low corner reprojection error. Measures the black square, excluding the white border. Define +X to marker-right and +Y to marker-bottom; all marker tops face -Y. Centers are in mm. Before projector collection, set `BOARD_BOUNDS_MM` to `(min_x, min_y, max_x, max_y)` of the actual flat surface, not automatically the marker-center rectangle. Axes show X red, Y green, Z blue.
5. **Second display:** `python main.py projector --list`, then `python main.py projector`. Auto-selects only when exactly one non-primary display is present. B displays black; D displays a green center dot at (960,540). Set `PROJECTOR_MONITOR_INDEX` if needed. Verify exact pixel mapping physically.
6. **Green-dot detection:** `python main.py dot`. Measures increased green AND green dominance after black/dot settling. Shows the threshold mask, raw camera centroid, and area. Space repeats. Lock camera exposure/white balance/focus through its vendor controls if possible; reject reflections, clipped dots, or unstable exposure before collection.
7. **Collect projector correspondences:** finish the rigid mount first. Disable auto-keystone/dynamic geometry and keep focus/zoom/settings fixed. Run `python main.py collect --rigid-mount-ready`. Space starts a 5Ã—4 grid in the central 50% of the projector. Hold both rig and board stationary through the grid. Captures outside the board region, overlapping marker ink, with ambiguous blobs, or with board movement are rejected. At least 8 valid points are required to save one `data/pose_*.json` file. Re-run for **6â€“8 poses**, varying both tilt and depth by moving the whole rigid camera/projector pair relative to the board. Merely sliding a board flat is insufficient. Recheck the surface is flat and fully covers the configured bounds; a camera ray alone cannot prove a dot landed on that plane.
8. **Correspondence diagnostic:** `python main.py diagnose`. Reports homography residuals for every pose, including rejected RANSAC points in the reported errors. Large errors can indicate bad detections, motion, incorrect geometry, or lens distortion. Move an identified bad pose file out of `data` before solving again.
9. **Solve:** `python main.py solve`. Calibrates projector intrinsics, computes each `T_pb @ inverse(T_cb)`, reports per-pose rotation/translation consistency, then fits one fixed cameraâ†’projector transform across all observations. Refuses insufficient pose diversity or excessive residuals/transform spread. Diagnostic arrays are saved separately, even when quality checks fail. Passing limits saves `projector_calibration.npz`; passing these software limits does **not** establish physical accuracy.
10. **Physical validation:** `python main.py validate` targets the configured board center, or use `python main.py validate --target X_MM Y_MM` with measured numeric coordinates. Space projects, independently observes the landing point, and reports millimeter error. Results append to `data/world_validation.csv`. Move the whole rig, settle, and repeat the same physical target. Also sample targets around the usable board. Do this before assembly guidance.

11. **Part perception:** `python main.py perceive`. Needs step 2 and step 4 only â€” it is independent of the projector chain, so it can be demonstrated before calibration is solved. Place one part at a time into a still workspace; the window reports the observed board pose and the correction. `--steps FILE.json` replaces the mock CAD sequence. Measure your real parts into `perception/part_catalog.py` and tune their HSV ranges under venue light first, or nothing will be recognised.

Equivalent modules run with `python -m calibration.webcam_smoketest`, etc. Do not run files directly by path; module execution keeps imports consistent.

## ArUco rejection diagnostics

`python main.py aruco` shows a diagnostic panel beside the rotated preview and logs a report once per second, including rejected frames. It lists every detected ID, configured matches and unknown IDs, known/used marker counts, solvePnP success (or not attempted), mean/maximum/RMS corner reprojection error, the configured RMS threshold, and the exact rejection reason. The original detector settings, `SOLVEPNP_ITERATIVE` call, 2.0 px RMS gate, and positive-depth check are unchanged. Mean and maximum errors are diagnostic values; the gate still uses RMS.

Per-marker residuals help locate a bad correspondence. Cyan arrows follow each detected marker's decoded canonical corner 0 to corner 1; magenta dots show the model-projected corners and yellow lines show residuals, including rejected poses. All geometric calculations stay in RAW camera coordinates. Only the human preview rotates.

Startup output lists the loaded config/calibration paths, configured centers, 30 mm black-square side, assumed 0-degree physical rotation for each marker, and computed pairwise spacings. The current model uses the user's remeasured 493 mm horizontal and 305 mm vertical spacings. With all four unique known markers visible, a **diagnostic-only** homography fitted to marker centers estimates each printed marker's rotation and side lengths in the board frame. This audit is not used to modify or accept a pose. It depends on the center measurements, lens calibration, and a flat board, so it cannot independently certify the fixture dimensions. All marker rotations are currently modeled as zero; no rotated-marker compensation is applied automatically.

Press **S** to save the current raw frame, annotated preview, and full report under `data/aruco_debug/`. A bounded terminal-only capture and an offline replay are also available:

```powershell
python main.py aruco --frames 30 --no-preview --save-debug data/aruco_debug
python main.py aruco --image data/aruco_debug/raw.png --no-preview
```

Use the saved **raw.png**, not the rotated preview, for replay. The capture saves its final frame; press S interactively to preserve a particular failing frame. The diagnostic folder is ignored by Git. Confirm physical rotations against the cyan corner-direction arrows and remeasure the black square (excluding the white border) before attributing large residuals to calibration quality.

### Independent-marker and 0/1/3 pose comparison

Run `python main.py aruco --diagnose-poses` to add **diagnostic-only** fits alongside the unchanged production pose:

- Fit each detected marker independently using its four decoded canonical corners, the configured 30 mm side, and the current intrinsics/distortion. Print solver success, RMS and positive-depth/RMS acceptance. The local origin is the marker center, with X right and Y down in its canonical drawing.
- Fit the complete `[0,1,3]` subset without marker 2. If any of those IDs is missing or duplicated, report unavailable instead of silently choosing different markers. Print the subset RMS and residuals for every visible configured marker, including marker 2 explicitly labeled **HELD OUT**.
- Compare individual fits with an all-known-marker diagnostic fit and the original production result. Individual RMS below 2 px does not establish accurate depth, scale or camera calibration: each individual fit only constrains four corners.
- Open a separate **RAW / UNROTATED** window with numbered green detected corners `D0..D3` and magenta projected corners `P0..P3`, linked by residual lines. The ordinary human preview retains its existing 180-degree rotation. The raw view defaults to the 0/1/3 fit; use `--diagnostic-fit individual` or `--diagnostic-fit all` for the other comparisons.

Corner numbers follow the decoded printed marker: `0=TL, 1=TR, 2=BR, 3=BL`, never sorted by screen position. For the user-confirmed same-orientation fixture, all declared rotations default to zero. The terminal/report lists each canonical index's exact board XYZ and detected RAW UV, along with the image-based orientation audit. If a marker's actual printed rotation is known to differ, `--marker-rotation 2=90` declares it **only for diagnostic fits** (repeat for other IDs). Positive angles turn board +X toward +Y. For +90 degrees, canonical corners 0/1/2/3 map to physical board TR/BR/BL/TL. No detector corners are reordered, no rotation is inferred and applied automatically, and the production tracker/configuration is not modified. See [OpenCV's corner-order documentation](https://docs.opencv.org/4.11.0/d5/dae/tutorial_aruco_detection.html) and [planar solvePnP requirements](https://docs.opencv.org/4.9.0/d5/d1f/calib3d_solvePnP.html).

Press **S** to save the comparison JSON and numbered `correspondence_raw.png` at original resolution, plus `correspondence_panel_raw.png` with the report alongside it. Analyze an existing failing frame without reopening the camera:

```powershell
python main.py aruco --diagnose-poses --image data/aruco_debug/raw.png --no-preview --save-debug data/aruco_pose_comparison
```

Interpretation: if individuals pass and the full-board fit fails, investigate the shared planar geometry, physical marker orientation/height, and lens model. If 0/1/3 passes but marker 2's held-out residual is large, that localizes an inconsistency to marker 2 relative to those three. If 0/1/3 also fails, excluding marker 2 is insufficient. These comparisons alone cannot prove cardboard lift is the sole cause. This mode never changes the calibration file, measured fixture geometry, marker size, 2 px threshold, or the set of markers used by the production tracker.

## Shared geometry and data contracts

All translations and board coordinates are **mm**. OpenCV rotation vectors are **radians**. Marker corners are top-left, top-right, bottom-right, bottom-left in printed-marker coordinates.

```text
X_camera = R_cb @ X_board + t_cb
C_board = -R_cb.T @ t_cb
T_projector_board = T_projector_camera @ T_camera_board
```

- `perception.aruco.BoardTracker.estimate(raw_frame)` returns rvec, tvec, R, visible IDs, and corner RMS, or `None` for unreliable pose.
- `perception.geometry.camera_pixel_to_board(u, v, rvec, tvec, K, dist)` undistorts a raw pixel, transforms its ray into board coordinates, and intersects z=0. Points on elevated parts do **not** satisfy this plane assumption.
- `projection.world.board_point_to_projector(...)` composes transforms and uses projector distortion to obtain projector pixels.
- Each collected point saves exact rendered projector UV, raw camera UV, board XYZ, its measured camera rvec/tvec, ArUco RMS, and visible IDs. Each pose includes camera-calibration and fixture fingerprints to prevent mixed datasets.
- New camera calibration, fixture geometry, resolution, projector geometry settings, or relative mount movement requires reconsidering/recollecting calibration. Settings and physical mount changes cannot be detected automatically from file fingerprints.

## Perception: expected-part detection and placement validation

Answers one constrained question â€” *we expect part X now; did X appear, where, and is it right?* â€” never open-set object recognition. `perception.pipeline.detect_and_validate(frame_before, frame_after, expected_part, board_pose, camera_matrix, dist_coeffs)` returns:

```json
{
  "detected": true,
  "correct_part": true,
  "part_id": "red_l_plate",
  "observed_pose": {"x_mm": 118.2, "y_mm": 92.5, "z_mm": 0.0, "theta_deg": 17.3},
  "confidence": 0.91
}
```

Pass `expected_pose` as well and `result.error` carries `correct`, `dx_mm`, `dy_mm`, `dtheta_deg`. `result.status` is one of `ok | no_change | not_found | wrong_part | low_confidence | wrong_position | wrong_angle | wrong_position_and_angle`.

```text
perception/change_detector.py  before/after diff -> changed region. LOCALISATION ONLY.
perception/rectify.py          metric top-down view of the board plane -- match HERE
perception/part_matcher.py     HsvOutlineMatcher: is this the expected part?
perception/part_catalog.py     per-part HSV + top-down outline (mm) -- EDIT THIS for real parts
perception/pose_estimator.py   template-alignment x/y/theta in the rectified plane
perception/validator.py        observed vs CAD target -> the correction to apply
perception/pipeline.py         detect_and_validate(...), the front door
assembly/state_machine.py      baseline + stillness + step sequencing
assembly/demo_perception.py    python main.py perceive
```

**The camera views the workspace at 30-45 degrees, so nothing is matched in raw image space.** The changed region is rectified into a metric top-down view of the board plane first. That fixes three things at once: perspective foreshortening (~29% compression at 45 degrees), varying px/mm across a tilted view, and handedness -- a valid camera shows board +Y *upward* while `part_catalog` draws it downward, so a raw-space template match compares a part against its own mirror. Chiral parts then never match while symmetric ones keep working, which is a miserable failure to diagnose. `camera_height_above_board()` guards the pose that causes it.

**Conventions, now fixed** (the yaw sign/origin this README previously left open):

- **`theta_deg`** is the angle of the part's **+x axis in the BOARD frame**, CCW from board +x toward board +y, in `[0, 360)`. It is derived by mapping a direction through the board plane, never by reusing an image-space angle, so the upside-down mount and perspective cannot leak into it. A test asserts the same physical placement reports the same angle with the camera rolled 180Â°.
- **`theta_deg = 0`** means the part lies exactly like its `outline_mm` polygon in `part_catalog.py`. The drawing, not the code, defines each part's zero.
- **`x_mm` / `y_mm`** are the **area centroid of the part silhouette**, not the origin of its outline drawing. **The CAD target pose must use the same reference point.**
- **Angle errors are folded by `symmetry_deg`.** A 2Ã—4 brick placed end for end is not an error; an asymmetric part placed backwards is a 180Â° error.
- **`dx_mm`, `dy_mm`, `dtheta_deg` are the correction still to apply** (`expected - observed`), in board axes. Turning `+X` into a LEFT/RIGHT arrow depends on where the user stands and where the projector is, so that mapping belongs to the projection subsystem, not here.
- **Undistorted, then rectified.** Detection runs on undistorted frames in the same K, because lens distortion bends a silhouette before its angle is measured and cannot be undone afterwards. Matching and pose then run in the rectified board plane, where rectified pixels *are* board millimetres. `perception.geometry.board_point_from_undistorted_pixel` is the sibling of `camera_pixel_to_board` for pixels already undistorted; feeding undistorted pixels to the latter would undistort them twice.
- **Each step names the plane it rests on.** `Step.z_mm` (or `assembly_height` from the CAD side) is the height of the surface the part sits on. Measuring a raised part on the z=0 plane gives a systematic outward error of `r*z/(H-z)` -- about 4 mm for one brick layer near the board edge, which reads as drift, not noise.
- **Marker quads are excluded** from the change search, so markers never become candidate parts.
- **Call `set_baseline()` at SHOW_NEXT_STEP, before the user reaches in**, and with the projector showing whatever it will show during the step â€” its light contaminates the camera image. `AssemblyState` re-baselines automatically when a step completes.

Fusion supplies a part ID and CAD target pose; later register CAD to board/world with `T_world_part = T_world_CAD @ T_CAD_part`. `Step.from_dict` already accepts `{"part_id", "target_pose"}` and keeps unknown keys. No Fusion API or servo behavior is implemented here. Later servo sequence: move â†’ settle â†’ reacquire ArUco pose â†’ project; commanded servo angles are not world-pose measurements.

**Known perception limitations.** Same-coloured parts touching each other merge into one blob (â†’ `low_confidence`). A part that is its own mirror image cannot be flip-detected â€” `yellow_l_plate` has equal arms and is exactly this case, so prefer chiral outlines with strong concavities. Near-rectangles fit poorly at any angle. Webcam autofocus and auto-exposure drift shift the colours, and autofocus also changes the intrinsics. Perspective shear grows with camera tilt; the alignment assumes a roughly overhead view. **Every HSV range and outline in `part_catalog.py` is a placeholder measured from nominal LEGO geometry, not from your bricks under your light.**

## Verification

`python -m unittest discover -s tests -v` runs 93 synthetic checks of distorted raw-pixel geometry, upside-down ArUco detection, multi-marker pose, transform composition, green detection, rigid-calibration recovery, rejection of moving-mount/degenerate data, and the whole perception slice â€” silhouette alignment across the full circle, part pose recovered in board millimetres through the real lens model and the upside-down mount, hand and marker rejection, wrong-part naming, symmetry folding, and step sequencing. It does not replace hardware calibration or physical error measurements.

Perception is validated **synthetically only**: parts are rendered, projected through the real `camera_calibration_1080p.npz` lens model onto the measured 493×305 mm fixture, and recovered. 108 placements per row (3 parts × 3×3 positions × 4 angles), camera 700 mm from the board centre and mounted upside down.

**Flat parts, across the real operating range of camera angles** — rectification makes accuracy essentially independent of viewing angle:

| camera elevation | found | position mean / max | angle mean / max |
|---|---|---|---|
| 0° (overhead) | 108/108 | 0.18 / 0.58 mm | 0.33° / 2.0° |
| 30° | 108/108 | 0.17 / 0.52 mm | 0.63° / 2.0° |
| 40° | 108/108 | 0.16 / 0.46 mm | 0.63° / 2.5° |
| 50° | 108/108 | 0.20 / 0.46 mm | 0.80° / 3.0° |

About **87 ms** per `detect_and_validate` call at 1080p.

**Part height is the real limit, and it is not fixable from a top-down outline.** At an angle the camera sees a part's *sides*, so the silhouette grows by roughly `height × tan(elevation)` — about 7 mm for a LEGO brick at 35°. Matching that against a flat footprint template degrades and then fails outright:

| part height | found (at 35°) | position mean | template IoU |
|---|---|---|---|
| flat | 36/36 | 0.18 mm | 0.97 |
| 3.2 mm (plate) | 36/36 | 1.18 mm | 0.89 |
| 6 mm | **8/36** | 1.71 mm | 0.86 |
| 9.6 mm (brick) | **0/36** | — | — |

So **anything taller than ~4 mm needs a 3D mesh per component from the CAD side**, not a top-down outline: the expected silhouette then gets synthesised at runtime by projecting the mesh through the measured board pose, which is the only correct approach when the camera angle is variable. Flat plates work today; bricks do not.

Those numbers measure the geometry, not the world. No real webcam frame, no real brick, and no venue lighting has been through this yet. Real accuracy will be set by segmentation quality â€” HSV tuning, shadow, glare, projector light on the part â€” not by the alignment, and it will be worse. Measure it on hardware before trusting any of it.
