# CAD-guided assembly copilot

Hackathon MVP: establish a measured ArUco world frame, calibrate a rigid camera/projector pair, then independently measure where projected guidance lands. Fusion, part perception, and servo control are later integrations.

## Current setup

- Python 3.11, OpenCV 4 with contrib/ArUco, NumPy, screeninfo.
- Camera index **0**, raw **1920×1080**. The camera is upside down. Geometry always uses raw frames; only the annotated human preview rotates 180°.
- Projector: Windows **Extend**, **1920×1080**, non-primary monitor. Keep Windows scaling at 100% on the projector and verify the smoke test fills the display.
- Marker dictionary **DICT_4X4_50**, black-square size **30 mm** (confirmed). All markers have the same physical orientation.
- **Marker centers and flat-board bounds are not yet measured.** Enter them in `config.py`. Old fixture dimensions and old calibration numbers are not included.

## Environment

A local fresh environment can be created from the repository root:

```powershell
conda env create --prefix .\.conda-env --file environment.yml
conda activate .\.conda-env
```

On the original development machine, the already-tested environment is in the parent `HackTheNorth` folder. From this `Visionary` checkout, use `& ..\.conda-env\python.exe` in place of `python` (for example, `& ..\.conda-env\python.exe main.py webcam`). It stays outside Git; no environment relocation is needed.

If the environment already exists, activate it. If Conda is not on PATH, use `& "$env:USERPROFILE\miniconda3\Scripts\conda.exe"` instead of `conda`, or call `& .\.conda-env\python.exe` directly instead of `python` below. Do not install another OpenCV package alongside `opencv-contrib-python`.

## Run in this order

Run commands from the repository root. Every step is independent; stop and fix a failed smoke test before proceeding. Press **Esc** to exit preview tools.

1. **Webcam:** `python main.py webcam`. A bounded capture check is `python main.py webcam --frames 10 --no-preview`. A resolution mismatch fails instead of silently resizing.
2. **Camera intrinsics:** `python main.py camera-calibration --square-mm 25`. Measure the actual printed square; 25 mm is only the starting value. Use a **9×6 inner-corner** chessboard. Space captures; vary tilt, distance, and coverage across the image. Aim for 30 sharp views; C solves with at least 15. Saves `camera_calibration_1080p.npz` with `camera_matrix`, `dist_coeffs`, image size, RMS, and per-view errors. Review errors before proceeding; replacing a calibration requires `--overwrite`.
3. **ArUco detection:** `python main.py aruco --detect-only`. No fixture measurements or intrinsics needed for this mode.
4. **Board pose:** enter `MARKER_CENTERS_MM` and `BOARD_BOUNDS_MM` in `config.py`, then `python main.py aruco`. Needs at least 3 configured markers and a low corner reprojection error. Measures the black square, excluding the white border. Define +X to marker-right and +Y to marker-bottom; all marker tops face -Y. Centers are in mm. Bounds are `(min_x, min_y, max_x, max_y)` of the actual flat surface, not automatically the marker-center rectangle. Axes show X red, Y green, Z blue.
5. **Second display:** `python main.py projector --list`, then `python main.py projector`. Auto-selects only when exactly one non-primary display is present. B displays black; D displays a green center dot at (960,540). Set `PROJECTOR_MONITOR_INDEX` if needed. Verify exact pixel mapping physically.
6. **Green-dot detection:** `python main.py dot`. Measures increased green AND green dominance after black/dot settling. Shows the threshold mask, raw camera centroid, and area. Space repeats. Lock camera exposure/white balance/focus through its vendor controls if possible; reject reflections, clipped dots, or unstable exposure before collection.
7. **Collect projector correspondences:** finish the rigid mount first. Disable auto-keystone/dynamic geometry and keep focus/zoom/settings fixed. Run `python main.py collect --rigid-mount-ready`. Space starts a 5×4 grid in the central 50% of the projector. Hold both rig and board stationary through the grid. Captures outside the board region, overlapping marker ink, with ambiguous blobs, or with board movement are rejected. At least 8 valid points are required to save one `data/pose_*.json` file. Re-run for **6–8 poses**, varying both tilt and depth by moving the whole rigid camera/projector pair relative to the board. Merely sliding a board flat is insufficient. Recheck the surface is flat and fully covers the configured bounds; a camera ray alone cannot prove a dot landed on that plane.
8. **Correspondence diagnostic:** `python main.py diagnose`. Reports homography residuals for every pose, including rejected RANSAC points in the reported errors. Large errors can indicate bad detections, motion, incorrect geometry, or lens distortion. Move an identified bad pose file out of `data` before solving again.
9. **Solve:** `python main.py solve`. Calibrates projector intrinsics, computes each `T_pb @ inverse(T_cb)`, reports per-pose rotation/translation consistency, then fits one fixed camera→projector transform across all observations. Refuses insufficient pose diversity or excessive residuals/transform spread. Diagnostic arrays are saved separately, even when quality checks fail. Passing limits saves `projector_calibration.npz`; passing these software limits does **not** establish physical accuracy.
10. **Physical validation:** `python main.py validate` targets the configured board center, or use `python main.py validate --target X_MM Y_MM` with measured numeric coordinates. Space projects, independently observes the landing point, and reports millimeter error. Results append to `data/world_validation.csv`. Move the whole rig, settle, and repeat the same physical target. Also sample targets around the usable board. Do this before assembly guidance.

11. **Part perception:** `python main.py perceive`. Needs step 2 and step 4 only — it is independent of the projector chain, so it can be demonstrated before calibration is solved. Place one part at a time into a still workspace; the window reports the observed board pose and the correction. `--steps FILE.json` replaces the mock CAD sequence. Measure your real parts into `perception/part_catalog.py` and tune their HSV ranges under venue light first, or nothing will be recognised.

Equivalent modules run with `python -m calibration.webcam_smoketest`, etc. Do not run files directly by path; module execution keeps imports consistent.

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

Answers one constrained question — *we expect part X now; did X appear, where, and is it right?* — never open-set object recognition. `perception.pipeline.detect_and_validate(frame_before, frame_after, expected_part, board_pose, camera_matrix, dist_coeffs)` returns:

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
perception/part_matcher.py     HsvOutlineMatcher: is this the expected part?
perception/part_catalog.py     per-part HSV + top-down outline (mm) -- EDIT THIS for real parts
perception/pose_estimator.py   template-alignment x/y/theta, and the px -> board mm bridge
perception/validator.py        observed vs CAD target -> the correction to apply
perception/pipeline.py         detect_and_validate(...), the front door
assembly/state_machine.py      baseline + stillness + step sequencing
assembly/demo_perception.py    python main.py perceive
```

**Conventions, now fixed** (the yaw sign/origin this README previously left open):

- **`theta_deg`** is the angle of the part's **+x axis in the BOARD frame**, CCW from board +x toward board +y, in `[0, 360)`. It is derived by mapping a direction through the board plane, never by reusing an image-space angle, so the upside-down mount and perspective cannot leak into it. A test asserts the same physical placement reports the same angle with the camera rolled 180°.
- **`theta_deg = 0`** means the part lies exactly like its `outline_mm` polygon in `part_catalog.py`. The drawing, not the code, defines each part's zero.
- **`x_mm` / `y_mm`** are the **area centroid of the part silhouette**, not the origin of its outline drawing. **The CAD target pose must use the same reference point.**
- **Angle errors are folded by `symmetry_deg`.** A 2×4 brick placed end for end is not an error; an asymmetric part placed backwards is a 180° error.
- **`dx_mm`, `dy_mm`, `dtheta_deg` are the correction still to apply** (`expected - observed`), in board axes. Turning `+X` into a LEFT/RIGHT arrow depends on where the user stands and where the projector is, so that mapping belongs to the projection subsystem, not here.
- **Undistorted pixels.** Detection and pose run on undistorted frames in the same K, because lens distortion bends a silhouette before its angle is measured and cannot be undone afterwards. Board geometry then uses `perception.geometry.board_point_from_undistorted_pixel`, the sibling of `camera_pixel_to_board`; feeding undistorted pixels to the latter would undistort them twice. Parts are assumed to sit on the base plane z=0.
- **Marker quads are excluded** from the change search, so markers never become candidate parts.
- **Call `set_baseline()` at SHOW_NEXT_STEP, before the user reaches in**, and with the projector showing whatever it will show during the step — its light contaminates the camera image. `AssemblyState` re-baselines automatically when a step completes.

Fusion supplies a part ID and CAD target pose; later register CAD to board/world with `T_world_part = T_world_CAD @ T_CAD_part`. `Step.from_dict` already accepts `{"part_id", "target_pose"}` and keeps unknown keys. No Fusion API or servo behavior is implemented here. Later servo sequence: move → settle → reacquire ArUco pose → project; commanded servo angles are not world-pose measurements.

**Known perception limitations.** Same-coloured parts touching each other merge into one blob (→ `low_confidence`). A part that is its own mirror image cannot be flip-detected — `yellow_l_plate` has equal arms and is exactly this case, so prefer chiral outlines with strong concavities. Near-rectangles fit poorly at any angle. Webcam autofocus and auto-exposure drift shift the colours, and autofocus also changes the intrinsics. Perspective shear grows with camera tilt; the alignment assumes a roughly overhead view. **Every HSV range and outline in `part_catalog.py` is a placeholder measured from nominal LEGO geometry, not from your bricks under your light.**

## Verification

`python -m unittest discover -s tests -v` runs 93 synthetic checks of distorted raw-pixel geometry, upside-down ArUco detection, multi-marker pose, transform composition, green detection, rigid-calibration recovery, rejection of moving-mount/degenerate data, and the whole perception slice — silhouette alignment across the full circle, part pose recovered in board millimetres through the real lens model and the upside-down mount, hand and marker rejection, wrong-part naming, symmetry folding, and step sequencing. It does not replace hardware calibration or physical error measurements.

Perception is validated **synthetically only**: parts are rendered as flat polygons, projected through the lens model, and recovered. Over 45 placements (3×3 board positions × 5 angles, camera 600 mm up and rolled 180°, 2.5 px/mm): position error mean **0.10 mm**, max 0.24 mm; angle error mean **0.25°**, max 0.77°; **94 ms** mean per `detect_and_validate` call at 1080p, of which roughly a third is the two `cv2.undistort` calls that `AssemblyState` avoids repeating.

Those numbers measure the geometry, not the world. No real webcam frame, no real brick, and no venue lighting has been through this yet. Real accuracy will be set by segmentation quality — HSV tuning, shadow, glare, projector light on the part — not by the alignment, and it will be worse. Measure it on hardware before trusting any of it.

Initial verification on this machine: Python 3.11.16, OpenCV 4.14.0, NumPy 2.4.6; all 10 synthetic tests passed and `pip check` reported no dependency conflicts. Camera index 0 captured 10 raw 1920×1080 frames successfully outside the execution sandbox. Only the primary 1920×1200 laptop display was enumerated. No real intrinsic/projector calibration or physical world-lock validation has been completed yet.

Reference: [OpenCV calibration and transform conventions](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
