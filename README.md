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

Future perception interface:

```json
{
  "detected": true,
  "correct_part": true,
  "part_id": "example",
  "observed_pose": {"x_mm": 0, "y_mm": 0, "z_mm": 0, "theta_deg": 0},
  "confidence": 0.0
}
```

Perception must use the same board frame. Define yaw sign/origin with the teammate before integration. Fusion supplies a part ID and CAD target pose; later register CAD to board/world with `T_world_part = T_world_CAD @ T_CAD_part`. No Fusion API or servo behavior is implemented here. Later servo sequence: move → settle → reacquire ArUco pose → project; commanded servo angles are not world-pose measurements.

## Verification

`python -m unittest discover -s tests -v` runs synthetic checks of distorted raw-pixel geometry, upside-down ArUco detection, multi-marker pose, transform composition, green detection, rigid-calibration recovery, and rejection of moving-mount/degenerate data. It does not replace hardware calibration or physical error measurements.

Initial verification on this machine: Python 3.11.16, OpenCV 4.14.0, NumPy 2.4.6; all 10 synthetic tests passed and `pip check` reported no dependency conflicts. Camera index 0 captured 10 raw 1920×1080 frames successfully outside the execution sandbox. Only the primary 1920×1200 laptop display was enumerated. No real intrinsic/projector calibration or physical world-lock validation has been completed yet.

Reference: [OpenCV calibration and transform conventions](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
