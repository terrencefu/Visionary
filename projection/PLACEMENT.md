# Standalone LEGO placement demo

## Stationary diagnostics

Placement now logs every measurement to `data/placement_motion_<timestamp>.jsonl`.
The header records the saved reference pose; samples record current rvec/tvec,
translation delta in mm, relative rotation angle in degrees, marker IDs/count,
reprojection RMS, raw pixel-drift history, filtered drift and window jitter.
Translation is the difference of board-to-camera tvecs, not an independent
measurement of servo motion. Drift is the maximum predicted camera-pixel movement
of any saved reference object corner. It includes pose-estimation noise and bias.

The placement-only monitor takes a nine-sample median of projected corner
coordinates. A stable startup window establishes a fixed secondary baseline only
if it agrees with the saved calibration pose; it never rebases away an offset.
Both saved-reference and baseline drift retain the 1.5 px default limit.
An isolated raw outlier blanks output without ending the session. A filtered
offset lasting 0.5 seconds stops projection. Excessive window jitter or missing
tracking stops it after two seconds. No ArUco solve or calibration file is changed.
Persistent pose offset may mean movement OR systematic estimator bias; the log
does not claim to distinguish them conclusively from a single camera.

Run the placement at (150,120), which fits the currently measured calibration hull:

```powershell
python main.py placement --part-id LEGO-2x4 --center 150 120 --width 32 --height 16 --rotation 0 --stationary-ready
```

Leave the rig untouched for 20–30 seconds, then inspect the log: isolated raw peaks
with low filtered drift should recover, while a sustained offset should stop.
In a separate trial deliberately move the rigid head slightly and hold its new
position; projection should blank and stop when filtered drift exceeds the limit
for 0.5 seconds. Motion smaller than the threshold may not trigger it. Stop and
recalibrate manually before further guidance after movement; this command never
calibrates or drives servos. Projector-only motion remains invisible to ArUco.

Explicit tuning options are `--movement-px`, `--movement-window` and
`--movement-persist`. Defaults preserve the existing spatial threshold; use logged
stationary and deliberate-motion measurements before changing them.

With the calibrated camera/projector rig and board still in the same position:

```powershell
python main.py placement --part-id LEGO-2x4 --center 150 100 --width 32 --height 16 --rotation 0 --stationary-ready
```

Press Space when READY; Escape/Q exits. The green footprint has corners
(134,92), (166,92), (166,108), (134,108) mm. Width is along local X and height
along local Y. Positive rotation turns +X toward +Y, clockwise in a board drawing
with X right and Y down. The label is placed beside the outline; the crosshair
marks the requested center. All geometric endpoints use the saved homography.

Footprints must fit entirely within both the usable bounds and the calibrated
inlier hull. No scaling-to-fit or extrapolation is performed. The command also
requires space for the label. Existing stationary-rig checks blank projection
during tracking loss, allow two seconds to recover, and stop after three reliable
poses confirm excessive drift in the original board-alignment tool; placement uses
the filtered time-based check described above. A projector moving independently of the camera
cannot be detected by ArUco: stop and recalibrate after any movement.

This command reads the existing planar calibration without modifying it. It
does not perform calibration, physical validation, part detection, Fusion
integration, assembly sequencing, or servo control. It guides placement on the
flat board; it does not compensate for projection onto the top of an elevated brick.
