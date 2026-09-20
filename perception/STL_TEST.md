# Manual change and STL test

Run from the repository with the hackthenorth environment active:

```powershell
python main.py perceive --cad Fusion_output --part-id "389423 Bright Blue Technic Brick 1 x 6 with Holes"
```

Keep the rig stationary and the cardboard clear. Click the camera window and
press Space to capture the baseline, then place the expected blue brick upright
inside the yellow boundary and remove your hand. Once a green change box appears,
press V. The terminal prints all candidate overlap scores and a result. The
comparison window shows observed silhouette in green, best STL silhouette in red,
and overlap in yellow. Verification is a snapshot; press V again after settling
to test a correction. Remove the object before resetting with Space. Q/Esc exits.

Other expected component names:

- `Plate 1x10 Silver`
- `2850 Medium Stone Grey Technic Engine Cylinder Head`

`python main.py perceive --change-only` still runs stage one alone. The original
`python main.py perceive` still runs the old catalogue-based assembly demo.

The detection-only cardboard polygon is `DETECTION_WORKSPACE_MM` in config.py,
30 mm beyond the marker-center boundaries as supplied. Marker ink is excluded.
Calibration bounds, saved calibrations, and projector rendering are unchanged.
The polygon is fixed at baseline capture to avoid pose jitter moving its edges;
recapture after moving the rig. Geometry uses undistorted, unrotated camera pixels.

The verifier loads component-local STL triangles in mm, checks their bounding
dimensions against the assembly export, applies the occurrence's 3D rotation,
and places each candidate on z=0. CAD up points toward the camera-facing board
normal. For a camera on board -Z, a proper 180-degree X rotation maps CAD up
to -Z while preserving handedness (both Y and Z change sign). The board pose,
intrinsics, and preview rotation are unchanged. It renders full perspective mesh silhouettes
using the current camera pose, searches board yaw (15 degrees, then 3 degrees),
and compares silhouettes after image-centroid alignment without rescaling.
It does not use or alter the saved planar projector homography.

This is experimental visible-shape verification for isolated parts, not general
3D recognition or assembly placement validation. Resting orientation must match
the export (arbitrary tumbling and stacked parts are unsupported). STL colour is
not available or used. The observed outline comes from frame difference, so
shadows, lighting changes, and touching objects can contaminate it. Interior
features may be lost by the filled change contour. Similar visible shapes cannot
be reliably distinguished. Overlap is not a calibrated confidence probability.

The default match gate is overlap >= 0.80 with >= 0.08 advantage over the runner-up;
otherwise the result is UNCERTAIN. A confident rival gives INCORRECT SHAPE. These
are initial test gates, not physically validated accuracy guarantees. Review the
overlap image and test both correct and wrong parts under actual venue lighting
before using results for assembly advancement. No automatic advancement is added.
