# Manual change and STL test

## Placement-checked four-step assembly MVP

```powershell
python main.py perceive --cad Fusion_output --part-id "Plate 1x10 Silver" --anchor
```

Clear the cardboard and press Space. Place the first silver plate flat, studs up,
then remove your hand and press V. The STL identity check and metric anchor fit
run as before. Inspect the green body-deck outline and press Enter to confirm.
The anchor registration is now fixed; keep the first plate and board stationary.

Enter confirms the initial anchor. For each subsequent step, V checks placement;
Enter performs a fresh check and advances only on PASS. The sequence is first
plate, second plate, grey cylinder head, blue Technic brick, then completion.
The current component is printed in the terminal and camera preview. Completion
blanks the projector and exits. Q/Esc exits early. B blanks and resets the assembly;
clear the workspace before capturing a fresh baseline. Missing ArUco pose blanks
projection and prevents advancing until tracking returns. Invalid/off-screen
projection stops the preview. No servo commands or calibration-file writes.

Anchor motion tracking, local recovery, and periodic projector blanking have been
removed. The program does not continuously detect if a loose plate moves. Secure/check the
anchor manually. Later checks compare a blank-projector baseline from the prior
accepted step with a settled current frame. They verify the expected STL shape,
fit metric XY/yaw using its CAD base height, and compare against registered CAD
targets. Defaults are 3 mm position and 8 degrees rotation, with 180-degree LEGO
symmetry accepted. Failed or uncertain checks block advancement. The terminal
prints expected/observed position, correction, overlap, and assumed base Z.
Previous parts are not all reverified at every step; the current addition is
checked against the fixed assembly registration. Height is assumed from CAD, not
independently measured. Frame-difference contamination, shadows, or occlusion in
stacked assemblies can cause uncertainty. No override to bypass these checks is
added. Each Enter rechecks, so an old V success cannot accept a changed placement.

The first two plate outlines use the validated 3.33 mm body deck by default.
For steps 3 and 4, the target mesh's XY bounding envelope is sampled and placed on
surfaces of previously accepted CAD parts, using vertical triangle intersections.
Where no installed surface lies below a sample, the cardboard is used. Segments
across height discontinuities are omitted. This avoids aiming at the absent part's
future top, but remains approximate placement guidance, not a mating-surface or
contact-point detector. It assumes earlier parts match their CAD placements and
does not yet test occlusion along the projector beam. Physically validate these
raised guidance steps before relying on them for the demo.

Registration uses the full occurrence transform and mesh-origin offset. A 180-degree
symmetric anchor pose is accepted. The four-step controller validates dependency
order and currently requires one PLACE operation per CAD step.

### Height comparison

During either projected plate preview, press **1** for maximum STL height
(5.18 mm), or **2** for the inferred broad body deck (3.33 mm). The body deck is
the default after the physical alignment test favored it. It is
selected by the largest horizontal triangle area between bottom and stud top.
This is a mesh-derived diagnostic assumption, not a new physical measurement.
Only the outline height changes: the fitted anchor XY/yaw, CAD registration,
and saved calibration stay fixed. Start with the green outline on the existing
first plate, and keep both plate and rig still while toggling. If the body-height
outline aligns better, surface height contributes to the offset. A remaining
offset can still come from anchor fitting or calibration. The red target for an
absent second plate strikes the cardboard, so it is not a valid test of landing
accuracy on a raised surface until that plate is present.

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
