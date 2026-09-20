# Visionary — CAD-guided assembly copilot

Hack the North MVP: a calibrated webcam observes LEGO placement, a calibrated
projector shows the next target, and an ArUco marker on a movable base allows
repositioning **between** placement steps. Current handoff: **2026-09-20**.

Read this first when continuing in Claude Code on a new computer. This README
supersedes older fixture descriptions and historical diagnostic reports.

## Quick start on a new computer

```powershell
git clone https://github.com/terrencefu/Visionary.git
cd Visionary
conda env create --file environment.yml
conda activate hackthenorth
python main.py webcam --frames 10 --no-preview
python main.py aruco
python main.py projector --list
python main.py projector
```

`environment.yml` creates Python 3.11 and installs `requirements.txt`.
Alternatively: `conda create -n hackthenorth python=3.11 pip`, activate it, then
`python -m pip install -r requirements.txt`. Install only **opencv-contrib-python**,
not additional OpenCV wheels. Run commands from the repository root. For
subcommand help use e.g. `python -m assembly.demo_perception --help` (the top-level
launcher intercepts `--help`).

Hardware/UI was developed on Windows with DirectShow. Linux/macOS have not been
physically tested. Close other camera applications before starting a preview.
The original checkout was
`C:\Users\terre\OneDrive\Documents\ChatGPT\HackTheNorth\Visionary`; that absolute
path is not required on the new computer. Historical reports may contain it.

### Hardware and calibration portability

- External webcam: AC310, originally `CAMERA_INDEX = 0`, RAW **1920 x 1080**.
  Device indices can change on another computer; check `config.py`.
- Camera and projector must stay rigid relative to each other. No servo command
  is issued by the current assembly workflow.
- Projector: Windows **Win+P -> Extend**, non-primary **1920 x 1080** display.
  Keep projector scaling at 100%; disable keystone/automatic geometric correction.
  Auto-selection works only with exactly one non-primary display. Otherwise set
  `PROJECTOR_MONITOR_INDEX` after checking `projector --list`.
- Preview alone rotates 180 degrees. Never rotate/resize RAW geometry inputs.
- `camera_calibration_1080p.npz` and `projector_calibration.npz` are included.
  A computer change alone does not require recalibration if the same devices,
  lens/focus, resolution, projector settings, and rigid mount are preserved.
  A different camera, changed focus/zoom, or changed relative mount can invalidate
  these files even when their software fingerprints still pass. Physically
  validate before using guidance. Do not automatically overwrite calibration.
- Current guidance uses **3D camera/projector calibration**, not the saved planar
  homography. The legacy planar file is included for historical reproducibility.

## Physical fixture — current values

All lengths are millimetres. Dictionary: **DICT_4X4_50**. Fixed marker black-square
side: **30 mm**, excluding white margin. Marker tops share the same orientation.

| Fixed marker ID | Center X | Center Y |
|---|---:|---:|
| 0 | 0 | 0 |
| 1 | 303 | 0 |
| 2 | 0 | 203 |
| 3 | 303 | 204 |

Preserve the 1 mm asymmetry. **Do not restore the obsolete 493 x 305 mm fixture.**

- `BOARD_BOUNDS_MM = (30, 30, 273, 173)` is the projector calibration region.
- `DETECTION_WORKSPACE_MM = ((-30,-30),(333,-30),(333,234),(-30,233))`
  is the separate cardboard detection polygon. The user reported cardboard
  extending 30 mm beyond the outer marker-center boundaries.
- Moving base: **ID 5**, **30 mm black square**, attached to a stiff flat sheet.
  Attach the anchor to that sheet so the marker-to-anchor offset cannot change.
  Do not duplicate IDs 0-3 on the movable sheet.
- Move all installed parts together. Two loose plates do not automatically form
  a rigid assembly; marker tracking cannot detect a part slipping on the sheet.

## Integrated CAD pipeline

1. Run the updated AssemblyGuide exporter inside Fusion to produce `assembly.json`
   with a plan, occurrence transforms, and `color` fields. Assign appearances that
   match the real pieces. Run the toCV exporter for the same design to produce
   `toCV_output.json` and its referenced component-local STL meshes.
2. Place the exports together in one folder. Do not mix exports from different
   design revisions. The checked-in export at integration time had no colour
   fields, so it needs re-exporting before strict mode can start.
3. Check the inputs and existing calibration without opening hardware:

```powershell
python main.py assemble --cad Fusion_output --check-only
```

4. Run the tracked assembly:

```powershell
python main.py assemble --cad Fusion_output --base-marker-id 5 --base-marker-size 30
```

The first planned part is selected as the anchor automatically; no component name
is required on the command line. Sequence, dependencies, geometry, occurrence
orientation, and colours come from CAD. Part colour overrides component colour.
Repeated instances may have different colours and orientations. The current
3D camera/projector calibration, fixture, and moving-base transform are reused.

Space captures the initial empty-base baseline. Place the first planned part,
press V to register it, inspect the outline, and press Enter to confirm. Later
steps use the accepted Enter frame automatically: add the next requested part,
remove hands, V checks, Enter rechecks and advances only on a pass. M then Space
is recovery; B resets. No servo motion is issued.

RGB colours are converted to generic camera HSV bands in `perception/cad_colour.py`.
Hue wrap and neutral/black/white colours are supported. `CAD_COLOUR_*` settings
control the starting bands; actual lighting can require tuning. Installed surfaces
are classified against the current target's colour band for visibility comparison.
No LEGO-name lookup is used by strict `assemble`. Legacy `perceive --cad --anchor`
can still load old exports without colour fields and labels its fallback in the
startup report. An explicitly invalid CAD colour never silently falls back.

Strict mode rejects missing, textured/unavailable, or insufficiently dominant
mixed colours. It validates mesh references/units, operation dependencies and
occurrence transforms. Current scope is rigid assemblies resting on a tracked
flat base, CAD root +Z up, one new part at a time, and CAD-assumed height. It is
not general six-degree-of-freedom object recognition or hidden-connection validation.
`ASSEMBLY_SEARCH_*`, `ASSEMBLY_MIN_*`, placement tolerances, and the ambiguity margin
are shared configuration, not per-component rules. Saved checks include the colour
profiles so replay does not depend on a later re-export or renamed components.

## Legacy demo: moving-base assembly

```powershell
python main.py perceive --cad Fusion_output --part-id "2850 Medium Stone Grey Technic Engine Cylinder Head" --anchor --base-marker-id 5 --base-marker-size 30
```

### Initial anchor

1. Put the sheet with marker 5 on the fixed board, **without the engine block**.
   Keep marker 5 and at least three reliable fixed-board markers visible.
2. Press **Space** to capture a baseline.
3. Attach the engine block in its CAD upright orientation without shifting the sheet. Remove
   your hand. Press **V** once a change is detected.
4. The program verifies STL shape and fits metric anchor X/Y/yaw. Inspect the
   green outline, then press **Enter** to accept the anchor.

### Colour-based changes after the first anchor

The first grey engine-block anchor still uses the existing STL registration.
For subsequent placements, the physical **blue 1x6** and **red 1x10 plates** are
selected using HSV colour masks of newly added pixels. `Plate 1x10 Silver` stays
as the exported CAD name but maps to physical red in `PLACEMENT_PART_COLOURS`.
Unchanged earlier red pieces cannot satisfy the next red step.

For later blue/red additions, `perception/assembly_matcher.py` now tests the
expected CAD placement and nearby XY/yaw alternatives. A perspective-correct
camera depth buffer accounts for installed parts hiding the target. New colour
pixels trigger the check; the comparison uses the full current colour mask in a
fixed neighbourhood, including predicted installed surfaces of the same colour.
It does not fit a full isolated STL to the novelty fragment.

**V** checks; **Enter** rechecks and advances only on **PLACEMENT OK (visible
evidence)**. **ADJUST PLACEMENT** is issued only for a strong displaced hypothesis
that clearly beats acceptable alternatives. **INSUFFICIENT EVIDENCE** blocks
advancement without claiming a measured correction. Look at the separate
**Assembly visibility** window: green is predicted visible target, red is observed
colour, yellow is their overlap. Coordinate boxes remain in Placement comparison.

The existing 3 mm / 8 degree tolerances and 0.76 agreement floor remain.
`PLACEMENT_AMBIGUITY_MARGIN = 0.005` in `config.py` controls the required score
advantage for all visible-assembly placement checks, with no part-specific override.
The separate isolated-STL identity margin remains 0.08. Placement logs and saved
reports include the active margin and measured score gap.
This MVP setting accepts near-tied alternatives, so decisions near the position
or yaw limit may be less stable. Test both a correct and one-stud-offset placement.
Local search is bounded to +/-10 mm on each axis and +/-30 degrees;
a boundary winner remains uncertain. At least 40 cropped pixels, 25% target
visibility, and 15% support from new pixels are required. These conservative
visibility gates are experimental, not measured accuracy guarantees. The current
CAD support height and anchor registration are still assumed, and unknown hands
or non-CAD occluders are not reconstructed. Initial engine identification and
uncoloured/standalone STL diagnostics still use the original fitter.

Failed checks save the visibility comparison, all searched hypotheses, exact
assembly meshes, and valid-image mask for offline replay:

```powershell
python -m perception.placement_evidence data/placement_check_REPLACE_WITH_TIMESTAMP
```

The new checker passed synthetic correct, offset, rotated, half-hidden, and
missing/hidden-target tests. The saved real failure from this session still has
insufficient agreement (expected ~0.48, best local candidate ~0.63); no physical
full-sequence success is claimed. Calibration and geometry were not changed.

The standalone `python -m perception.diagnose_placement ...` command uses colour
selection and the original isolated-part fitter, not assembly visibility. Keep the engine/support assembly present
in the baseline, then add only the new brick. It saves `4_colour_change.png`, the
selected region, pose fit, and camera/board data for inspection. Keep projection
off during the independent diagnostic; the main MVP already blanks it for checks.

### Every subsequent part

1. After **Enter** confirms a placement (or the initial anchor), its clean,
   projector-blank frame and corresponding pose/registration become the next
   baseline. The next target is displayed automatically; no Space is needed.
2. Slide/rotate the whole flat base if desired, keeping installed parts attached.
   Blue/red checks compensate for camera/base motion. Add the requested part,
   stop moving, and remove your hands.
3. **V** checks placement. **Enter** checks again and advances only on a pass.
   A failed check or V alone does not replace the baseline.
4. Repeat through the four placements; completion blanks projection and exits.

Space is only needed for the initial baseline or an explicit recovery.
**M** cancels the pending placement and enters recovery; press **Space** to
capture a fresh baseline when ready. Remove any
unverified new part *before* re-baselining, or the new piece will be absorbed into
the baseline. **B** resets the whole assembly; clear it before a new baseline.
**Q/Esc** exits. Keys apply to an OpenCV window with keyboard focus.
Press a check key once: the preview shows Capturing/Checking while work runs.
Repeated step keys are ignored during that work. Guidance geometry is cached
between steps; tracking updates its transform without recomputing CAD surfaces.
The initial engine fit also uses a fresh projector-blank capture. Its first outline
confirms the engine anchor; after Enter the outline guides the next part.

During blue/red placement, marker loss pauses projection/checking and reacquisition
resumes the same baseline. The current fixed-board pose and marker-5 registration
update the expected CAD position. Previous colour pixels are reprojected over the
CAD height range, including elevated surfaces, before detecting new colour. Newly
revealed image borders are excluded. The 1.5 px board drift gate no longer blocks
these coloured steps; position/angle tolerances are unchanged. Initial engine
registration and uncoloured diagnostics still need a stationary baseline. Large
view changes, occlusion, or overlapping same-colour pieces may require removing
the unverified addition and taking a fresh baseline with M then Space.

This assumes the camera and projector remain rigidly attached, the assembly moves
as a unit with marker 5, and previously accepted parts remain attached.
The moving-base registration is constrained to a flat board after checking marker
height within 5 mm and tilt within 15 degrees. This is **flat sliding/yaw support**,
not arbitrary handheld/tilted assembly verification. Keep fixed markers visible
in the full workflow; the standalone tracking test needs them only initially.

### Sequence and what is actually verified

The current Fusion plan is:

1. `2850 Medium Stone Grey Technic Engine Cylinder Head:2` (anchor)
2. `389423 Bright Blue Technic Brick 1 x 6 with Holes:1`
3. `Plate 1x10 Silver:2`
4. `Plate 1x10 Silver:1`

Order, dependencies, and target transforms come from `Fusion_output/assembly.json`.
`assembly/manual_guidance.py` tracks the current step and gates advancement;
workflow phases currently live in `perception/demo_change.py`. This is a small
stateful controller, **not yet a generalized declarative state-machine engine**.

Each later check verifies the **current addition**, not all installed components.
It compares a blank-projector before/after image and, for coloured additions,
evaluates visible CAD hypotheses at the assumed support height against the
updated target. It does not independently measure height or confirm hidden
connections. The code cannot enforce the promise not to add parts during the
move phase. Stillness detection is not hand recognition: a stationary hand may
pass that check. Shadows, glare, touching parts, and occlusion can cause uncertainty.

The first two outlines use the physically tested plate body height **3.33 mm**.
The maximum STL height is **5.18 mm**; it gave an approximately 4 mm backward
landing offset in user testing. Keys **1/2** compare maximum/body heights for
those plate previews without refitting anchor XY/yaw.

For later steps, the target's XY bounding envelope is sampled onto the highest
surfaces of already-accepted CAD meshes (or cardboard where unsupported).
Height discontinuities are not joined. These are approximate placement cues,
not extracted mating interfaces. Projector-ray occlusion is not yet checked.
Raised full-sequence guidance needs continued physical validation.

## Independent tests / older modes

Detection tests now show a separate **Detection coordinates (unrotated)** window.
Yellow marks the observed image region; placement checks additionally draw green
expected and red fitted STL boxes. The bottom panel lists top-left/bottom-right
pixel coordinates, pixel dimensions, mask centroid, and available model-center
board X/Y in mm, yaw, assumed Z, and correction. A mask centroid is not a measured
physical part center. Pixel coordinates refer to the undistorted, unrotated image;
the normal human preview can still rotate independently.

For a standalone blue-brick check on the engine support:

```powershell
python -m perception.diagnose_placement --cad Fusion_output --part-id "389423 Bright Blue Technic Brick 1 x 6 with Holes" --base-z -19.2
```

Keep projection off. Space captures the supporting assembly before adding the
brick; Space again checks it. The coordinate overlay is displayed and saved as
`7_coordinate_boxes.png` in the printed diagnostic directory. It shows the fitted
pose, not an expected assembly target, because this standalone tool has no anchor
registration. The assembly's **Placement comparison** window shows both, and failed
checks save the same labelled overlay in their `placement_check_*` directory.

```powershell
# Moving marker + one identified anchor; CAMERA OVERLAY ONLY
python main.py moving-base --marker-id 5 --marker-size 30

# Stage 1 only: any changed object, workspace mask, no STL matching
python main.py perceive --change-only

# Isolated STL identity test: Space baseline, place object, V compares
python main.py perceive --cad Fusion_output --part-id "389423 Bright Blue Technic Brick 1 x 6 with Holes"

# Fixed-anchor sequence without moving-base support
python main.py perceive --cad Fusion_output --part-id "Plate 1x10 Silver" --anchor

# Standalone 3D projection tests
python main.py board-overlay
python main.py placement --part-id LEGO-2x4 --center 150 100 --width 32 --height 16 --rotation 0
python main.py validate
```

`python main.py perceive` without CAD options runs the **older HSV/2D-outline
catalogue demo**, not the current STL workflow. Its `assembly/state_machine.py`
is a separate controller. Do not assume changes there affect the new CAD loop.
The old pipeline also contains a +Z camera-side assumption that the newer STL
renderer explicitly handles differently.

The standalone moving-base test was reported working by the user. It binds once,
then follows marker 5 rather than continuously re-identifying the LEGO. Loss hides
the overlay; reacquisition preserves the same marker-to-anchor binding. No
calibration is modified. See [moving-base details](perception/MOVING_BASE.md) and
[STL/anchor details](perception/STL_TEST.md).

## Data, geometry, and module map

### Fusion colour export

Run the updated `assembly/AssemblyGuide/AssemblyGuide.py` inside Fusion's Scripts
and Add-Ins dialog to regenerate `assembly.json`. Each `components[]` and `parts[]`
entry now includes `color`: RGB integers (0–255), hex, appearance name, status,
source, representative area fraction, and an appearance palette. Component data
describes native surfaces; part data uses occurrence-context surfaces or an
explicit occurrence override. Prefer the part colour for an individual instance.

The representative colour is taken from the largest known appearance surface
group. Mixed appearances remain listed separately; textures, unknown colours,
and ambiguous shader properties are flagged rather than inferred from LEGO names.
These are CAD appearance colours, not calibrated camera HSV thresholds. Assign
the actual physical colours in Fusion (e.g. red plates even if their names say
Silver). Existing geometry/plan fields are unchanged. The integrated `assemble`
command consumes these fields automatically; older exports can still use the
explicitly labelled legacy fallback through `perceive`.

API references: [Occurrence appearance](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/fusion_Occurrence.htm),
[face appearance](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/fusion_BRepFace_appearance.htm),
and [ColorProperty](https://help.autodesk.com/cloudhelp/ENU/Fusion-360-API/files/core_ColorProperty.htm).

`Fusion_output/assembly.json`, `Fusion_output/toCV_output.json`, and all three
STLs under `Fusion_output/meshes/` are included. The manifest filename is
**toCV_output.json**, not `tocv.json`.

- STL units are mm; meshes are in **component-local** coordinates.
- Occurrence `position`/`rotation` place a mesh in `fusion_root_assembly`.
- Some operation `target_position` values refer to a named reference point,
  not the component origin or silhouette centroid. Do not interchange them.
- The second plate has component ID `Plate 1x10 Silver__2`; its
  `fusion_component_name` resolves to the shared `Plate 1x10 Silver` mesh.
- Exported registration placeholders are not measured physical transforms.
- Anchor fitting uses centered, upright geometry; CAD registration explicitly
  accounts for the occurrence and mesh-centering offsets.

```text
X_camera = R_camera_board @ X_board + t_camera_board
T_projector_board = T_projector_camera @ T_camera_board
T_marker_CAD = inverse(T_board_marker_initial) @ T_board_CAD_initial
T_board_CAD_current = T_board_marker_current @ T_marker_CAD
```

Board X points from marker 0 toward 1; Y points toward the lower row. In the
current physical setup, above the board is **negative board Z**. The STL renderer
uses a proper rotation (flipping both Y and Z where needed) to map CAD up to the
camera-facing normal. Flipping Z alone mirrors chiral geometry. Preview rotation
never participates in pose, ray intersection, STL rendering, or projection.
Lengths are mm; OpenCV rvecs are radians; user-facing yaw is degrees. This LEGO
MVP treats 180-degree yaw symmetry as equivalent.

| Module | Responsibility |
|---|---|
| `config.py` | Hardware, measured fixture, bounds, thresholds |
| `main.py` | CLI routing |
| `perception/aruco.py` | Fixed board pose and diagnostics |
| `perception/change_detector.py` | Masked frame differences and stillness |
| `perception/stl_matcher.py` | STL loading, perspective silhouettes, identity overlap |
| `perception/anchor.py` | Metric XY/yaw fit, mesh-height inference, CAD registration |
| `perception/moving_base.py` | Standalone ID-5/anchor tracking test and marker solver |
| `assembly/base_registration.py` | Flat moving-base transform, marker loss/baseline invalidation |
| `assembly/manual_guidance.py` | CAD step order, placement checks, receiving-surface guidance |
| `perception/demo_change.py` | Live CAD/moving-base workflow and controls |
| `projection/world.py` | Saved 3D calibration loading and XYZ-to-projector mapping |
| `projection/guidance.py` | Distortion-aware sampled edges; accepts XY or XYZ |
| `assembly/state_machine.py` | Older catalogue perception state machine, separate path |
| `assembly/AssemblyGuide/AssemblyGuide.py` | Fusion-host exporter (requires `adsk`) |
| `assembly/toCV/toCV.py` | Fusion-host STL/manifest export |
| `hardware/arduino/pan_tilt_servos/` | Firmware; not integrated into the assembly loop |

The attempted optical-flow anchor recovery was explicitly scrapped. Do not
restore its periodic blinking or local feature-search recovery. ArUco base
tracking is the chosen direction. No host-side pan/tilt integration is required.

## Current acceptance settings

| Check | Current setting |
|---|---|
| Fixed board pose | >=3 known markers, positive depth, RMS <=3 px |
| Projector collector pose | >=3 markers, RMS <=5 px |
| Placement | <=3 mm XY error, <=8 degrees yaw (modulo 180) |
| STL identity | overlap >=0.80, >=0.08 advantage over runner-up |
| Metric anchor fit | unshifted silhouette overlap >=0.80 |
| Perception stillness | 6 consecutive frames |
| Existing board drift check | 1.5 px; unchanged |
| Moving-base armed displacement | >3 px invalidates baseline |
| Projector solve RMS gate | 4 px |
| Per-pose rig consistency | 15 mm translation, 2 degrees rotation |

Overlap is a silhouette score, not a probability. `0.787 < 0.80` means identity
may have passed but metric pose fitting was rejected. Retry/check segmentation;
do not automatically loosen thresholds or add an arbitrary XY correction.

## Calibration and historical artifacts

Active camera RMS: approximately **0.443699 px**, 30 chessboard views, 9x6 inner
corners, measured 25 mm squares. Active eight-pose projector solve: projector
RMS **2.479 px**, fixed-rig RMS **3.121 px**. The user observed camera-measured
landing errors of **1.15–3.41 mm** at a board-center target across multiple rig
positions. This is not an independent ruler test or a guarantee on raised parts.

The current grid is **5 columns x 4 rows**, projector u=520–1160, v=470–610.
Off-board, marker-overlap, ambiguity, pose, and movement rejection remain active.
For intentional recalibration only:

```powershell
python main.py camera-calibration --square-mm 25
python main.py collect --rigid-mount-ready
python main.py diagnose
python main.py solve
python main.py validate
```

Do not run `solve --overwrite` just because you changed computers. Existing files
are preserved; explicitly requested overwrites and experimental exports have
separate CLI controls. Camera/projector signatures catch some mismatches but
cannot detect physical remounting or changed optics automatically.

Legacy planar tools remain selectable:

```powershell
python main.py planar calibrate --stationary-ready --dot-color magenta --overwrite
python main.py planar verify-current --stationary-ready --dot-color magenta
```

These are not prerequisites for the main assembly demo. Green is the default
calibration dot; magenta is supported. A planar homography is tied to a stationary
rig/board relationship. Its historical experimental ~7.973 px result should not
be confused with the active 3D calibration.

The handoff includes the active NPZ files, the legacy `planar_calibration.json`,
and local `data/` calibration datasets, diagnostics, logs, validation CSV, and
camera diagnostic images. Old backup folders are deliberately separate from the
eight active root-level `data/pose_*.json` files. Do not mix them into a new solve.
`.gitignore` still excludes newly generated data/NPZ artifacts by default; already
tracked handoff files are retained. Review intentional new data before force-adding.

The earlier 2.824-versus-18.801 fixed-rig discrepancy was reproduced on different
input selections: the exact first seven post-mount poses gave **2.823796705 px**.
Reports/manifests under `data/reproduction_*` preserve that investigation. It did
not establish mount movement as the cause. `calibration/RESULTS.md` describes an
older fixture/failing capture and is historical, not current setup instructions.

## Verification and Claude Code handoff

```powershell
python -m unittest discover -s tests -q
```

Tests cover synthetic geometry, calibration, masks, STL recognition, anchor
registration, placement rejection, moving-base transforms, and projection.
They do not prove live performance under new lighting. Handoff verification on 2026-09-20: **182 tests passed** in the full suite
(23 seconds on the original machine).

Physically reported working: original 3D projection validation, STL/anchor test,
body-height correction, and standalone moving-base tracking. Full moving-base
four-step sequencing was implemented and tested synthetically; the user has not
yet confirmed that entire hardware workflow. Thick/occluded parts remain the main
risk: the verifier still compares the changed region to an isolated expected
mesh rather than rendering a fully occlusion-aware assembly image.

Recommended continuation:

1. Reproduce the standalone moving-base test and projector smoke test on the new
   computer without changing calibration.
2. Run the moving-base sequence above, first with two plates. Verify that moving
   between steps changes projected targets and that wrong placements do not advance.
3. Test cylinder-head/blue-brick visibility and raised guidance. Preserve failing
   frames and scores before changing thresholds.
4. Improve assembly-aware visibility/segmentation if needed; don't infer full 3D
   correctness from a silhouette or assume a new baseline proves a placement.
5. Later extract explicit workflow states from the live loop. Keep the current
   working commands and raw-frame/coordinate conventions intact.

User preferences: act directly on simple requests; test substantial one-shot
changes. Do not silently alter measured geometry, calibration, or thresholds.
Use the existing 3D calibration rather than replacing it with planar mode. Keep
movement and adding a part separate. Pan/tilt is intentionally out of scope now.
