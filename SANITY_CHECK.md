# Script sanity check — 2026-09-20

Reviewed checkout: `40fd012`. Runtime code, measurements, thresholds, and saved
calibrations were not changed. This is an offline audit, not a completed hardware
acceptance test.

## Results

- All **209 unit tests passed** in 70.436 seconds using the `hackthenorth` Python.
- All **86 Python files** parsed successfully, including the Fusion scripts.
- Camera and projector calibration loaders passed. Saved camera/fixture signatures
  match current configuration; eight root-level pose datasets load successfully.
- Saved projector RMS is **2.479270 px**, fixed-rig RMS **3.121113 px**,
  `quality_passed=True`. The solver was NOT run.
- Thirteen of fourteen routed modules returned help successfully. The dot smoke
  test has no argument parser: invoking its module with `--help` starts hardware
  initialization instead. That attempt failed to open camera 0; it did not measure
  a dot. This does not establish which physical webcam index is correct.
- Fusion-host execution, Arduino compilation, live camera tracking, and physical
  projection/assembly accuracy were not validated in this audit.

## Priority findings

### 1. Incomplete masks are still fitted as complete parts

`perception/colour_change.py:43-48` subtracts previous same-colour pixels.
`assembly/manual_guidance.py:125` then sends that novelty mask to the complete
isolated-mesh fitter in `perception/anchor.py`. The fitter does not render installed
parts to determine which portions of the new part should actually be visible.

Reproduced without hardware: an old red rectangle at `(100,100,100,40)` and a new
red rectangle at `(150,100,100,40)` produce a candidate bounding box of
`(201,100,49,40)`. More than half the new part is excluded, but the downstream
template remains the whole part. Shadows, highlights, and actual occlusion can
also make the colour mask incomplete. This can cause low overlap or biased XY.

This verifies a failure mechanism, NOT the cause of the reported 5.52 mm blue-brick
offset. The historical blue-brick check still needs its exact frame, registration,
and mask to establish that cause. Changing overlap thresholds cannot fix the
mask/template mismatch.

Recommended direction: use change/colour to locate a candidate, then compare
expected visible geometry using the installed assembly and actual camera view.
When visibility is insufficient, report uncertainty instead of a confident offset.

### 2. README startup instructions no longer match the CAD export

The command at `README.md:83` was executed through argument validation and fails:
`--anchor requires --part-id "2850 Medium Stone Grey Technic Engine Cylinder Head"`.
Current flattened placement order is:

1. Engine head `:2` (anchor)
2. Blue Technic brick `:1`
3. Plate `:2`
4. Plate `:1`

The documented plate-first sequence and plate-specific height guidance are stale.
The acceptance table also says 0.80/0.80 overlap and 3 px moving-marker drift;
configuration now says 0.50/0.76 and 7 px respectively. Furthermore, the 7 px
invalidation gate is bypassed once adaptive motion is enabled (see below).

Current command, from the repository root:

```powershell
python main.py perceive --cad Fusion_output --part-id "2850 Medium Stone Grey Technic Engine Cylinder Head" --anchor --base-marker-id 5 --base-marker-size 30
```

### 3. Standalone projection can swallow Space/Q/Esc events

`hardware/projector.py:47` calls `waitKey` inside `black()` and discards its return.
`projection/guidance.py:89` calls it before its own keyboard handler at line 96.
Reproduced with a mocked event stream: Space is consumed by `black()` and the
outer handler receives no key. This affects placement/board-overlay readiness.
The live CAD loop already uses `blank_projector()` to avoid this issue, so the
fix is currently incomplete across tools. `dot()` also discards a keyboard event.

### 4. CAD mode silently ignores CLI placement tolerances

`assembly/demo_perception.py:110-111` accepts `--tol-mm` and `--tol-deg`, but the
CAD dispatch at line 136 does not pass them on. A mocked dispatch with 9 mm and
30 degrees confirmed neither reaches the CAD runner. `ManualAssembly` reads
configuration values instead: 3 mm / 8 degrees. These options only affect the
older catalogue demo. Wire them through explicitly or reject them in CAD mode.

### 5. Movement policy changed from the original between-steps contract

`perception/demo_change.py:288` enables `moving.adapt_motion=True` after binding.
`assembly/base_registration.py:48` consequently stops invalidating the armed
baseline on marker motion. The test suite explicitly checks that even a 60 px
marker displacement remains armed in adaptive mode.

This is intentional in recent code and supports colour-history compensation,
but it means the original rule "move only between steps" is no longer enforced.
Marker loss still pauses projection; planar height/tilt checks still apply.
The flag applies to the whole CAD session, although only coloured additions use
motion-compensated history. Future uncoloured additions can therefore receive
unaligned before/after masks if the base moves while the fixed board stays still.
Choose and document one motion policy; do not infer that 7 px remains an active
assembly movement limit.

### 6. Experimental calibration export lacks diagnostics provenance

`calibration/solve_projector_calibration.py:86` loads an existing diagnostics NPZ,
then lines 103-105 publish it with the CURRENT camera/fixture signatures and pose
names. Diagnostics do not carry a checked fingerprint tying their parameters to
those inputs. Stale diagnostics can be mislabelled if the input files change.
This affects `solve --experimental`; it is not evidence that the active normal
calibration is wrong. Preserve input signatures and dataset hashes with the solve
result and verify them before experimental export.

## Architecture and other limitations

| Area | What is actually connected |
|---|---|
| Active CAD assembly | `demo_change` -> `ManualAssembly` -> colour/change mask -> STL pose fit -> position/yaw gate; Enter advances |
| Moving base | Marker 5 supplies a frozen marker-to-CAD binding, then updates flat XY/yaw registration |
| Active guidance | Saved 3D projector model plus current board pose; no planar homography or servo calls |
| Legacy perception | `AssemblyState` and the HSV/2D catalogue are a separate path, used without `--cad` |
| Planar tools | Separate optional calibration, stationary overlays, and read-only current-position verification |
| CAD export | Fusion `AssemblyGuide` supplies plan/transforms; `toCV` supplies local meshes/manifest |
| InstructionGenerator | Old hard-coded External1/4/3 example; its output schema is not the active CAD controller or `Step.from_dict` schema |
| Arduino | Standalone serial servo firmware, not connected to assembly guidance |

- Raw-frame geometry and preview rotation are separated in the active path.
  The negative board-Z convention is intentionally handled with a proper rotation.
- Current marker geometry remains `(0,0), (303,0), (0,203), (303,204)`, with 30 mm sides.
- Camera index is 0, but its inline comment calls AC310 index 1. Saved signatures
  identify calibration files, not the physical device. The capture wrapper checks
  resolution but does not identify the camera or lock focus/exposure. Verify device
  identity manually before interpreting metric results; do not guess/change it here.
- Stillness checks motion, not hands. A stationary hand passes the six-frame gate;
  this was reproduced with a stationary foreground region.
- Previously accepted parts are assumed rigidly attached. Later checks do not
  revalidate every installed part or measure actual support height/connections.
- `load_models` keeps one upright model per component name, based on its first
  occurrence rotation. Different rotations of repeated components would require
  occurrence-specific fitting. The two current plate occurrences share a rotation.
- Guidance samples vertical intersections with installed CAD surfaces. It does
  not account for a different surface intercepting the projector ray first.
- Colour-history reprojection sweeps all CAD heights conservatively; after view
  changes it can suppress legitimate nearby same-colour additions.
- Test coverage includes clean synthetic fitting and rejection cases, but not an
  end-to-end, occlusion-aware run of the actual complete assembly. Some tests seed
  yaw or mark steps checked directly. Passing tests do not establish demo readiness.
- Colour placement tests save synthetic failure evidence into the real ignored
  `data/placement_check_*` directory. This run created two such folders; their
  reports have null board RMS/empty marker IDs. They are test artifacts, not real
  assembly observations. Tests should redirect evidence to temporary directories.

## Recommended order

1. Correct README/config descriptions, shared keyboard handling, and CLI argument
   semantics; keep calibration and physical measurements unchanged.
2. Restore or explicitly choose the intended movement-between-steps contract.
3. Fix the visible-mask/full-STL mismatch using saved real frames and tests of
   correct placement, deliberate offsets, and partial occlusion.
4. Demonstrate all four placements physically, including the two red plates,
   before calling the full sequence reliable. Keep ambiguous results uncertain.
