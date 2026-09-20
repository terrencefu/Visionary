# Moving base + anchor test

## Full assembly with movement between steps

```powershell
python main.py perceive --cad Fusion_output --part-id "Plate 1x10 Silver" --anchor --base-marker-id 5 --base-marker-size 30
```

Start with the marked sheet in place but no plate. Space captures the initial
baseline; attach the first plate without shifting the sheet and press V. Inspect
the green anchor outline, then Enter accepts the anchor and enters the move phase.

Before EVERY later placement:

1. Projection is blank. Slide/rotate the whole assembly on its base if desired.
2. Set it flat, remove hands, and press Space. A settled camera frame becomes the
   fresh baseline; the CAD-to-board transform updates from marker 5's unchanged
   marker-to-CAD binding. The target then projects at the new location.
3. Add only the requested part, with the base stationary. V checks placement;
   Enter takes a fresh check and advances only after a pass.
4. The next move phase begins automatically. Even if no move is needed, Space
   explicitly arms the next placement before adding its part.

Keep the fixed-board markers and ID 5 visible. The current version permits flat
sliding/rotation only; set-down height/tilt is checked (5 mm / 15 degrees) and the
accepted transform is constrained to the board plane to avoid single-marker
depth noise changing the assembly height. Raised part heights still come from CAD.

Marker loss blanks projection and blocks checking. More than 3 px maximum marker
corner displacement during an armed placement invalidates its baseline, even if
the base returns to the old location. Remove any unverified new part, press M to
cancel that placement and enter the move phase, then Space to arm again. M does
not undo accepted steps or accept a placement. B resets the whole assembly.

No comparison is made across the repositioning: move first, capture baseline,
then add. The software cannot prove that you did not add something during the
move phase; follow that restriction. Likewise, loose parts must remain fixed
relative to the sheet: marker tracking cannot detect independent slipping. The
placement check still verifies the current addition, not every prior component.
No calibration files, fixed-board geometry, or servo commands are changed.

## Standalone tracking test

```powershell
python main.py moving-base --marker-id 5 --marker-size 30
```

Use DICT_4X4_50 ID 5, with a measured 30 mm black square (white margin excluded).
Tape it flat to a stiff sheet/base. Do not reuse fixed-board IDs 0-3. This command
uses the existing camera calibration and the silver plate STL by default. It does
not open the projector, move servos, advance an assembly, or write calibration.

1. Lay the marked base flat on the fixed board, with room for the silver plate.
   Keep all fixed markers and marker 5 visible. The base must not flex.
2. With no plate on the base, press Space to capture the baseline.
3. Attach the first silver plate to the base without shifting the base; keep the
   plate clear of the marker. Remove your hand and press V when a candidate is found.
4. After CORRECT SHAPE and BOUND, a green body-height outline and axes should lie
   over the physical plate. Slide and rotate the entire base together on the board.
5. Cover marker 5: the outline must disappear. Uncover it: tracking should resume
   with the original binding. The fixed board markers are only needed for initial
   registration, not subsequent tracking. Q/Esc exits; R discards binding and resets.

If the base shifts before identification, remove the plate and recapture baseline.
The initial anchor footprint should remain within the configured detection region.
The marker-to-anchor offset is measured from their simultaneous camera poses at
registration; no manual ruler offset or alignment to the marker is required.
Geometry is computed from RAW marker pixels with lens distortion accounted for;
only the final human preview rotates. The stored offset never changes on marker loss.

This tests marker-based rigid tracking, not continuous anchor re-identification.
If the plate slips independently, the overlay follows the predicted attached pose
and cannot detect the slip. Start flat; a single small planar marker can produce
noisy depth/tilt estimates, especially at grazing angles or under occlusion. Marker
positive-depth and reprojection checks remain active. No hand detector is added;
once bound, a visible marker can still be tracked while a hand is present.
