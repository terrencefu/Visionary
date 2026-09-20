# LEGO guidance using the current 3D calibration

```powershell
python main.py placement --part-id LEGO-2x4 --center 150 100 --width 32 --height 16 --rotation 0
python main.py board-overlay
```

Press Space when READY; Escape/Q exits. Both commands load projector_calibration.npz
and recompute projector coordinates from the current RAW ArUco pose on every frame.
They do not read planar_calibration.json or compare against its old reference pose.
The rigid camera/projector head can move relative to the board. Its internal mounting
and projector settings must stay unchanged. No servo commands are sent.

Lost tracking immediately blanks projection and allows two seconds for recovery.
Invalid-depth or off-screen guidance stays blank. Existing calibration signatures,
pose-quality checks and workspace bounds remain enforced. Board-space edges are
sampled through the projector distortion model. The placement footprint is 32 by
16 mm; rotation turns +X toward +Y. The board overlay shows red marker-center
outlines and green workspace bounds.

The optional --stationary-ready flag is accepted for old command compatibility;
3D guidance does not require returning to an old planar calibration position.
The explicitly named planar calibration tools remain available for historical work;
they are not used by these guidance commands. Existing calibration files are unchanged.
