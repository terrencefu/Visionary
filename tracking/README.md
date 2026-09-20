# Automatic ArUco board following

`follow-board` keeps the selected board's workspace centre at an aim point in the camera image by commanding the rigid pan/tilt head over USB. If the board or the whole head/base moves while enough markers remain visible, visual feedback corrects the aim. This is a hardware-untested prototype; synthetic control/geometry and mocked serial tests are provided.

## Start here

Use the team's `hackthenorth` environment from the repository root:

```sh
python -m pip install -r requirements.txt
python main.py follow-board
```

This opens a **preview only**, with no serial connection or movement. It needs your teammate's `camera_calibration_1080p.npz` and the actual board geometry in `config.py`. Calibration files are ignored by Git and must be copied locally. The current config, not older dimensions in the root README, is authoritative. The green circle is the projected centre of `BOARD_BOUNDS_MM`; the cyan cross is the aim point. Space starts/pauses the simulated controller.

Close Arduino Serial Monitor before live control. Keep the immediate-command `hardware/arduino/pan_tilt_servos/pan_tilt_servos.ino` already uploaded; this branch does not change the firmware or restore acceleration. List ports with:

```sh
python -m serial.tools.list_ports
```

For a first pan-only check, replace `COM5` with your port (macOS: `/dev/cu.usbmodem...`):

```sh
python main.py follow-board --live --port COM5 --pan-only --pan-sign 1 --max-step 2 --interval 0.3
```

Space enables movement **only when the board is visible**. Check that the green circle approaches the cyan cross. If it goes away, pause and rerun with `--pan-sign -1`. The physical servo mounting determines the sign; `1` above is a test value, not a measured direction. Once pan is correct, test tilt at the same small step and choose its sign similarly:

```sh
python main.py follow-board --live --port COM5 --pan-sign 1 --tilt-sign -1 --max-step 2 --interval 0.3
```

Replace both signs with the verified values. Then omit the step/interval options for the defaults: at most 12 us per correction, at least 0.15 s between corrections, 25 px deadband, gain 0.04 us/px. These are starting settings, not tuned gains. Each servo command takes effect immediately; the software only limits how far each feedback correction can jump. Reduce gain/step or increase interval if the head oscillates.

Space pauses (holds last command). Q/Esc exits (holds). **O disables both signals**, which may let the loaded mechanism drop; support it first. The program never automatically homes or sweeps. Opening USB may reset the Arduino; on first start its reported default pulses are P1500/T1000, not measured angles, so first enable can move directly there. Keep this known starting position clear. Use narrower verified `--pan-limits MIN MAX` and `--tilt-limits MIN MAX` when needed; reported starting pulses must be inside them. Firmware limits remain the outer bounds.

## Two boards

The default follows configured IDs 0,1,2,3. For a second board with **the same measured marker positions, size and orientation**, use unique IDs at corresponding locations:

```sh
python main.py follow-board --marker-ids 4 5 6 7
```

Add the live-control options once verified. The order maps new IDs to the existing 0,1,2,3 positions. Other board IDs are ignored, and the global calibration config is not rewritten. At least three selected markers must be visible. Different board geometry needs a separate measured map; do not assume this option handles it. Duplicate IDs on two visible boards are ambiguous and rejected. This command follows one selected board, not an automatic task-based board switcher.

## Projector integration

The camera and projector move together physically. By default the controller centres the board **in the camera**; their optical axes are offset, so that does not guarantee exact centring in the projector. Use `--aim RAW_X RAW_Y` to choose a camera aim point that gives good projector coverage at your working distance. The preview rotates 180 degrees for humans; aim coordinates and control signs always refer to the unrotated RAW image.

After rigid camera/projector calibration is available, add `--project` to the same command. It uses the existing 3D `board_scene`/`render_scene` and latest board pose to project the board overlay. It blanks during correction, marker loss, pause, or out-of-projector bounds, and waits `SETTLE_SECONDS` after centring before displaying. This is a timed software gate, not measured physical settling. The overlay labels retain the reference fixture IDs 0–3 even when following remapped IDs.

Do not run a second camera/projection process alongside this command. A saved stationary planar homography cannot be reused after head movement; this tool uses the existing 3D rigid transform instead. Relative movement between camera and projector still invalidates that transform. Pan-only mode only centres horizontally; check vertical coverage yourself.

## Behaviour and limits

- The target is the same geometric workspace centre even if one marker disappears; it is not the average of currently visible markers.
- Three consecutive accepted frames are required before corrections resume. Missing, duplicate, poor-fit or off-screen targets cause **no further commands**, holding the last pulse. No blind search when the board leaves view entirely.
- USB requests are acknowledged by the existing sketch. Rejection, reset or timeout exits tracking; already-commanded servos may still finish moving to their last position.
- Servo pulses are commanded positions, not encoder feedback. Firmware STATUS is used on connect; a PC-side pulse limit cannot establish physical clearance.
- The assembly change detector compares image baselines. If this controller is later connected to `AssemblyState`, pause validation during movement and capture a fresh baseline after settling; do not compare frames across head movement.

Run all tests with `python -m unittest discover -s tests -v`. New tests cover synthetic disturbance recovery with both servo signs, command limits and timing, marker loss/reacquisition, selected-board pose with a second board in view, partial visibility, and the serial protocol. Camera capture, servo direction, real convergence, clearance, settling and projection accuracy still require a mounted-rig test.

## Test without hardware or a teammate

```sh
python -m tracking.simulate
```

No camera, Arduino, or calibration file is needed. This renders actual ArUco marker images, runs `BoardTracker` and `FollowController`, and models servo response with an assumed 0.002 rad/us scale and 0.18 s lag. Scenarios include board movement, a rig orientation bump, and moving the board while markers are hidden. Outputs are `data/tracking_simulation/simulation.gif`, `results.png`, `report.json`, and the full `trace.json`. The animation plays at 5x simulated speed. The script exits with an error if nominal convergence, loss handling, or pulse limits fail.

For an intentionally reversed pan direction:

```sh
python -m tracking.simulate --wrong-sign --output data/tracking_simulation_wrong_sign
```

That scenario demonstrates why the physical direction check matters: the board can leave view, after which tracking holds rather than searching. These assumed mechanics cannot prove real-world convergence or projector accuracy. To test alone on hardware, leave the projector off, place the marked board in front of the mounted camera, run preview first, then the small-step pan-only command above. Move the board gently and cover the markers to check loss/recovery. The valid local camera calibration must match the camera actually connected.
