# Upload pan/tilt firmware (Arduino UNO R4 WiFi)

The sketch is `pan_tilt_servos/pan_tilt_servos.ino`. It is already on main and
on `codex/aruco-pan-tilt-tracking`. It uses USB serial, not Wi-Fi. Commands apply
immediately; there is no acceleration ramp.

## Windows Arduino IDE

1. Install **Arduino UNO R4 Boards** using Boards Manager.
2. Install **Adafruit PWM Servo Driver Library** using Library Manager, including
   its dependency **Adafruit BusIO** when prompted. `Wire` comes with the board core.
3. Open `pan_tilt_servos/pan_tilt_servos.ino` in Arduino IDE.
4. Select **Arduino UNO R4 WiFi** and its actual **COM** port.
5. Upload. This flashes the connected board; pulling GitHub alone does not.
6. Open Serial Monitor at **115200 baud**, select **Newline**, and send `STATUS`.
   The sketch starts with outputs OFF. Expected initial response:
   ```text
   Pan: commanded 1500 us; OFF
   Tilt: commanded 1000 us; OFF
   ```
7. Close Serial Monitor before starting Python so it releases the COM port.

The controller expects a PCA9685 at I2C address `0x40`, pan on channel **0**,
tilt on channel **1**, and 50 Hz servo pulses. Keep the existing verified wiring
and external servo supply. The first position command enables the output and
can move immediately; startup STATUS values are stored commands, not measured angles.

## Webcam tracking on the tracking branch

With local work saved, fetch and switch to the tracking branch, then pull it:

```powershell
git fetch origin
git switch codex/aruco-pan-tilt-tracking
git pull --ff-only origin codex/aruco-pan-tilt-tracking
conda activate hackthenorth
python -m pip install -r requirements.txt
python main.py follow-board
```

Preview sends no servo commands. The camera calibration must match the actual
camera and 1920x1080 capture. For live control, connect both the mounted webcam
and Arduino USB to the laptop running Python. Find its port:

```powershell
python -m serial.tools.list_ports
```

Replace COM5 with that port and test small pan corrections:

```powershell
python main.py follow-board --live --port COM5 --pan-only --pan-sign 1 --max-step 2 --interval 0.3
```

Click the camera preview and press Space to start/pause; Q/Esc exits and holds.
If the board moves away from the target cross, pause/quit and reverse the sign
to `--pan-sign -1`. First enable may jump to the Arduino's reported position
(P1500/T1000 after reset); ensure that position is clear. Add tilt only after
checking its direction. See the tracking README for the full procedure.

The existing `placement`, `perceive`, and `moving-base` MVP commands do not
activate servo tracking. Do not launch a second process competing for the same
camera. `follow-board --project` provides the separate tracking/board-overlay demo,
not the full assembly workflow. ID 5 is reserved for the moving assembly base;
`follow-board` currently tracks the fixed multi-marker board (at least 3 markers).
