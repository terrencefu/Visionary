import argparse
import cv2
from hardware.camera import Camera, show_preview
from calibration.common import run_cli


def main():
    parser = argparse.ArgumentParser(description="Verify raw 1920x1080 camera capture.")
    parser.add_argument("--frames", type=int, default=0, help="Exit after N frames (0 = interactive).")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    if args.frames < 0 or (args.no_preview and args.frames == 0):
        raise ValueError("For --no-preview specify a positive --frames.")
    with Camera() as camera:
        count = 0
        while True:
            raw = camera.read()
            count += 1
            if not args.no_preview:
                show_preview(raw, "RAW capture 1920x1080 | preview rotated only | Esc exits")
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
            if args.frames and count >= args.frames:
                break
        print(f"PASS: captured {count} raw frames at {raw.shape[1]}x{raw.shape[0]}.")


if __name__ == "__main__":
    run_cli(main)
