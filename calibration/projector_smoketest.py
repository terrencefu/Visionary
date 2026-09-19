import argparse
import cv2
from hardware.projector import Projector, monitors
from calibration.common import run_cli


def main():
    parser = argparse.ArgumentParser(description="Exact projector-center dot on a second display.")
    parser.add_argument("--list", action="store_true", help="Only list monitors.")
    args = parser.parse_args()
    if args.list:
        monitors()
        return
    with Projector() as projector:
        print("B: black; D: center dot; Esc: exit. Check full-screen output and pixel mapping physically.")
        print("Center:", projector.dot(projector.width//2, projector.height//2))
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('b'):
                projector.black()
            elif key == ord('d'):
                projector.dot(projector.width//2, projector.height//2)


if __name__ == "__main__":
    run_cli(main)
