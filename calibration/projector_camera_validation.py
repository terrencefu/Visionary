"""Green-dot measurement smoke test; no geometry or projector calibration required."""
import cv2
from hardware.camera import Camera, show_preview
from hardware.projector import Projector
from perception.green_dot import detect_green_dot
from calibration.common import run_cli


def main():
    with Projector() as projector, Camera() as camera:
        while True:
            projector.black()
            background = camera.settled_frame()
            uv = projector.dot(projector.width//2, projector.height//2)
            raw = camera.settled_frame()
            dot, mask = detect_green_dot(background, raw)
            view = raw.copy()
            if dot:
                cv2.circle(view, (round(dot.u), round(dot.v)), 12, (0, 0, 255), 2)
                print(f"Projector {uv} -> RAW camera ({dot.u:.2f}, {dot.v:.2f}), area {dot.area:.0f} px")
            else:
                print("No unambiguous green dot. Inspect exposure, reflections, surface, thresholds.")
            show_preview(view, "Space: repeat | Esc: exit")
            show_preview(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), name="Green detection mask")
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord('q')):
                break


if __name__ == "__main__":
    run_cli(main)
