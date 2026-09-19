"""Capture is always RAW; preview rotation never changes geometric inputs."""
import os
import time
import textwrap
import cv2
import config


class Camera:
    def __enter__(self):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
        try:
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {config.CAMERA_INDEX}.")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_SIZE[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_SIZE[1])
            self.read()
            print(f"Camera {config.CAMERA_INDEX}: {config.CAMERA_SIZE}, RAW geometry")
            return self
        except Exception:
            self.cap.release()
            raise

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera capture failed.")
        actual = (frame.shape[1], frame.shape[0])
        if actual != config.CAMERA_SIZE:
            raise RuntimeError(f"Camera returned {actual}; require {config.CAMERA_SIZE}. Do not resize for calibration.")
        return frame

    def settled_frame(self):
        deadline = time.monotonic() + config.SETTLE_SECONDS
        while True:
            frame = self.read()  # drain buffered/exposure-transition frames
            if cv2.waitKey(1) & 0xFF == 27:
                raise KeyboardInterrupt
            if time.monotonic() >= deadline:
                return frame

    def __exit__(self, *_):
        self.cap.release()


def preview_frame(frame, text="", debug_lines=None, rotate=None):
    """Create a display copy; all input geometry and pixel coordinates stay RAW."""
    rotate = config.PREVIEW_ROTATE_180 if rotate is None else rotate
    view = cv2.rotate(frame, cv2.ROTATE_180) if rotate else frame.copy()
    # Human-readable text after rotation; geometry annotations before rotation.
    if text:
        cv2.putText(view, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if debug_lines:
        # Put diagnostics beside the image so the panel never hides a marker.
        lines = [line for item in debug_lines for line in (textwrap.wrap(item, 86) or [""])]
        import numpy as np
        panel = np.zeros((max(view.shape[0], 30 + 25*len(lines)), 870, 3), dtype=np.uint8)
        for i, line in enumerate(lines):
            cv2.putText(panel, line, (12, 28 + 25*i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (240, 240, 240), 1, cv2.LINE_AA)
        canvas = np.zeros((panel.shape[0], view.shape[1] + panel.shape[1], 3), dtype=np.uint8)
        canvas[:view.shape[0], :view.shape[1]] = view
        canvas[:, view.shape[1]:] = panel
        view = canvas
    return view


def show_preview(frame, text="", name="Camera preview", debug_lines=None, rotate=None):
    view = preview_frame(frame, text, debug_lines, rotate)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, view)
