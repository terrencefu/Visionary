"""Capture is always RAW; preview rotation never changes geometric inputs."""
import os
import time
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


def show_preview(frame, text="", name="Camera preview"):
    view = cv2.rotate(frame, cv2.ROTATE_180) if config.PREVIEW_ROTATE_180 else frame.copy()
    # Human-readable text after rotation; geometry annotations before rotation.
    if text:
        cv2.putText(view, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, view)
