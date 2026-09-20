import ctypes
import os
import cv2
import numpy as np
from screeninfo import get_monitors
import config


def monitors():
    if os.name == "nt":
        # Set before creating ANY windows: monitor coordinates must be physical pixels.
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    result = get_monitors()
    for i, m in enumerate(result):
        print(f"Display {i}: {m.width}x{m.height} at ({m.x},{m.y}), primary={m.is_primary}, {m.name}")
    return result


class Projector:
    name = "Projector output"

    def __enter__(self):
        displays = monitors()
        idx = config.PROJECTOR_MONITOR_INDEX
        if idx is None:
            candidates = [i for i, m in enumerate(displays) if not m.is_primary]
            if len(displays) < 2 or len(candidates) != 1:
                raise RuntimeError("Use Windows EXTEND mode and set PROJECTOR_MONITOR_INDEX explicitly if ambiguous.")
            idx = candidates[0]
        if not 0 <= idx < len(displays) or displays[idx].is_primary:
            raise ValueError("Choose a non-primary projector display.")
        m = displays[idx]
        if (m.width, m.height) != config.PROJECTOR_SIZE:
            raise ValueError(f"Projector is {m.width}x{m.height}; config expects {config.PROJECTOR_SIZE}.")
        self.width, self.height = config.PROJECTOR_SIZE
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.name, m.x, m.y)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.black()
        return self

    def black(self):
        cv2.imshow(self.name, np.zeros((self.height, self.width, 3), dtype=np.uint8))
        cv2.waitKey(1)

    def dot(self, u, v):
        if not np.isfinite([u, v]).all():
            raise ValueError("Invalid projector pixel.")
        u, v = int(round(u)), int(round(v))
        r = config.DOT_RADIUS_PX
        if not r <= u < self.width-r or not r <= v < self.height-r:
            raise ValueError(f"Dot ({u},{v}) is outside projector bounds.")
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        colors = {'green': (0,255,0), 'magenta': (255,0,255)}
        if config.DOT_COLOR not in colors:
            raise ValueError('Dot color must be green or magenta.')
        cv2.circle(frame, (u, v), r, colors[config.DOT_COLOR], -1)
        cv2.imshow(self.name, frame)
        cv2.waitKey(1)
        return u, v  # exact rendered integer pixel center

    def __exit__(self, *_):
        self.black()
        cv2.destroyWindow(self.name)
