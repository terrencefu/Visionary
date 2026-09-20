"""Keep OpenCV window events on the main thread while CPU checks run."""
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
import config
from hardware.camera import show_preview


def blank_projector(projector):
    # Projector.black() calls waitKey and discards its result. In the live loop
    # that steals Space/V/Enter before the actual keyboard handler can see them.
    cv2.imshow(projector.name, np.zeros((config.PROJECTOR_SIZE[1],config.PROJECTOR_SIZE[0],3),np.uint8))


def pump_events():
    key = cv2.waitKey(1) & 0xff
    if key in (27, ord('q')):
        raise KeyboardInterrupt


def run_check(frame, label, function, *args, **kwargs):
    print(label + '... please wait.', flush=True)
    show_preview(frame, label + '... please wait; Q/Esc exits')
    quit_requested = False
    # Only numerical work runs here: imshow/waitKey/camera access stay on main.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(function, *args, **kwargs)
        while not future.done():
            key = cv2.waitKey(20) & 0xff
            if key in (27, ord('q')):
                quit_requested = True
            # Ignore repeated step keys while busy rather than queueing advances.
        if quit_requested:
            raise KeyboardInterrupt
        return future.result()
