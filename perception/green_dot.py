from dataclasses import dataclass
import cv2
import numpy as np
import config


@dataclass
class Dot:
    u: float
    v: float
    contour: np.ndarray
    area: float


def detect_green_dot(background, illuminated):
    if background.shape != illuminated.shape:
        raise ValueError("Background and illuminated frame sizes differ.")
    old = background.astype(np.int16)
    new = illuminated.astype(np.int16)
    hsv = cv2.cvtColor(illuminated, cv2.COLOR_BGR2HSV)
    if config.DOT_COLOR == 'green':
        increase = new[:, :, 1] - old[:, :, 1]
        dominance = new[:, :, 1] - np.maximum(new[:, :, 0], new[:, :, 2])
        hue = (hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 90)
    elif config.DOT_COLOR == 'magenta':
        increase = np.minimum(new[:,:,0]-old[:,:,0], new[:,:,2]-old[:,:,2])
        dominance = np.minimum(new[:,:,0],new[:,:,2])-new[:,:,1]
        # OpenCV uint8 H is 0..179 (not 0..360); magenta is around 150.
        # This range does not cross the red hue wrap at 179 -> 0.
        hue = (hsv[:,:,0] >= 130) & (hsv[:,:,0] <= 175)
    else:
        raise ValueError('Dot color must be green or magenta.')
    mask = ((increase >= config.MIN_GREEN_INCREASE) &
            (dominance >= config.MIN_GREEN_DOMINANCE) & hue &
            (hsv[:,:,1] >= 50) & (hsv[:,:,2] >= 50)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not config.MIN_DOT_AREA_PX <= area <= config.MAX_DOT_AREA_PX:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 0 or y <= 0 or x+w >= mask.shape[1] or y+h >= mask.shape[0]:
            continue  # clipped dot has a biased centroid
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0 or 4*np.pi*area/perimeter**2 < 0.2:
            continue
        moments = cv2.moments(contour)
        candidates.append(Dot(moments["m10"]/moments["m00"],
                              moments["m01"]/moments["m00"], contour, area))
    candidates.sort(key=lambda item: item.area, reverse=True)
    # Similar strong blobs are ambiguous; do not silently select a reflection.
    if not candidates or (len(candidates) > 1 and candidates[1].area > 0.5*candidates[0].area):
        return None, mask
    return candidates[0], mask
