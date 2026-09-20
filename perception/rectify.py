"""Metric top-down view of the board plane. Match here, never in raw image space.

The camera looks at the workspace from 30-45 degrees, so the raw image is a bad
place to compare a part against its catalogue outline. Three things are wrong
with it at once:

  * **Foreshortening.** A tilted plane compresses by roughly cos(elevation) --
    about 29% at 45 degrees -- so a rotated template never fits.
  * **Handedness.** A valid overhead camera shows board +X to the right but
    board +Y UPWARD, while part_catalog draws outlines with y DOWN. Matching a
    template in raw pixels therefore compares a shape against its own mirror,
    and no in-plane rotation can undo that. Chiral parts silently never match,
    symmetric ones keep working, and the failure looks like anything but a
    coordinate bug.
  * **Scale.** px/mm varies across a tilted view, so one number cannot describe it.

Rectifying fixes all three by construction. In the rectified view board +X is
right, board +Y is DOWN, and px/mm is an exact constant we picked -- which is
precisely the frame part_catalog's outlines are drawn in.

A rectification is tied to ONE plane height. Parts resting on an earlier layer
must be rectified at their own z, or they are measured on the wrong plane.
"""
import cv2
import numpy as np


def board_to_image_homography(rvec, tvec, K, z_mm=0.0):
    """Homography mapping board (x_mm, y_mm) on the plane z=z_mm to image pixels.

    Pass undistorted-image K and use it on undistorted pixels; the homography
    cannot represent lens distortion.
    """
    R = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    t = np.asarray(tvec, float).reshape(3)
    # A point (x, y, z_mm) in camera coords is R[:,0]x + R[:,1]y + (R[:,2]z + t).
    return np.asarray(K, float) @ np.column_stack([R[:, 0], R[:, 1], R[:, 2] * float(z_mm) + t])


class BoardRectifier:
    """A metric top-down window onto the board plane.

    origin_mm is the board point at rectified pixel (0, 0); px_per_mm is exact.
    """

    def __init__(self, rvec, tvec, K, origin_mm, size_mm, px_per_mm, z_mm=0.0):
        if px_per_mm is None or px_per_mm <= 0:
            raise ValueError("px_per_mm must be positive; use .around(...) to pick one.")
        self.rvec, self.tvec, self.K = rvec, tvec, np.asarray(K, float)
        self.origin_mm = np.asarray(origin_mm, float).reshape(2)
        self.size_mm = np.asarray(size_mm, float).reshape(2)
        self.px_per_mm = float(px_per_mm)
        self.z_mm = float(z_mm)

        self.width = max(1, int(round(self.size_mm[0] * self.px_per_mm)))
        self.height = max(1, int(round(self.size_mm[1] * self.px_per_mm)))

        # rect pixel -> board mm: board = origin + rect / scale  (board +Y is DOWN
        # in the rectified view, matching part_catalog's drawing convention).
        self._board_from_rect = np.array([[1.0 / self.px_per_mm, 0.0, self.origin_mm[0]],
                                          [0.0, 1.0 / self.px_per_mm, self.origin_mm[1]],
                                          [0.0, 0.0, 1.0]])
        self._image_from_rect = (board_to_image_homography(rvec, tvec, self.K, self.z_mm)
                                 @ self._board_from_rect)
        self._rect_from_image = np.linalg.inv(self._image_from_rect)

    # --- construction ---------------------------------------------------------

    @classmethod
    def around(cls, rvec, tvec, K, centre_mm, radius_mm, px_per_mm=None, z_mm=0.0):
        """A square window centred on a board point.

        px_per_mm=None picks the local image scale, so the rectified view neither
        invents detail nor throws any away.
        """
        centre = np.asarray(centre_mm, float).reshape(2)
        if px_per_mm is None:
            px_per_mm = local_scale_px_per_mm(rvec, tvec, K, centre, z_mm)
        return cls(rvec, tvec, K, origin_mm=centre - radius_mm,
                   size_mm=(2 * radius_mm, 2 * radius_mm), px_per_mm=px_per_mm, z_mm=z_mm)

    @classmethod
    def for_region(cls, rvec, tvec, K, bbox_px, pad_mm=12.0, px_per_mm=None, z_mm=0.0):
        """A window covering an image-space bbox, mapped onto the board plane."""
        x, y, w, h = (float(v) for v in bbox_px)
        corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        H_inv = np.linalg.inv(board_to_image_homography(rvec, tvec, K, z_mm))
        board = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        low, high = board.min(axis=0) - pad_mm, board.max(axis=0) + pad_mm
        centre = (low + high) / 2.0
        radius = float(np.max(high - low)) / 2.0
        return cls.around(rvec, tvec, K, centre, radius, px_per_mm, z_mm)

    # --- coordinates ----------------------------------------------------------

    @property
    def centre_px(self):
        return (self.width / 2.0, self.height / 2.0)

    def to_board(self, x_px, y_px):
        """Rectified pixel -> board millimetres."""
        return self.origin_mm + np.array([float(x_px), float(y_px)]) / self.px_per_mm

    def to_rect(self, x_mm, y_mm):
        """Board millimetres -> rectified pixel."""
        return (np.array([float(x_mm), float(y_mm)]) - self.origin_mm) * self.px_per_mm

    def theta_to_board(self, theta_rect_deg):
        """Rectified in-plane angle -> board angle.

        align_outline measures CCW on screen with y downward. Board +Y is
        downward here, so screen-CCW runs from board +X toward board -Y, which is
        the opposite sense to the board convention (CCW from +X toward +Y).
        """
        return float((-float(theta_rect_deg)) % 360.0)

    def theta_from_board(self, theta_board_deg):
        return float((-float(theta_board_deg)) % 360.0)

    # --- the image ------------------------------------------------------------

    def warp(self, undistorted_frame):
        """Rectified patch. The frame must already be undistorted in the same K."""
        return cv2.warpPerspective(undistorted_frame, self._rect_from_image,
                                   (self.width, self.height), flags=cv2.INTER_LINEAR)


def local_scale_px_per_mm(rvec, tvec, K, point_mm, z_mm=0.0):
    """Image pixels per board millimetre at a board point, averaged over both axes.

    Measured from the board pose, so it follows tilt and perspective.
    """
    point = np.asarray(point_mm, float).reshape(2)
    samples = np.array([[point[0], point[1], z_mm],
                        [point[0] + 1.0, point[1], z_mm],
                        [point[0], point[1] + 1.0, z_mm]])
    projected = cv2.projectPoints(samples, rvec, tvec, np.asarray(K, float),
                                  np.zeros(5))[0].reshape(-1, 2)
    scale = 0.5 * (float(np.linalg.norm(projected[1] - projected[0]))
                   + float(np.linalg.norm(projected[2] - projected[0])))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Board scale is degenerate at this point.")
    return scale
