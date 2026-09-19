import hashlib
import json
import numpy as np
import config


def load_camera():
    path = config.CAMERA_CALIBRATION
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}. Run python -m calibration.calibrate_camera.")
    with np.load(path, allow_pickle=False) as data:
        if "image_size" not in data or tuple(data["image_size"]) != config.CAMERA_SIZE:
            raise ValueError("Camera calibration must include the correct image_size; recalibrate.")
        K = data["camera_matrix"].astype(float)
        dist = data["dist_coeffs"].astype(float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or not np.isfinite(dist).all():
        raise ValueError("Invalid camera calibration arrays.")
    return K, dist


def fixture_signature():
    config.validate_fixture()
    payload = dict(size=config.MARKER_SIZE_MM, centers=config.MARKER_CENTERS_MM,
                   bounds=config.BOARD_BOUNDS_MM, dictionary=config.ARUCO_DICTIONARY)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def camera_signature():
    return hashlib.sha256(config.CAMERA_CALIBRATION.read_bytes()).hexdigest()


def inside_board(point, margin=0.0):
    x0, y0, x1, y1 = config.BOARD_BOUNDS_MM
    x, y = point[:2]
    return x0 + margin <= x <= x1 - margin and y0 + margin <= y <= y1 - margin


def run_cli(main):
    import cv2
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.")
    except (ValueError, RuntimeError, FileNotFoundError, OSError, cv2.error) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    finally:
        cv2.destroyAllWindows()
