"""Small launcher for independently testable subsystems."""
import argparse
import runpy
import sys

TOOLS = {
    "webcam": "webcam_smoketest",
    "camera-calibration": "calibrate_camera",
    "aruco": "aruco_pose_test",
    "projector": "projector_smoketest",
    "dot": "projector_camera_validation",
    "collect": "collect_projector_calibration",
    "diagnose": "check_calibration_correspondences",
    "solve": "solve_projector_calibration",
    "validate": "validate_world_projection",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assembly copilot calibration tools. Begin with webcam.")
    parser.add_argument("tool", choices=TOOLS)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_module("calibration." + TOOLS[args.tool], run_name="__main__")
