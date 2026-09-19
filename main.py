"""Small launcher for independently testable subsystems."""
import argparse
import runpy
import sys

TOOLS = {
    "webcam": "calibration.webcam_smoketest",
    "camera-calibration": "calibration.calibrate_camera",
    "aruco": "calibration.aruco_pose_test",
    "projector": "calibration.projector_smoketest",
    "dot": "calibration.projector_camera_validation",
    "collect": "calibration.collect_projector_calibration",
    "diagnose": "calibration.check_calibration_correspondences",
    "solve": "calibration.solve_projector_calibration",
    "validate": "calibration.validate_world_projection",
    "perceive": "assembly.demo_perception",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assembly copilot calibration tools. Begin with webcam.")
    parser.add_argument("tool", choices=TOOLS)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_module(TOOLS[args.tool], run_name="__main__")
