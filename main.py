"""Small launcher for independently testable subsystems."""
import argparse
import runpy
import sys

TOOLS = {
    "follow-board": "tracking.follow_board",
    "webcam": "calibration.webcam_smoketest",
    "camera-calibration": "calibration.calibrate_camera",
    "aruco": "calibration.aruco_pose_test",
    "projector": "calibration.projector_smoketest",
    "dot": "calibration.projector_camera_validation",
    "collect": "calibration.collect_projector_calibration",
    "diagnose": "calibration.check_calibration_correspondences",
    "solve": "calibration.solve_projector_calibration",
    "validate": "calibration.validate_world_projection",
    "planar": "calibration.planar",
    "placement": "projection.placement",
    "board-overlay": "projection.guidance",
    "perceive": "assembly.demo_perception",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assembly copilot calibration tools. Begin with webcam.")
    parser.add_argument("tool", choices=TOOLS)
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    sys.argv = [sys.argv[0], *remaining]
    runpy.run_module(TOOLS[args.tool], run_name="__main__")
