"""Strict, data-driven CAD assembly entry point."""
from assembly.demo_perception import main
from calibration.common import run_cli

if __name__=='__main__':
    run_cli(lambda: main(integrated=True))
