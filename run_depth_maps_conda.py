#!/usr/bin/env python
"""Wrapper to run depth map generation in conda environment."""

import subprocess
import sys
import os

# Run the depth maps script with conda environment
env = os.environ.copy()

# Script to run
script = "generate_depth_maps.py"

print(f"Running {script} in conda environment...")
print("=" * 60)

# Just run it directly - the conda env should have opencv-contrib
result = subprocess.run(
    [sys.executable, script],
    cwd="c:\\Users\\anubh\\OneDrive\\Documents\\Security System",
    capture_output=False,
    text=True
)

sys.exit(result.returncode)
