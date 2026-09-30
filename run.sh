#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run the ZED2i Plane Visualizer
# Make sure the ZED camera is already publishing:
#   ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
# ─────────────────────────────────────────────────────────────────────────────
set -e
source /opt/ros/humble/setup.bash
cd "$(dirname "$0")"
python3 plane_visualizer.py
