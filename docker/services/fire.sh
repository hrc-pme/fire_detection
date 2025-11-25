#!/bin/bash

source /opt/ros/humble/setup.bash

# 啟動 foxglove_bridge
# ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765 &

python3 /workspace/src/ros2_fire_detector.py &
python3 /workspace/src/move2fire.py
