#!/usr/bin/env bash

echo "Checking environment..."
echo

if [ -f /opt/ros/humble/setup.bash ]; then
    echo "[OK] ROS 2 Humble"
else
    echo "[MISSING] ROS 2 Humble"
fi

if command -v ign >/dev/null 2>&1; then
    echo "[OK] Ignition / Gazebo"
    ign gazebo --versions 2>/dev/null || true
else
    echo "[MISSING] Ignition Gazebo"
fi

if command -v vcs >/dev/null 2>&1; then
    echo "[OK] vcstool"
else
    echo "[MISSING] vcstool"
fi

if command -v colcon >/dev/null 2>&1; then
    echo "[OK] colcon"
else
    echo "[MISSING] colcon"
fi

if command -v rosdep >/dev/null 2>&1; then
    echo "[OK] rosdep"
else
    echo "[MISSING] rosdep"
fi
