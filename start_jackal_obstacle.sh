#!/usr/bin/env bash
set -eo pipefail

cd "$HOME/rl_robot_navigation"

source /opt/ros/humble/setup.bash
source jackal_rl_ws/install/setup.bash

export IGN_GAZEBO_RESOURCE_PATH="$PWD/jackal_rl_ws/install/clearpath_platform_description/share:/opt/ros/humble/share:${IGN_GAZEBO_RESOURCE_PATH:-}"
export GZ_SIM_RESOURCE_PATH="$IGN_GAZEBO_RESOURCE_PATH"

WORLD="jackal_obstacle_test"
URDF="/tmp/jackal_updated_lidar.urdf"

echo "Generating Jackal robot description..."
xacro jackal_rl_ws/config/jackal/robot.urdf.xacro > "$URDF"

echo "Starting Gazebo..."
ign gazebo -r "jackal_rl_ws/worlds/${WORLD}.sdf" &
GAZEBO_PID=$!

cleanup() {
    echo "Stopping Gazebo..."
    kill "$GAZEBO_PID" 2>/dev/null || true
    wait "$GAZEBO_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for Gazebo to become ready..."

READY=0
for i in $(seq 1 60); do
    if ign service -l | grep -q "/world/${WORLD}/create"; then
        READY=1
        break
    fi
    sleep 1
done

if [ "$READY" -ne 1 ]; then
    echo "ERROR: Gazebo world creation service unavailable."
    exit 1
fi

echo "Spawning Jackal automatically..."

ros2 run ros_gz_sim create \
    -world "$WORLD" \
    -name j100_0000 \
    -file "$URDF" \
    -x 0 -y 0 -z 0.3

echo
echo "========================================"
echo "Gazebo + Jackal startup complete"
echo "World: $WORLD"
echo "Robot: j100_0000"
echo "LiDAR minimum range: 0.1 m"
echo "========================================"

wait "$GAZEBO_PID"
