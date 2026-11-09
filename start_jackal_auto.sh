#!/usr/bin/env bash
set -Eeuo pipefail

export IGN_PARTITION=jackal_rl
export GZ_PARTITION=jackal_rl
export IGN_IP=127.0.0.1
export GZ_IP=127.0.0.1

cd "$(dirname "$0")"

# ROS setup scripts may reference unset environment variables.
# Temporarily disable nounset while sourcing them.
set +u
source /opt/ros/humble/setup.bash
source jackal_rl_ws/install/setup.bash
set -u

WORLD="jackal_obstacle_test"
ROBOT="j100_0000"
URDF="/tmp/jackal_visible.urdf"

export IGN_GAZEBO_SYSTEM_PLUGIN_PATH="/opt/ros/humble/lib:${IGN_GAZEBO_SYSTEM_PLUGIN_PATH:-}"
export IGN_GAZEBO_RESOURCE_PATH="/opt/ros/humble/share:$PWD/jackal_rl_ws/install:$PWD/jackal_rl_ws/src:${IGN_GAZEBO_RESOURCE_PATH:-}"

wait_for() {
    local label="$1"
    local attempts="$2"
    shift 2

    for ((i=1; i<=attempts; i++)); do
        if "$@" >/dev/null 2>&1; then
            echo "READY: $label"
            return 0
        fi
        sleep 1
    done

    echo "ERROR: Timed out waiting for $label"
    return 1
}

echo "========== AUTOMATIC JACKAL STARTUP =========="

echo "[1/7] Checking existing Gazebo"
if pgrep -f '^ign gazebo server' >/dev/null; then
    echo "ERROR: Gazebo is already running."
    echo "Close the existing simulation before starting this script."
    exit 1
fi

echo "[2/7] Generating simulation URDF"
ros2 run xacro xacro \
    jackal_rl_ws/config/jackal/robot.urdf.xacro \
    is_sim:=true \
    namespace:="$ROBOT" \
    gazebo_controllers:="$PWD/jackal_rl_ws/config/jackal/platform/config/control.yaml" \
    -o /tmp/jackal_simulation_ready.urdf

python3 - <<'PY'
from pathlib import Path
import re

src = Path("/tmp/jackal_simulation_ready.urdf")
dst = Path("/tmp/jackal_visible.urdf")
base = Path("/opt/ros/humble/share/clearpath_platform_description")

def replace(match):
    path = base / match.group(1)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.as_uri()

text = re.sub(
    r"(?:package://|model://)clearpath_platform_description/([^\s\"<>]+)",
    replace,
    src.read_text()
)
dst.write_text(text)
print(f"Generated {dst}")
PY

echo "[3/7] Starting robot_state_publisher"
nohup ros2 run robot_state_publisher robot_state_publisher \
    --ros-args \
    -r __ns:=/"$ROBOT" \
    -p use_sim_time:=true \
    -p robot_description:="$(cat "$URDF")" \
    > /tmp/jackal_robot_state_publisher.log 2>&1 \
    < /dev/null &

wait_for "robot_state_publisher" 20 \
    ros2 node info "/$ROBOT/robot_state_publisher"

echo "[4/7] Starting Gazebo Fortress"
nohup ign gazebo -r \
    jackal_rl_ws/worlds/jackal_obstacle_test.sdf \
    > /tmp/jackal_gazebo_control.log 2>&1 \
    < /dev/null &

wait_for "Gazebo world" 30 \
    ign service -l

echo "[5/7] Spawning visible Jackal"
timeout 40 ros2 run ros_gz_sim create \
    -world "$WORLD" \
    -name "$ROBOT" \
    -file "$URDF" \
    -x 0 -y 0 -z 0.3

wait_for "controller manager" 40 \
    ros2 service type "/$ROBOT/controller_manager/list_controllers"

echo "[6/7] Activating wheel controllers"
timeout 30 ros2 run controller_manager spawner \
    joint_state_broadcaster \
    --controller-manager "/$ROBOT/controller_manager"

timeout 30 ros2 run controller_manager spawner \
    platform_velocity_controller \
    --controller-manager "/$ROBOT/controller_manager"

echo "[7/7] Starting ROS-Gazebo bridges"
nohup ros2 run ros_gz_bridge parameter_bridge \
    '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock' \
    > /tmp/jackal_clock_bridge.log 2>&1 \
    < /dev/null &

nohup ros2 run ros_gz_bridge parameter_bridge \
    "/$ROBOT/sensors/lidar3d_0/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan" \
    > /tmp/jackal_lidar_bridge.log 2>&1 \
    < /dev/null &

echo
echo "=== CONTROLLER STATUS ==="
ros2 control list_controllers \
    -c "/$ROBOT/controller_manager"

echo
echo "========== JACKAL STARTUP COMPLETE =========="
echo "Gazebo is running in the background."
echo "Robot: $ROBOT"
echo "World: $WORLD"
