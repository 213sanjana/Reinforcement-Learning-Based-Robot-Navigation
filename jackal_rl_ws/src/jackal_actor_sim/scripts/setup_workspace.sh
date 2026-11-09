#!/usr/bin/env bash

set -e

# ---------------------------------------------------------
# Jackal Actor Simulation workspace setup
# Ubuntu 22.04 + ROS 2 Humble + Gazebo Fortress
# ---------------------------------------------------------

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default workspace:
# ~/jackal_actor_ws
#
# You can optionally provide another workspace:
# ./setup_workspace.sh ~/jackal_actor_ws_test

WS="${1:-$HOME/jackal_actor_ws}"

echo "=========================================="
echo "Jackal Actor Simulation Setup"
echo "=========================================="
echo "Repository : $REPO_DIR"
echo "Workspace  : $WS"
echo

# ---------------------------------------------------------
# Check ROS 2 Humble
# ---------------------------------------------------------

if [ ! -f /opt/ros/humble/setup.bash ]; then
    echo "ERROR: ROS 2 Humble was not found."
    echo "Expected: /opt/ros/humble/setup.bash"
    exit 1
fi

source /opt/ros/humble/setup.bash

# ---------------------------------------------------------
# Check required tools
# ---------------------------------------------------------

for command in git vcs rosdep colcon; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "ERROR: Required command '$command' was not found."
        exit 1
    fi
done

# ---------------------------------------------------------
# Create workspace
# ---------------------------------------------------------

echo "[1/6] Creating workspace..."

mkdir -p "$WS/src"

# ---------------------------------------------------------
# Import repositories
# ---------------------------------------------------------

echo "[2/6] Importing source repositories..."

cd "$WS"

vcs import src < "$REPO_DIR/dependencies.repos"

# ---------------------------------------------------------
# Install robot.yaml
# ---------------------------------------------------------

echo "[3/6] Installing Clearpath robot configuration..."

mkdir -p "$HOME/clearpath"

cp "$REPO_DIR/config/robot.yaml" \
   "$HOME/clearpath/robot.yaml"

# ---------------------------------------------------------
# Install custom Gazebo world
# ---------------------------------------------------------

echo "[4/6] Installing custom Gazebo world..."

CLEARPATH_SIM="$WS/src/clearpath_simulator"

if [ ! -d "$CLEARPATH_SIM" ]; then
    echo "ERROR: clearpath_simulator was not imported."
    exit 1
fi

cp "$REPO_DIR/worlds/jackal_actor_world.sdf" \
   "$CLEARPATH_SIM/clearpath_gz/worlds/jackal_actor_world.sdf"

# ---------------------------------------------------------
# Apply Humble world-spawn patch
# ---------------------------------------------------------

echo "[5/6] Applying Clearpath Gazebo spawn patch..."

cd "$CLEARPATH_SIM"

PATCH="$REPO_DIR/patches/clearpath_spawn_world.patch"

if git apply --check "$PATCH" >/dev/null 2>&1; then
    git apply "$PATCH"
    echo "Patch applied."
else
    echo "Patch could not be applied automatically."
    echo "It may already be applied or the upstream source may differ."
fi

# ---------------------------------------------------------
# Install dependencies and build
# ---------------------------------------------------------

echo "[6/6] Installing ROS dependencies..."

cd "$WS"

rosdep install \
    --from-paths src \
    --ignore-src \
    -r \
    -y \
    --rosdistro humble

echo
echo "Building workspace..."

colcon build --symlink-install

echo
echo "=========================================="
echo "Setup complete."
echo "=========================================="
echo
echo "Run:"
echo
echo "source /opt/ros/humble/setup.bash"
echo "source $WS/install/setup.bash"
echo
