# Jackal Actor Simulation

<p align="center">
  <img src="assets/gazebo.png" alt="Gazebo simulation showing the Jackal robot, actor, obstacles, and teleoperation panel">
</p>

<p align="center">
  <img src="assets/rviz2.png" alt="RViz visualization showing Jackal sensor point clouds and camera view">
</p>

ROS 2 Humble / Gazebo Fortress simulation containing:

- Clearpath Jackal J100
- Intel RealSense D435 RGB-D camera
- Velodyne VLP-16 3D LiDAR
- Scripted Gazebo human actor
- Custom Gazebo environment
- RViz visualization

## Tested Environment

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Fortress
- Clearpath Jackal J100

## Repository Structure

```text
jackal_actor_sim/
├── config/
│   └── robot.yaml
├── worlds/
│   └── jackal_actor_world.sdf
├── patches/
│   └── clearpath_spawn_world.patch
├── scripts/
│   ├── check_environment.sh
│   └── setup_workspace.sh
└── dependencies.repos
```

## Prerequisites

ROS 2 Humble and Gazebo Fortress must already be installed.

Install basic development tools:
```bash
sudo apt update

sudo apt install -y \
    git \
    python3-vcstool \
    python3-rosdep \
    python3-colcon-common-extensions
```
## Installation

Clone this repository:
```bash
cd ~

git clone https://github.com/putraaryawinata/jackal_actor_sim.git

cd jackal_actor_sim
```

Check the environment:
```bash
./scripts/check_environment.sh
```
Create and build the ROS workspace:
```bash
./scripts/setup_workspace.sh
```
Source it:
```bash
source /opt/ros/humble/setup.bash
source ~/jackal_actor_ws/install/setup.bash
```

## Run the simulation
Terminal 1 — Gazebo
```bash
source /opt/ros/humble/setup.bash
source ~/jackal_actor_ws/install/setup.bash

ros2 launch clearpath_gz simulation.launch.py \
    world:=jackal_actor_world \
    x:=0.0 \
    y:=0.0 \
    z:=0.3 \
    yaw:=0.0 \
    rviz:=false
```
Terminal 2 — RViz
```bash
source /opt/ros/humble/setup.bash
source ~/jackal_actor_ws/install/setup.bash

ros2 launch clearpath_viz view_robot.launch.py \
    namespace:=j100_0000 \
    use_sim_time:=true
```
## Sensor Topics

Check available sensor topics:
```bash
ros2 topic list | grep -E 'camera|lidar|points'
```
Expected topics include:
```text
/j100_0000/sensors/camera_0/color/image
/j100_0000/sensors/camera_0/depth/image
/j100_0000/sensors/camera_0/points
/j100_0000/sensors/lidar3d_0/points
```
Check the Velodyne:
```bash
ros2 topic hz /j100_0000/sensors/lidar3d_0/points
```
Check the RGB camera:
```bash
ros2 topic hz /j100_0000/sensors/camera_0/color/image
```

## Teleoperation
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
    --ros-args \
    -r __ns:=/j100_0000
```

## Notes
This repository does not vendor the Clearpath ROS repositories. Exact
dependencies are recorded in dependencies.repos and imported using
vcstool.

The repository also applies a small patch to the ROS 2 Humble Clearpath
Gazebo spawn launch file so that the robot is explicitly spawned into the
requested Gazebo world.
