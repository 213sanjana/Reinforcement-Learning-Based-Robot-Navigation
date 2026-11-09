# Reinforcement Learning-Based Robot Navigation

A robotics project focused on developing and evaluating reinforcement learning-based navigation for a simulated mobile robot using **ROS 2 Humble, Gazebo Fortress, and a Clearpath Jackal J100**.

The project explores how reinforcement learning can be applied to robot navigation through simulated environments, combining LiDAR-based perception, robot motion control, reward design, and navigation evaluation.

## Project Overview

The project includes two complementary navigation environments:

- **GridWorld:** A discrete environment for developing and evaluating reinforcement learning algorithms and classical path-planning methods.
- **Jackal Robot Simulation:** A ROS 2 and Gazebo-based environment for applying a trained PPO policy to mobile robot navigation.

The Jackal navigation interface uses a 40-dimensional observation space containing LiDAR measurements and robot state information. The PPO policy produces continuous linear and angular velocity commands for the simulated robot.

## Project Preview

### Jackal Robot Simulation

![Jackal Robot](images/jackal_robot.png)

### Industrial Warehouse Environment

![Industrial Warehouse](images/industrial_warehouse.png)

### Small Warehouse Environment

![Small Warehouse](images/small_warehouse.png)

### Café Environment

![Café Environment](images/cafe_environment.png)

*Add your actual project screenshots to the `images/` folder using the filenames shown above.*

## Demonstration Videos

The repository includes demonstrations of the robot navigating different simulated environments.

| Environment | Demonstration |
|---|---|
| Industrial Warehouse | [Watch Demo](videos/industrial_warehouse_demo.mp4) |
| Small Warehouse | [Watch Demo](videos/small_warehouse_demo.mp4) |
| Obstacle Course | [Watch Demo](videos/obstacle_course_demo.mp4) |
| Large Aisle Hall | [Watch Demo](videos/aisle_warehouse_demo.mp4) |
| Café Environment | [Watch Demo](videos/cafe_navigation_demo.mp4) |
| Ground Environment | [Watch Demo](videos/ground_demo.mp4) |
| Open Environment | [Watch Demo](videos/open_environment_demo.mp4) |

A separate Nav2-based point-to-point navigation demonstration is also included:

[Watch Nav2 Café Navigation](videos/nav2_cafe_demo.mp4)

## Technologies Used

- **Programming:** Python
- **Robotics Middleware:** ROS 2 Humble
- **Simulation:** Gazebo Fortress
- **Reinforcement Learning:** PPO, DQN, Double DQN, Q-Learning
- **Deep Learning:** PyTorch
- **RL Framework:** Stable-Baselines3
- **Environment Framework:** Gymnasium
- **Robot:** Clearpath Jackal J100
- **Perception:** LiDAR and odometry

## Reinforcement Learning and Navigation

### GridWorld

The GridWorld environment provides a controlled setup for experimenting with reinforcement learning and classical planning algorithms.

Implemented approaches include:

- Q-Learning
- Deep Q-Network (DQN)
- Double DQN
- Behavioral Cloning
- Proximal Policy Optimization (PPO)
- Curriculum-based PPO
- A* Path Planning

The environment supports configurable obstacle layouts, goal-directed navigation, and performance evaluation.

### Jackal Robot Navigation

The simulated Jackal platform extends the project into a continuous robot navigation setting.

Key components include:

- 36 LiDAR sectors for environmental perception.
- Robot state information, including goal distance, goal heading, and velocity.
- A 40-dimensional observation interface.
- PPO-based continuous action prediction.
- Linear and angular velocity control.
- Collision detection and episode termination.
- Safety mechanisms for robot motion.

The project also includes separate navigation implementations for different Gazebo environments.

## Running the Simulations

### 1. Environment Setup

Tested environment:

- Ubuntu 22.04
- Python 3.10
- ROS 2 Humble
- Gazebo Fortress

Create and activate the Python environment:

```bash
cd ~/rl_robot_navigation

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements/base.txt
pip install -r requirements/training.txt
pip install -r requirements/dqn.txt
pip install -r requirements/ppo_transfer.txt
```

For ROS 2 workspace setup and build instructions, refer to the package structure and existing ROS 2 configuration files in `jackal_rl_ws/`.

### 2. Common Terminal Setup

Run this in each terminal used for the simulations:

```bash
cd /home/sanjana/rl_robot_navigation

source /opt/ros/humble/setup.bash
source /home/sanjana/rl_robot_navigation/jackal_rl_ws/install/setup.bash

export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
```

### 3. Industrial Warehouse Navigation

This demonstration uses the industrial warehouse environment and a PPO checkpoint to navigate a predefined route containing 14 checkpoints.

**Terminal 1 – Launch Gazebo:**

```bash
export IGN_PARTITION=jackal_rl_external
export GZ_PARTITION=jackal_rl_external
export IGN_IP=127.0.0.1
export GZ_IP=127.0.0.1

ros2 launch jackal_rl_bringup jackal_external_warehouse.launch.py mode:=simulation_only
```

**Terminal 2 – Run the PPO navigation policy:**

```bash
export IGN_PARTITION=jackal_rl_external
export GZ_PARTITION=jackal_rl_external
export IGN_IP=127.0.0.1
export GZ_IP=127.0.0.1

export PYTHONPATH="/home/sanjana/rl_robot_navigation:/home/sanjana/rl_robot_navigation/jackal_rl_ws/src/jackal_rl_navigation${PYTHONPATH:+:$PYTHONPATH}"

export WAREHOUSE_RL_LIVE_AUTHORIZATION=I_AUTHORIZE_JACKAL_MOTION

.venv/bin/python -m evaluation.warehouse_autonomous \
  --config configs/warehouse_ppo.yaml \
  --checkpoint artifacts/warehouse_ppo/runs/run_20261103T054820Z_bf036311 \
  --authorize-live
```

### 4. Small Warehouse Navigation

**Terminal 1 – Launch Gazebo:**

```bash
export IGN_PARTITION=jackal_rl
export GZ_PARTITION=jackal_rl
export IGN_IP=127.0.0.1
export GZ_IP=127.0.0.1

ros2 launch jackal_rl_bringup jackal_demo_sim.launch.py mode:=simulation_only
```

**Terminal 2 – Run navigation:**

```bash
.venv/bin/python -m evaluation.small_warehouse \
  --config configs/warehouse_ppo.yaml \
  --checkpoint artifacts/warehouse_ppo/runs/run_20261103T054820Z_bf036311 \
  --authorize-live
```

### 5. Obstacle Course Navigation

**Gazebo world:**

`jackal_rl_ws/worlds/obstacle_warehouse/jackal_obstacle_warehouse.sdf`

Launch Gazebo using the common demo simulation command, then run:

```bash
.venv/bin/python -m evaluation.obstacle_warehouse \
  --config configs/warehouse_ppo.yaml \
  --checkpoint artifacts/warehouse_ppo/runs/run_20261103T054820Z_bf036311 \
  --authorize-live
```

### 6. Large Aisle Hall Navigation

**Gazebo world:**

`jackal_rl_ws/worlds/aisle_warehouse/jackal_aisle_warehouse.sdf`

Launch Gazebo using the common demo simulation command, then run:

```bash
.venv/bin/python -m evaluation.aisle_warehouse \
  --config configs/warehouse_ppo.yaml \
  --checkpoint artifacts/warehouse_ppo/runs/run_20261103T054820Z_bf036311 \
  --authorize-live
```

### 7. Café Environment Navigation

**Gazebo world:**

`jackal_rl_ws/worlds/house/jackal_house.sdf`

Launch Gazebo using the common demo simulation command, then run:

```bash
export PYTHONPATH="/home/sanjana/rl_robot_navigation:/home/sanjana/rl_robot_navigation/jackal_rl_ws/src/jackal_rl_navigation${PYTHONPATH:+:$PYTHONPATH}"

.venv/bin/python -m evaluation.warehouse_autonomous \
  --config configs/warehouse_ppo.yaml \
  --checkpoint artifacts/warehouse_ppo/runs/run_20261103T054820Z_bf036311 \
  --world house \
  --authorize-live
```

### 8. Ground Environment Navigation

**Gazebo world:**

`jackal_rl_ws/worlds/ground/jackal_ground.sdf`

Launch Gazebo using the common demo simulation command, then run:

```bash
.venv/bin/python -m evaluation.warehouse_autonomous \
  --config configs/warehouse_ppo.yaml \
  --checkpoint artifacts/warehouse_ppo/runs/run_20261103T054820Z_bf036311 \
  --world ground \
  --authorize-live
```

### 9. Open Environment Navigation

**Gazebo world:**

`jackal_rl_ws/worlds/demo/jackal_demo_open.sdf`

Launch Gazebo using the common demo simulation command, then run:

```bash
.venv/bin/python -m evaluation.warehouse_autonomous \
  --config configs/warehouse_ppo.yaml \
  --checkpoint artifacts/warehouse_ppo/runs/run_20261103T054820Z_bf036311 \
  --world open \
  --authorize-live
```

**Note:** The café, ground, and open environment demonstrations use LiDAR-based roaming behavior. They are not the same predefined checkpoint-route task as the industrial warehouse demonstration. These runs can be stopped with `Ctrl+C`.

### 10. Nav2 Café Navigation

A separate point-to-point navigation demonstration uses Nav2 and a map generated using SLAM Toolbox.

This demonstration is independent of the PPO navigation policy.

The goal used in the recorded demonstration is:

```text
x = 0.0
y = -7.0
```

Stop the PPO navigation process before starting Nav2. Only one velocity-command publisher should control the robot at a time.

### Optional: RViz Visualization

After starting the simulator, RViz can be launched using:

```bash
rviz2 -d /home/sanjana/rl_robot_navigation/jackal_rl_ws/src/jackal_rl_navigation/config/jackal_demo.rviz --ros-args -p use_sim_time:=true
```

Set the Fixed Frame to `map` if required by the visualization configuration.

## Running GridWorld

Activate the Python environment:

```bash
cd ~/rl_robot_navigation
source .venv/bin/activate
```

Start interactive manual navigation:

```bash
python -m scripts.manual_navigation --seed 42
```

The GridWorld environment can also be used to explore the implemented reinforcement learning algorithms and A* path planning.

## Repository Structure

Generated folders such as `.venv/`, `build/`, `install/`, and `log/` stay on this machine and are listed in `.gitignore`.

```text
rl_robot_navigation/
│
├── agents/
│   └── Reinforcement learning agents
│
├── environments/
│   └── GridWorld environment
│
├── planners/
│   └── Classical path-planning algorithms
│
├── training/
│   └── Model training implementations
│
├── evaluation/
│   └── GridWorld and Jackal evaluation scripts
│
├── curriculum/
│   └── Curriculum learning
│
├── imitation/
│   └── Behavior-cloning data and transfer
│
├── models/
│   └── Network definitions
│
├── navigation/
│   └── Jackal navigation helpers
│
├── ros_integration/
│   └── ROS interface helpers
│
├── scripts/
│   └── Manual GridWorld navigation
│
├── visualization/
│   └── Training and benchmark plots
│
├── checkpoint_tabs/
│   └── Checkpoint route panel
│
├── configs/
│   └── Environment and algorithm configurations
│
├── jackal_rl_ws/
│   ├── src/
│   │   ├── jackal_rl_navigation/
│   │   ├── jackal_rl_bringup/
│   │   ├── jackal_actor_sim/
│   │   └── clearpath packages
│   │
│   └── worlds/
│       ├── external_warehouse/
│       ├── small_warehouse/
│       ├── obstacle_warehouse/
│       ├── aisle_warehouse/
│       ├── house/
│       ├── ground/
│       └── demo/
│
├── artifacts/
│   ├── benchmarks/
│   ├── behavior_cloning/
│   ├── curriculum/
│   ├── evaluation_reports/
│   ├── experiments/
│   ├── expert_dataset/
│   ├── imitation/
│   ├── ppo/
│   └── warehouse_ppo/
│
├── tests/
│   └── Automated tests
│
├── images/
│   └── Project screenshots
│
├── videos/
│   └── Navigation demonstration videos
│
├── requirements/
│   ├── base.txt
│   ├── training.txt
│   ├── dqn.txt
│   └── ppo_transfer.txt
│
├── README.md
└── .gitignore
```

## Testing

The repository includes automated tests for GridWorld environments, path planning, reward shaping, and offline Jackal navigation components.

Run selected GridWorld tests:

```bash
.venv/bin/python -m pytest -q -o testpaths= \
  tests/test_gridworld.py \
  tests/test_astar.py \
  tests/test_bfs_shaping.py \
  tests/test_qlearning.py
```

These tests run independently of the Gazebo simulator.

## Experiment Records

Selected model checkpoints, evaluation reports, and experiment records are retained in the `artifacts/` directory.

These files provide supporting evidence of the development and evaluation process. The simulation demonstrations above show the corresponding navigation interfaces and environments.

---

**Project:** Reinforcement Learning-Based Robot Navigation  
**Platform:** ROS 2 Humble | Gazebo Fortress | Clearpath Jackal J100  
**Primary Algorithm:** Proximal Policy Optimization (PPO)
