import os
import re
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def generate_launch_description():
    root = Path.home() / "rl_robot_navigation"
    ws = root / "jackal_rl_ws"

    os.environ["IGN_PARTITION"] = "jackal_rl"
    os.environ["GZ_PARTITION"] = "jackal_rl"
    os.environ["IGN_IP"] = "127.0.0.1"
    os.environ["GZ_IP"] = "127.0.0.1"

    robot_name = "j100_0000"
    world_name = "jackal_obstacle_test"

    world = ws / "worlds" / "jackal_obstacle_test_collision.sdf"
    xacro_file = ws / "config" / "jackal" / "robot.urdf.xacro"
    controllers = ws / "config" / "jackal" / "platform" / "config" / "control.yaml"

    if not all(p.is_file() for p in (world, xacro_file, controllers)):
        raise RuntimeError("Missing Jackal world, xacro, or controller configuration")

    # Generate the robot description before launching Gazebo.
    import subprocess

    result = subprocess.run(
        [
            "xacro",
            str(xacro_file),
            "is_sim:=true",
            f"namespace:={robot_name}",
            f"gazebo_controllers:={controllers}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    mesh_root = Path(
        "/opt/ros/humble/share/clearpath_platform_description"
    )

    def resolve_mesh(match):
        path = mesh_root / match.group(1)
        if not path.is_file():
            raise RuntimeError(f"Missing Jackal mesh: {path}")
        return path.as_uri()

    robot_description = re.sub(
        r"(?:package://|model://)clearpath_platform_description/([^\s\"<>]+)",
        resolve_mesh,
        result.stdout,
    )

    # Include short-range LiDAR returns for obstacle avoidance.
    robot_description = robot_description.replace(
        "<min>0.9</min>",
        "<min>0.1</min>",
    )

    urdf_path = Path("/tmp/jackal_visible.urdf")
    urdf_path.write_text(robot_description)

    resource_path = os.pathsep.join(
        [
            "/opt/ros/humble/share",
            str(ws / "install"),
            str(ws / "src"),
            os.environ.get("IGN_GAZEBO_RESOURCE_PATH", ""),
        ]
    )

    gazebo = ExecuteProcess(
        cmd=["ign", "gazebo", "-s", "-r", str(world)],
        output="screen",
        additional_env={
            "IGN_PARTITION": "jackal_rl",
            "GZ_PARTITION": "jackal_rl",
            "IGN_IP": "127.0.0.1",
            "GZ_IP": "127.0.0.1",
            "IGN_GAZEBO_RESOURCE_PATH": resource_path,
            "IGN_GAZEBO_SYSTEM_PLUGIN_PATH":
                "/opt/ros/humble/lib:" +
                os.environ.get("IGN_GAZEBO_SYSTEM_PLUGIN_PATH", ""),
        },
    )

    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace=robot_name,
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True,
            }
        ],
        output="screen",
        additional_env={
            "IGN_PARTITION": "jackal_rl",
            "GZ_PARTITION": "jackal_rl",
        },
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-world", world_name,
            "-name", robot_name,
            "-file", str(urdf_path),
            "-x", "0",
            "-y", "0",
            "-z", "0.3",
        ],
        output="screen",
        additional_env={
            "IGN_PARTITION": "jackal_rl",
            "GZ_PARTITION": "jackal_rl",
        },
    )

    joint_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            f"/{robot_name}/controller_manager",
            "--controller-manager-timeout", "90",
        ],
        output="screen",
        additional_env={
            "IGN_PARTITION": "jackal_rl",
            "GZ_PARTITION": "jackal_rl",
        },
    )

    velocity_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "platform_velocity_controller",
            "--controller-manager",
            f"/{robot_name}/controller_manager",
            "--controller-manager-timeout", "90",
        ],
        output="screen",
        additional_env={
            "IGN_PARTITION": "jackal_rl",
            "GZ_PARTITION": "jackal_rl",
        },
    )

    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock"
        ],
        output="screen",
        additional_env={
            "IGN_PARTITION": "jackal_rl",
            "GZ_PARTITION": "jackal_rl",
        },
    )

    lidar_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            f"/{robot_name}/sensors/lidar3d_0/scan"
            "@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan"
        ],
        output="screen",
        additional_env={
            "IGN_PARTITION": "jackal_rl",
            "GZ_PARTITION": "jackal_rl",
        },
    )

    contact_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/jackal_rl/obstacle_contacts"
            "@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts"
        ],
        output="screen",
        additional_env={
            "IGN_PARTITION": "jackal_rl",
            "GZ_PARTITION": "jackal_rl",
        },
    )

    # Wait for Gazebo to initialize before spawning.
    delayed_spawn = TimerAction(
        period=8.0,
        actions=[spawn_robot],
    )

    # Start controllers after the spawn process finishes.
    start_controllers = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[joint_broadcaster],
        )
    )

    start_velocity = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_broadcaster,
            on_exit=[velocity_controller],
        )
    )

    return LaunchDescription([
        SetEnvironmentVariable("IGN_PARTITION", "jackal_rl"),
        SetEnvironmentVariable("GZ_PARTITION", "jackal_rl"),
        SetEnvironmentVariable("IGN_IP", "127.0.0.1"),
        SetEnvironmentVariable("GZ_IP", "127.0.0.1"),
        state_publisher,
        gazebo,
        clock_bridge,
        lidar_bridge,
        contact_bridge,
        start_controllers,
        start_velocity,
        delayed_spawn,
    ])
