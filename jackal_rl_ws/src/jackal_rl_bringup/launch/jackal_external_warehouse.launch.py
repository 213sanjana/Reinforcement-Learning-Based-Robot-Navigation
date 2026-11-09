"""Stationary-by-default launch for the external industrial warehouse.

Importing this file does not start Gazebo. The default mode is
simulation_only. The robot namespace stays j100_0000, so this launch
refuses to start while any other Ignition server is already running.
It does not signal those processes.
"""

import os
import re
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    EmitEvent,
    LogInfo,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PARTITION = "jackal_rl_external"


def _controller_spawner_arguments(controller, robot_name):
    # Discovery readiness is handled by Humble's spawner, independently of
    # the bounded service-response and activation-switch timeouts.
    return [
        controller,
        "--controller-manager", f"/{robot_name}/controller_manager",
        "--controller-manager-timeout", "90",
        "--service-call-timeout", "10",
        "--switch-timeout", "5",
    ]


def _after_success(stage, next_action=None):
    """Do not turn failed spawn/load exits into another startup attempt."""
    def on_exit(event, context):
        if context.is_shutdown:
            return []
        if event.returncode != 0:
            reason = f"External warehouse {stage} failed (exit {event.returncode})"
            return [LogInfo(msg=reason), EmitEvent(event=Shutdown(reason=reason))]
        return [] if next_action is None else [next_action]
    return on_exit


def _on_server_exit(event, context):
    if context.is_shutdown:
        return []
    reason = f"External warehouse server exited (exit {event.returncode})"
    return [LogInfo(msg=reason), EmitEvent(event=Shutdown(reason=reason))]


def _gazebo_server_pids():
    """Return every ign gazebo server PID. This only reads /proc."""

    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw_cmd = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        args = raw_cmd.replace(b"\x00", b" ").decode(errors="replace").split()
        if len(args) < 3 or Path(args[0]).name != "ign" or args[1] != "gazebo":
            continue
        if "-s" not in args[2:]:
            continue
        found.append(int(entry.name))
    return found


def _launch_setup(context, *args, **kwargs):
    root = Path.home() / "rl_robot_navigation"
    ws = root / "jackal_rl_ws"
    world = Path(LaunchConfiguration("world_file").perform(context))
    world_name = LaunchConfiguration("world_name").perform(context)
    mode = LaunchConfiguration("mode").perform(context).strip()
    gui = LaunchConfiguration("gui").perform(context).strip().lower() in {
        "1", "true", "yes",
    }
    if mode not in {
        "simulation_only",
        "classical_navigation",
        "rl_navigation",
        "evaluation",
    }:
        raise RuntimeError(f"Unknown launch mode {mode!r}")
    if mode == "rl_navigation":
        raise RuntimeError(
            "rl_navigation does not load a checkpoint and does not start motion. "
            "GridWorld weights are not a Jackal policy. "
            "The protected Phase 6G.1 zip is not opened."
        )
    if mode == "evaluation":
        raise RuntimeError(
            "evaluation mode does not start a moving episode from this launch. "
            "Use one explicit episode runner and a new artifacts/experiments directory."
        )

    existing = _gazebo_server_pids()
    if existing:
        pids = ", ".join(str(pid) for pid in existing)
        raise RuntimeError(
            "An Ignition Gazebo server is already running "
            f"(pid {pids}). The external warehouse uses the same robot "
            "namespace j100_0000 and the same /clock bridge, so this launch "
            "does not start a second server and does not stop the existing "
            "process. Stop the other ros2 launch with SIGINT first. "
            "Do not use killall."
        )

    robot_name = "j100_0000"
    xacro_file = ws / "config" / "jackal" / "robot.urdf.xacro"
    controllers = ws / "config" / "jackal" / "platform" / "config" / "control.yaml"
    model_root = ws / "worlds" / "external_warehouse" / "models"
    if not all(path.is_file() for path in (world, xacro_file, controllers)):
        raise RuntimeError("Missing external warehouse, xacro, or controller configuration")
    if not model_root.is_dir():
        raise RuntimeError(f"Missing warehouse models: {model_root}")

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
    mesh_root = Path("/opt/ros/humble/share/clearpath_platform_description")

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
    robot_description = robot_description.replace("<min>0.9</min>", "<min>0.1</min>")
    urdf_path = Path("/tmp/jackal_external_warehouse.urdf")
    urdf_path.write_text(robot_description)

    partition = {
        "IGN_PARTITION": PARTITION,
        "GZ_PARTITION": PARTITION,
        "IGN_IP": "127.0.0.1",
        "GZ_IP": "127.0.0.1",
    }
    resource_path = os.pathsep.join(
        [
            str(model_root),
            "/opt/ros/humble/share",
            str(ws / "install"),
            str(ws / "src"),
            os.environ.get("IGN_GAZEBO_RESOURCE_PATH", ""),
        ]
    )
    gazebo_env = {
        **partition,
        "IGN_GAZEBO_RESOURCE_PATH": resource_path,
        "SDF_PATH": str(model_root),
        "IGN_GAZEBO_SYSTEM_PLUGIN_PATH":
            "/opt/ros/humble/lib:"
            + os.environ.get("IGN_GAZEBO_SYSTEM_PLUGIN_PATH", ""),
    }
    gazebo = ExecuteProcess(
        cmd=["ign", "gazebo", "-s", "-r", str(world)],
        output="screen",
        additional_env=gazebo_env,
    )
    actions = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=robot_name,
            parameters=[{"robot_description": robot_description, "use_sim_time": True}],
            output="screen",
            additional_env=partition,
        ),
        gazebo,
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=["/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock"],
            output="screen",
            additional_env=partition,
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                f"/{robot_name}/sensors/lidar3d_0/scan"
                "@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan"
            ],
            output="screen",
            additional_env=partition,
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                "/jackal_rl/obstacle_contacts"
                "@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts"
            ],
            output="screen",
            additional_env=partition,
        ),
    ]
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-world", world_name,
            "-name", robot_name,
            "-file", str(urdf_path),
            "-x", "0", "-y", "0", "-z", "0.3",
        ],
        output="screen",
        additional_env=partition,
    )
    joint_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=_controller_spawner_arguments("joint_state_broadcaster", robot_name),
        output="screen",
        additional_env=partition,
    )
    velocity_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=_controller_spawner_arguments("platform_velocity_controller", robot_name),
        output="screen",
        additional_env=partition,
    )
    actions.extend([
        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn_robot,
                on_exit=_after_success("robot spawn", joint_broadcaster),
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=joint_broadcaster,
                on_exit=_after_success("joint-state broadcaster", velocity_controller),
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=velocity_controller,
                on_exit=_after_success("velocity controller"),
            )
        ),
        RegisterEventHandler(
            OnProcessExit(target_action=gazebo, on_exit=_on_server_exit)
        ),
        TimerAction(period=8.0, actions=[spawn_robot]),
    ])
    if gui:
        actions.append(TimerAction(period=2.0, actions=[
            ExecuteProcess(
                cmd=["ign", "gazebo", "-g"],
                output="screen",
                additional_env=gazebo_env,
            )
        ]))
    if mode == "simulation_only":
        print(
            "Launch mode simulation_only: external warehouse, one Jackal, "
            "sensors, and controllers. No navigation node is started and "
            "this launch does not publish velocity."
        )
    else:
        navigation_launch = (
            ws / "src" / "jackal_rl_navigation" / "launch" / "demo_navigation.launch.py"
        )
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(navigation_launch)),
            launch_arguments=[
                ("namespace", robot_name),
                ("goal_x", "0.0"),
                ("goal_y", "-3.0"),
                ("goal_frame", "odom"),
                ("goal_controller", "classical_baseline"),
                ("enable_goal_loop", "true"),
            ],
        ))
    return actions


def generate_launch_description():
    default_world = (
        Path.home()
        / "rl_robot_navigation"
        / "jackal_rl_ws"
        / "worlds"
        / "external_warehouse"
        / "jackal_industrial_warehouse.sdf"
    )
    return LaunchDescription([
        DeclareLaunchArgument("world_file", default_value=str(default_world)),
        DeclareLaunchArgument("world_name", default_value="industrial-warehouse"),
        DeclareLaunchArgument("mode", default_value="simulation_only"),
        DeclareLaunchArgument("gui", default_value="true"),
        SetEnvironmentVariable("IGN_PARTITION", PARTITION),
        SetEnvironmentVariable("GZ_PARTITION", PARTITION),
        SetEnvironmentVariable("IGN_IP", "127.0.0.1"),
        SetEnvironmentVariable("GZ_IP", "127.0.0.1"),
        OpaqueFunction(function=_launch_setup),
    ])
