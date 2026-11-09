"""Gazebo Fortress launch for a selected demonstration world.

Executing this file starts Gazebo, spawns the Jackal, and bridges sensors.
Phase 7 live Gate 2 is closed. Do not run it without a separate authorization.
Importing the module does not start a process. World generation and xacro
run only inside the launch setup callback.
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _gazebo_process_pids(role):
    """Return ign gazebo PIDs already attached to partition jackal_rl.

    role is "server" (``ign gazebo -s``) or "gui" (``ign gazebo -g``).
    This only reads /proc. It does not signal any process.
    """

    if role not in ("server", "gui"):
        raise ValueError(role)
    flag = "-s" if role == "server" else "-g"
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw_cmd = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        # Some ign wrappers store the whole command in one cmdline field.
        args = raw_cmd.replace(b"\x00", b" ").decode(errors="replace").split()
        if len(args) < 3 or Path(args[0]).name != "ign" or args[1] != "gazebo":
            continue
        if flag not in args[2:]:
            continue
        try:
            raw_env = (entry / "environ").read_bytes()
        except OSError:
            continue
        partition = None
        for item in raw_env.split(b"\x00"):
            if item.startswith(b"IGN_PARTITION="):
                partition = item.split(b"=", 1)[1].decode(errors="replace")
            elif item.startswith(b"GZ_PARTITION=") and partition is None:
                partition = item.split(b"=", 1)[1].decode(errors="replace")
        if partition == "jackal_rl":
            found.append(int(entry.name))
    return found


def _sdf_world_name(path: Path) -> str:
    """Read the world name Ignition will load from this file."""

    world = ET.parse(path).getroot().find("world")
    name = None if world is None else world.get("name")
    if not name:
        raise RuntimeError(f"World file has no world name: {path}")
    return name


def _launch_setup(context, *args, **kwargs):
    root = Path.home() / "rl_robot_navigation"
    ws = root / "jackal_rl_ws"
    world = Path(LaunchConfiguration("world_file").perform(context))
    requested_name = LaunchConfiguration("world_name").perform(context)
    world_name = _sdf_world_name(world)
    if requested_name != world_name:
        print(
            f"Spawning into world {world_name!r} from {world.name}. "
            f"The launch argument world_name:={requested_name} is not used."
        )
    mode = LaunchConfiguration("mode").perform(context).strip()
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

    os.environ["IGN_PARTITION"] = "jackal_rl"
    os.environ["GZ_PARTITION"] = "jackal_rl"
    os.environ["IGN_IP"] = "127.0.0.1"
    os.environ["GZ_IP"] = "127.0.0.1"

    robot_name = "j100_0000"
    xacro_file = ws / "config" / "jackal" / "robot.urdf.xacro"
    controllers = ws / "config" / "jackal" / "platform" / "config" / "control.yaml"

    if not all(path.is_file() for path in (world, xacro_file, controllers)):
        raise RuntimeError("Missing Jackal world, xacro, or controller configuration")

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

    urdf_path = Path("/tmp/jackal_visible.urdf")
    urdf_path.write_text(robot_description)

    resource_path = os.pathsep.join(
        [
            "/opt/ros/humble/share",
            str(ws / "install"),
            str(ws / "src"),
            str(ws / "worlds" / "models"),
            str(ws / "worlds" / "external_warehouse" / "models"),
            os.environ.get("IGN_GAZEBO_RESOURCE_PATH", ""),
        ]
    )
    partition = {
        "IGN_PARTITION": "jackal_rl",
        "GZ_PARTITION": "jackal_rl",
        "IGN_IP": "127.0.0.1",
        "GZ_IP": "127.0.0.1",
    }

    gazebo_env = {
        **partition,
        "IGN_GAZEBO_RESOURCE_PATH": resource_path,
        "IGN_GAZEBO_SYSTEM_PLUGIN_PATH":
            "/opt/ros/humble/lib:"
            + os.environ.get("IGN_GAZEBO_SYSTEM_PLUGIN_PATH", ""),
    }
    # One server. -s keeps this process headless. -g is only a GUI client
    # on the same partition; it is started once, and skipped when a GUI
    # is already attached. Existing Gazebo processes are not signalled.
    existing_server = _gazebo_process_pids("server")
    if existing_server:
        pids = ", ".join(str(pid) for pid in existing_server)
        raise RuntimeError(
            "A Gazebo server is already running on partition jackal_rl "
            f"(pid {pids}). This launch does not start a second server "
            "and does not stop existing Gazebo processes. Stop the previous "
            "jackal_demo_sim launch by sending SIGINT to its ros2 launch "
            "process. Do not use killall."
        )
    existing_gui = _gazebo_process_pids("gui")
    gazebo = ExecuteProcess(
        cmd=["ign", "gazebo", "-s", "-r", str(world)],
        output="screen",
        additional_env=gazebo_env,
    )
    gazebo_gui = ExecuteProcess(
        cmd=["ign", "gazebo", "-g"],
        output="screen",
        additional_env=gazebo_env,
    )
    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace=robot_name,
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True,
            "frame_prefix": "",
        }],
        # The node namespace would otherwise publish the body on
        # /j100_0000/tf. RViz only listens to /tf, which is why every
        # link reports "No transform".
        remappings=[
            ("joint_states", "platform/joint_states"),
            ("tf", "/tf"),
            ("tf_static", "/tf_static"),
        ],
        output="screen",
        additional_env=partition,
    )
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
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", f"/{robot_name}/controller_manager",
            "--controller-manager-timeout", "90",
        ],
        output="screen",
        additional_env=partition,
    )
    velocity_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "platform_velocity_controller",
            "--controller-manager", f"/{robot_name}/controller_manager",
            "--controller-manager-timeout", "90",
        ],
        output="screen",
        additional_env=partition,
    )
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock"],
        output="screen",
        additional_env=partition,
    )
    lidar_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            f"/{robot_name}/sensors/lidar3d_0/scan"
            "@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
            f"/{robot_name}/sensors/camera_0/image"
            "@sensor_msgs/msg/Image[ignition.msgs.Image",
        ],
        output="screen",
        additional_env=partition,
    )
    from ament_index_python.packages import get_package_share_directory

    slam = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            str(Path(get_package_share_directory("slam_toolbox")) / "config" / "mapper_params_online_async.yaml"),
            {
                "use_sim_time": True,
                "odom_frame": "odom",
                "map_frame": "map",
                "base_frame": "base_link",
                "scan_topic": f"/{robot_name}/sensors/lidar3d_0/scan",
                "max_laser_range": 12.0,
                "min_laser_range": 0.12,
                "map_update_interval": 1.0,
                "minimum_travel_distance": 0.2,
                "minimum_travel_heading": 0.2,
                "transform_timeout": 1.0,
            },
        ],
        additional_env=partition,
    )
    odom_tf = ExecuteProcess(
        cmd=[
            "python3",
            str(ws / "src" / "jackal_rl_bringup" / "scripts" / "publish_odom_tf.py"),
        ],
        output="screen",
        additional_env=partition,
    )
    contact_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/jackal_rl/obstacle_contacts"
            "@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts"
        ],
        output="screen",
        additional_env=partition,
    )
    delayed_spawn = TimerAction(period=8.0, actions=[spawn_robot])
    delayed_gui = TimerAction(period=2.0, actions=[gazebo_gui])
    start_controllers = RegisterEventHandler(
        OnProcessExit(target_action=spawn_robot, on_exit=[joint_broadcaster])
    )
    start_velocity = RegisterEventHandler(
        OnProcessExit(target_action=joint_broadcaster, on_exit=[velocity_controller])
    )
    actions = [
        state_publisher,
        gazebo,
        clock_bridge,
        lidar_bridge,
        contact_bridge,
        odom_tf,
        slam,
        start_controllers,
        start_velocity,
        delayed_spawn,
    ]
    if existing_gui:
        print(
            "Gazebo GUI already attached to partition jackal_rl "
            f"(pid {', '.join(str(pid) for pid in existing_gui)}). "
            "This launch will not start another GUI."
        )
    else:
        actions.insert(2, delayed_gui)
    if mode == "simulation_only":
        print(
            "Launch mode simulation_only: Gazebo, one robot, sensors, and "
            "controllers. No navigation node is started and this launch does "
            "not publish velocity."
        )
    else:
        navigation_launch = (
            ws / "src" / "jackal_rl_navigation" / "launch" / "demo_navigation.launch.py"
        )
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(navigation_launch)),
            launch_arguments=[
                ("namespace", "j100_0000"),
                ("goal_x", "4.0"),
                ("goal_y", "0.0"),
                ("goal_frame", "odom"),
                ("goal_controller", "classical_baseline"),
                ("enable_goal_loop", "true"),
            ],
        ))
        print(
            "Launch mode classical_navigation: the navigation interface is the "
            "only added velocity publisher. Do not start a second controller."
        )
    return actions


def generate_launch_description():
    default_world = (
        Path.home()
        / "rl_robot_navigation"
        / "jackal_rl_ws"
        / "worlds"
        / "demo"
        / "jackal_demo_open.sdf"
    )
    return LaunchDescription([
        DeclareLaunchArgument("world_file", default_value=str(default_world)),
        DeclareLaunchArgument("world_name", default_value="jackal_demo_open"),
        DeclareLaunchArgument("mode", default_value="simulation_only"),
        SetEnvironmentVariable("IGN_PARTITION", "jackal_rl"),
        SetEnvironmentVariable("GZ_PARTITION", "jackal_rl"),
        SetEnvironmentVariable("IGN_IP", "127.0.0.1"),
        SetEnvironmentVariable("GZ_IP", "127.0.0.1"),
        OpaqueFunction(function=_launch_setup),
    ])
