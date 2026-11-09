"""Launch Jackal RL navigation interface."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = Path(
        get_package_share_directory("jackal_rl_navigation")
    )

    config = package_dir / "config" / "navigation.yaml"

    return LaunchDescription([
        Node(
            package="jackal_rl_navigation",
            executable="robot_interface",
            name="jackal_rl_interface",
            output="screen",
            parameters=[str(config)],
        )
    ])
