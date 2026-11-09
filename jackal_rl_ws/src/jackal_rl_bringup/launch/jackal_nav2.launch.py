"""Nav2 map, global costmap, and navigation panel for the running cafe.

The smoothed command is published on /j100_0000/nav2/cmd_vel. It is not
forwarded to the Jackal, so the learned policy can keep driving.
"""

from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = str(Path(__file__).resolve().parents[1] / "config" / "nav2_params.yaml")
    common = dict(parameters=[params], output="screen")
    cmd_remap = [("cmd_vel", "cmd_vel_nav")]
    return LaunchDescription([
        Node(
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            remappings=cmd_remap,
            **common,
        ),
        Node(
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            **common,
        ),
        Node(
            package="nav2_behaviors",
            executable="behavior_server",
            name="behavior_server",
            remappings=cmd_remap,
            **common,
        ),
        Node(
            package="nav2_bt_navigator",
            executable="bt_navigator",
            name="bt_navigator",
            **common,
        ),
        Node(
            package="nav2_waypoint_follower",
            executable="waypoint_follower",
            name="waypoint_follower",
            **common,
        ),
        Node(
            package="nav2_velocity_smoother",
            executable="velocity_smoother",
            name="velocity_smoother",
            remappings=[
                ("cmd_vel", "cmd_vel_nav"),
                ("cmd_vel_smoothed", "/j100_0000/nav2/cmd_vel"),
            ],
            **common,
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            **common,
        ),
    ])
