"""Navigation interface launch definition.

Running this file starts a ROS 2 node and is not authorized while Phase 7
live Gate 2 is closed. The file does not start Gazebo. The node can publish
a zero Twist from its own stop path once it is actually executed.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("namespace", default_value="j100_0000"),
        DeclareLaunchArgument("goal_x", default_value="4.0"),
        DeclareLaunchArgument("goal_y", default_value="0.0"),
        DeclareLaunchArgument("goal_frame", default_value="odom"),
        DeclareLaunchArgument("goal_controller", default_value="classical_baseline"),
        DeclareLaunchArgument("enable_goal_loop", default_value="false"),
        Node(
            package="jackal_rl_navigation",
            executable="robot_interface",
            name="jackal_rl_interface",
            output="screen",
            parameters=[{
                "namespace": LaunchConfiguration("namespace"),
                "lidar_bins": 36,
                "goal_x": LaunchConfiguration("goal_x"),
                "goal_y": LaunchConfiguration("goal_y"),
                "goal_frame": LaunchConfiguration("goal_frame"),
                "goal_controller": LaunchConfiguration("goal_controller"),
                "enable_goal_loop": LaunchConfiguration("enable_goal_loop"),
                "sensor_timeout": 0.5,
                "use_sim_time": True,
            }],
        ),
    ])
