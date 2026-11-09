from setuptools import find_packages, setup
from glob import glob
import os

package_name = "jackal_rl_navigation"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    package_data={package_name: ["gazebo_transport_client.cpp"]},
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml") + glob("config/*.rviz"),
        ),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="sanjana",
    maintainer_email="",
    description="ROS 2 observation and safe velocity interface for Jackal RL navigation",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "robot_interface = jackal_rl_navigation.robot_interface:main",
        ],
    },
)
