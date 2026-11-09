"""Read-only ROS graph preflight for Jackal expert collection."""

import re
import subprocess
from dataclasses import dataclass


TOPICS = {
    "odometry": (
        "/j100_0000/platform/odom",
        "nav_msgs/msg/Odometry",
    ),
    "lidar": (
        "/j100_0000/sensors/lidar3d_0/scan",
        "sensor_msgs/msg/LaserScan",
    ),
    "collision": (
        "/jackal_rl/obstacle_contacts",
        "ros_gz_interfaces/msg/Contacts",
    ),
    "velocity": (
        "/j100_0000/platform/cmd_vel_unstamped",
        "geometry_msgs/msg/Twist",
    ),
}


@dataclass(frozen=True)
class PreflightResult:
    passed: bool
    checks: dict


def run_command(command, timeout=12):
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(
            command, 1, "", str(exc)
        )


def inspect_topic(topic):
    result = run_command(["ros2", "topic", "info", topic])

    if result.returncode != 0:
        return None

    output = result.stdout

    type_match = re.search(r"Type:\s*(\S+)", output)
    pub_match = re.search(
        r"Publisher count:\s*(\d+)", output
    )
    sub_match = re.search(
        r"Subscription count:\s*(\d+)", output
    )

    if not all((type_match, pub_match, sub_match)):
        return None

    return {
        "type": type_match.group(1),
        "publishers": int(pub_match.group(1)),
        "subscribers": int(sub_match.group(1)),
    }


def inspect_controllers():
    result = run_command([
        "ros2",
        "control",
        "list_controllers",
        "-c",
        "/j100_0000/controller_manager",
    ])

    if result.returncode != 0:
        return False

    active = set()

    ansi_pattern = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

    for line in result.stdout.splitlines():
        clean_line = ansi_pattern.sub("", line)
        parts = clean_line.split()

        if len(parts) >= 3 and parts[-1] == "active":
            active.add(parts[0])

    required = {
        "joint_state_broadcaster",
        "platform_velocity_controller",
    }

    return required.issubset(active)


def run_preflight():
    checks = {}

    for name, (topic, expected_type) in TOPICS.items():
        info = inspect_topic(topic)

        checks[f"{name}_topic"] = (
            info is not None
            and info["type"] == expected_type
        )

        if name in ("odometry", "lidar", "collision"):
            checks[f"{name}_publisher"] = (
                info is not None
                and info["publishers"] >= 1
            )

        if name == "velocity":
            checks["velocity_subscriber"] = (
                info is not None
                and info["subscribers"] >= 1
            )

            # No competing velocity publishers before collection.
            checks["velocity_no_competing_publisher"] = (
                info is not None
                and info["publishers"] == 0
            )

    checks["controllers_active"] = inspect_controllers()

    return PreflightResult(
        passed=all(checks.values()),
        checks=checks,
    )


def main():
    result = run_preflight()

    for name, passed in result.checks.items():
        print(
            f"{name}: {'PASS' if passed else 'FAIL'}"
        )

    print(
        "PREFLIGHT:",
        "PASS" if result.passed else "FAIL",
    )

    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
