"""Convert between Gazebo world coordinates and robot odometry.

The A* obstacle map is stored in the world frame. After a teleport,
odometry usually stays continuous, so planning must translate the
world route back into the odometry frame used by the controller.
"""

import math


def _rotate_to_odom(dx, dy, world_yaw, odom_yaw):
    cosine = math.cos(world_yaw)
    sine = math.sin(world_yaw)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    cosine = math.cos(odom_yaw)
    sine = math.sin(odom_yaw)
    return (
        cosine * local_x - sine * local_y,
        sine * local_x + cosine * local_y,
    )


def _rotate_to_world(dx, dy, world_yaw, odom_yaw):
    cosine = math.cos(odom_yaw)
    sine = math.sin(odom_yaw)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    cosine = math.cos(world_yaw)
    sine = math.sin(world_yaw)
    return (
        cosine * local_x - sine * local_y,
        sine * local_x + cosine * local_y,
    )


class WorldAlignedPlanner:
    """Plan on the world grid and return odometry-frame waypoints."""

    def __init__(self, planner, world_pose, odom_pose):
        self.planner = planner
        self.world_pose = tuple(float(v) for v in world_pose)
        self.odom_pose = tuple(float(v) for v in odom_pose)

        if len(self.world_pose) != 3 or len(self.odom_pose) != 3:
            raise ValueError("Poses must be (x, y, yaw)")

        if not all(math.isfinite(v) for v in (*self.world_pose, *self.odom_pose)):
            raise ValueError("Frame poses must be finite")

    def to_world(self, point):
        dx = float(point[0]) - self.odom_pose[0]
        dy = float(point[1]) - self.odom_pose[1]
        offset_x, offset_y = _rotate_to_world(
            dx, dy, self.world_pose[2], self.odom_pose[2]
        )
        return (
            self.world_pose[0] + offset_x,
            self.world_pose[1] + offset_y,
        )

    def to_odom(self, point):
        dx = float(point[0]) - self.world_pose[0]
        dy = float(point[1]) - self.world_pose[1]
        offset_x, offset_y = _rotate_to_odom(
            dx, dy, self.world_pose[2], self.odom_pose[2]
        )
        return (
            self.odom_pose[0] + offset_x,
            self.odom_pose[1] + offset_y,
        )

    def plan(self, start, goal):
        path = self.planner.plan(self.to_world(start), self.to_world(goal))

        if not path:
            return None

        return [self.to_odom(point) for point in path]

    def validate_path(self, waypoints):
        return self.planner.validate_path([
            self.to_world(point) for point in waypoints
        ])


def alignment_from_poses(world_pose, odom_pose):
    """Store the poses used to interpret an episode trajectory."""

    world = (float(world_pose[0]), float(world_pose[1]), float(world_pose[2]))
    odom = (float(odom_pose[0]), float(odom_pose[1]), float(odom_pose[2]))
    return {
        "world_pose": world,
        "odom_pose": odom,
    }


def points_to_world(points, alignment):
    """Convert odometry xy points with the alignment captured at reset."""

    if not alignment or not points:
        return [list(map(float, point)) for point in points] if points else []

    frame = WorldAlignedPlanner(
        planner=None,
        world_pose=alignment["world_pose"],
        odom_pose=alignment["odom_pose"],
    )
    converted = []

    for point in points:
        world = frame.to_world(point)
        if len(point) >= 3:
            converted.append([world[0], world[1], float(point[2])])
        else:
            converted.append([world[0], world[1]])

    return converted


def wait_for_fresh_sensors(env, robot, timeout=2.0):
    """Spin until LiDAR and odometry are inside the safety window."""

    import time

    if timeout <= 0:
        raise ValueError("timeout must be positive")

    deadline = time.monotonic() + timeout
    limit = robot.safety_monitor.sensor_timeout

    while True:
        env.executor.spin_once(timeout_sec=0.05)
        ages = robot._sensor_ages()

        if (
            ages is not None
            and all(math.isfinite(float(age)) for age in ages)
            and all(float(age) <= limit for age in ages)
        ):
            return True

        if time.monotonic() >= deadline:
            return False
