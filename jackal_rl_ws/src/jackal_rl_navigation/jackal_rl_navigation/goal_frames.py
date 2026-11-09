"""Coordinate conversion between Gazebo world and ROS odometry."""

import math


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def world_goal_to_odom(
    world_goal,
    world_robot_pose,
    odom_robot_pose,
):
    """Convert a world XY goal into the robot's odometry frame.

    Poses are (x, y, yaw).
    Goal is (x, y).
    """

    gx, gy = map(float, world_goal)
    wx, wy, wyaw = map(float, world_robot_pose)
    ox, oy, oyaw = map(float, odom_robot_pose)

    values = (gx, gy, wx, wy, wyaw, ox, oy, oyaw)

    if not all(math.isfinite(v) for v in values):
        raise ValueError("Coordinates must be finite")

    # Goal displacement expressed in the world frame.
    dx = gx - wx
    dy = gy - wy

    # Rotate world displacement into the robot's local frame.
    c = math.cos(wyaw)
    s = math.sin(wyaw)

    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy

    # Rotate local displacement into the odometry frame.
    c = math.cos(oyaw)
    s = math.sin(oyaw)

    odom_dx = c * local_x - s * local_y
    odom_dy = s * local_x + c * local_y

    return ox + odom_dx, oy + odom_dy


def odom_point_to_world(odom_point, world_robot_pose, odom_robot_pose):
    """Invert ``world_goal_to_odom`` for the same robot pose pair."""

    px, py = map(float, odom_point)
    wx, wy, wyaw = map(float, world_robot_pose)
    ox, oy, oyaw = map(float, odom_robot_pose)
    values = (px, py, wx, wy, wyaw, ox, oy, oyaw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Coordinates must be finite")

    dx = px - ox
    dy = py - oy
    cosine = math.cos(oyaw)
    sine = math.sin(oyaw)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    cosine = math.cos(wyaw)
    sine = math.sin(wyaw)
    return wx + cosine * local_x - sine * local_y, wy + sine * local_x + cosine * local_y


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def odom_pose_from_message(odom):
    position = odom.pose.pose.position
    orientation = odom.pose.pose.orientation

    return (
        float(position.x),
        float(position.y),
        yaw_from_quaternion(orientation),
    )
