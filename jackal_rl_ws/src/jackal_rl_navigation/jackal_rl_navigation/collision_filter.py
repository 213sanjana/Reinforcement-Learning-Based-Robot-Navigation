"""Conservative contact classification using exact SDF entity scopes.

False means a known wheel/support pair, robot self-contact, or a pair
unrelated to this robot. True also covers abnormal/unknown robot contacts;
the existing collision latch treats these as safety faults.
"""

# From the project's expanded Fortress SDFs, not mesh filenames. Keep the
# nested model scope: accepting arbitrary suffixes could hide obstacle contact.
SUPPORT_COLLISIONS = frozenset({
    "ground_plane::ground_link::collision",
    "aws_robomaker_warehouse_GroundB_01_001::"
    "aws_robomaker_warehouse_GroundB_01::ground_link::collision",
    # Cafe floor in jackal_house.sdf. The robot's wheel sensors publish this
    # contact on every step; it is the walking surface, not an obstacle.
    "cafe::link::main_floor",
})
WHEEL_COLLISIONS = frozenset(
    f"{position}_wheel_link::{position}_wheel_link_collision"
    for position in ("front_left", "front_right", "rear_left", "rear_right")
)

# Fixed-joint lumping names verified by `ign sdf -p` on the generated URDF.
ROBOT_COLLISIONS = WHEEL_COLLISIONS | frozenset({
    "base_link::base_link_fixed_joint_lump__chassis_link_collision",
    "base_link::base_link_fixed_joint_lump__camera_0_link_collision_1",
    "base_link::base_link_fixed_joint_lump__lidar3d_0_link_collision_2",
})


def _robot_entity(name, robot_name):
    for prefix in (robot_name + "::", robot_name + "/robot::"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return None


def is_robot_obstacle_collision(
    collision_a: str,
    collision_b: str,
    robot_name: str = "j100_0000",
) -> bool:
    """Return whether the pair must trigger the collision/safety latch.

    Only exact known wheel-to-support pairs are benign ground contact.
    Missing identifiers fail closed; an unknown robot link never earns a
    support exemption. Live scoping still needs stationary verification.
    """
    if not all(isinstance(name, str) and name.strip()
               for name in (collision_a, collision_b)):
        return True
    robot_a = _robot_entity(collision_a, robot_name)
    robot_b = _robot_entity(collision_b, robot_name)
    if robot_a is not None and robot_b is not None:
        return not (robot_a in ROBOT_COLLISIONS and robot_b in ROBOT_COLLISIONS)
    if robot_a is None and robot_b is None:
        return False
    robot, other = ((robot_a, collision_b) if robot_a is not None
                    else (robot_b, collision_a))
    return not (robot in WHEEL_COLLISIONS and other in SUPPORT_COLLISIONS)
