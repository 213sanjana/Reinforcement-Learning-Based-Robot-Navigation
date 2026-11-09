"""ROS 2 contact monitor for Jackal navigation."""

from rclpy.node import Node
from ros_gz_interfaces.msg import Contacts

from jackal_rl_navigation.collision_filter import (
    is_robot_obstacle_collision,
)


class CollisionMonitor:
    """Track robot-obstacle contacts from Gazebo.

    ``is_robot_obstacle_collision`` separates those contacts from normal
    ground contact. Only known wheel-support pairs are benign. Chassis-support
    and malformed contacts latch the same safety fault as obstacle contacts.
    This object does not publish velocity.
    """

    def __init__(
        self,
        node: Node,
        topic: str = "/jackal_rl/obstacle_contacts",
        robot_name: str = "j100_0000",
        *,
        on_collision=None,
    ):
        """Subscribe to robot-obstacle contact events."""
        self.topic = topic
        self.robot_name = robot_name
        self._on_collision = on_collision
        self.collision_detected = False
        self.collision_pairs = []
        self.messages_received = 0
        self.valid_messages = 0
        self.malformed_messages = 0
        self.malformed_contacts = 0

        self.subscription = node.create_subscription(
            Contacts,
            topic,
            self._contact_callback,
            10,
        )

    def _iter_contacts(self, msg):
        """Return contact entries, or None when the message is unreadable."""
        if msg is None:
            return None
        try:
            contacts = msg.contacts
        except Exception:
            return None
        if contacts is None or isinstance(contacts, (str, bytes)):
            return None
        try:
            return list(contacts)
        except TypeError:
            return None

    def _pair_names(self, contact):
        try:
            first = contact.collision1.name
            second = contact.collision2.name
        except Exception:
            return None
        if not all(isinstance(name, str) and name.strip()
                   for name in (first, second)):
            return None
        return first, second

    def _contact_callback(self, msg: Contacts):
        self.messages_received += 1
        contacts = self._iter_contacts(msg)
        if contacts is None:
            self.malformed_messages += 1
            self._latch_malformed()
            return

        self.valid_messages += 1
        obstacle = False
        for contact in contacts:
            names = self._pair_names(contact)
            if names is None:
                self.malformed_contacts += 1
                self._latch_malformed()
                continue
            first, second = names
            if is_robot_obstacle_collision(
                first,
                second,
                self.robot_name,
            ):
                self.collision_detected = True
                obstacle = True
                pair = (str(first), str(second))
                if pair not in self.collision_pairs:
                    self.collision_pairs.append(pair)

        if obstacle and self._on_collision is not None:
            self._on_collision(self.get_status())

    def _latch_malformed(self):
        """Unreadable contact data cannot establish safe support contact."""
        self.collision_detected = True
        if self._on_collision is not None:
            self._on_collision(self.get_status())

    def reset(self):
        """Clear collision state before a new episode.

        The navigation interface does not call this while an episode is open.
        """
        self.collision_detected = False
        self.collision_pairs.clear()
        self.messages_received = 0
        self.valid_messages = 0
        self.malformed_messages = 0
        self.malformed_contacts = 0

    def get_status(self):
        """Return current collision diagnostics."""
        return {
            "collision_detected": self.collision_detected,
            "collision_pairs": list(self.collision_pairs),
            "contact_messages_received": self.messages_received,
            "valid_contact_messages": self.valid_messages,
            "malformed_contact_messages": self.malformed_messages,
            "malformed_contacts": self.malformed_contacts,
        }
