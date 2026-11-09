"""Stationary, observation-only access to the external warehouse.

This session is separate from ``JackalGazeboEnv``. It subscribes to LiDAR,
odometry, and contacts, and it makes one read-only ``list_controllers``
query. It does not construct a velocity publisher, a command watchdog, or a
Gazebo world-control client. ``reset()`` waits for active controllers and
for fresh, advancing sensors, then returns the 40-D observation. It leaves
an emergency stop or a collision latch set. ``step()`` is refused.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np

from jackal_rl_navigation.collision_filter import is_robot_obstacle_collision
from jackal_rl_navigation.observation_builder import ObservationBuilder


REQUIRED_WORLD = "industrial-warehouse"
REQUIRED_PARTITION = "jackal_rl_external"
REQUIRED_NAMESPACE = "j100_0000"
REQUIRED_CONTROLLERS = frozenset({
    "joint_state_broadcaster",
    "platform_velocity_controller",
})
FRESHNESS_S = 0.5
COMMAND_TOPIC = "/j100_0000/platform/cmd_vel_unstamped"


@dataclass(frozen=True)
class SensorSample:
    """One LiDAR and odometry pair already converted out of ROS messages."""

    ranges: np.ndarray
    x: float
    y: float
    yaw: float
    linear_x: float
    linear_y: float
    angular_z: float
    scan_stamp_ns: int
    previous_scan_stamp_ns: int | None
    odom_stamp_ns: int
    previous_odom_stamp_ns: int | None
    scan_received: float
    odom_received: float
    scan_advanced_at: float | None
    odom_advanced_at: float | None


def require_external_warehouse_target(config) -> None:
    """Accept only the external warehouse on its own partition.

    A missing value, ``jackal_obstacle_test``, or ``jackal_rl`` is a
    configuration error. This function does not fill in a default world or
    partition and does not change the process environment.
    """

    world = config.get("world_name")
    partition = config.get("gazebo_partition")
    if world != REQUIRED_WORLD or partition != REQUIRED_PARTITION:
        raise RuntimeError(
            "Stationary observation requires world 'industrial-warehouse' and "
            "partition 'jackal_rl_external'. "
            f"Got world {world!r} and partition {partition!r}."
        )
    namespace = config.get("namespace", REQUIRED_NAMESPACE)
    if namespace != REQUIRED_NAMESPACE:
        raise RuntimeError(
            "Stationary observation is bound to namespace 'j100_0000'. "
            f"Got {namespace!r}."
        )


def assert_process_partition(environ=None) -> None:
    """Refuse to attach unless this process is already on the external partition.

    The observation path does not assign ``IGN_PARTITION`` or ``GZ_PARTITION``.
    """

    environ = os.environ if environ is None else environ
    ign = environ.get("IGN_PARTITION")
    gz = environ.get("GZ_PARTITION")
    if ign != REQUIRED_PARTITION or gz != REQUIRED_PARTITION:
        raise RuntimeError(
            "Process partition is not jackal_rl_external. "
            "The observation path will not change IGN_PARTITION or GZ_PARTITION."
        )


def require_active_controllers(response) -> None:
    """Require both Jackal controllers to be active in a real list response.

    Only a ``list_controllers`` payload counts. A missing readiness attribute
    is not treated as ready, and only ``state == "active"`` counts.
    """

    if not isinstance(response, dict):
        raise RuntimeError("controller_manager response is missing")
    controllers = response.get("controller")
    if not isinstance(controllers, list):
        raise RuntimeError("controller_manager response is missing")
    active = set()
    for item in controllers:
        if not isinstance(item, dict):
            raise RuntimeError("controller_manager response is missing")
        name = item.get("name")
        state = item.get("state")
        if isinstance(name, str) and state == "active":
            active.add(name)
    missing = REQUIRED_CONTROLLERS - active
    if missing:
        raise RuntimeError(
            "controllers are not active: " + ", ".join(sorted(missing))
        )


def controllers_dict(response) -> dict:
    """Copy a ``list_controllers`` service result into a plain mapping."""

    if response is None or not hasattr(response, "controller"):
        raise RuntimeError("controller_manager response is missing")
    try:
        items = list(response.controller)
    except TypeError as exc:
        raise RuntimeError("controller_manager response is missing") from exc
    copied = []
    for item in items:
        name = getattr(item, "name", None)
        state = getattr(item, "state", None)
        if not isinstance(name, str) or not isinstance(state, str):
            raise RuntimeError("controller_manager response is missing")
        copied.append({"name": name, "state": state})
    return {"controller": copied}


def _stamp_ns(stamp) -> int:
    try:
        sec = stamp.sec
        nanosec = stamp.nanosec
    except AttributeError as exc:
        raise RuntimeError("sensor timestamp is invalid") from exc
    if isinstance(sec, bool) or isinstance(nanosec, bool):
        raise RuntimeError("sensor timestamp is invalid")
    if not isinstance(sec, int) or not isinstance(nanosec, int):
        raise RuntimeError("sensor timestamp is invalid")
    if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
        raise RuntimeError("sensor timestamp is invalid")
    return sec * 1_000_000_000 + nanosec


def _yaw(orientation) -> float:
    sin_yaw = 2.0 * (
        float(orientation.w) * float(orientation.z)
        + float(orientation.x) * float(orientation.y)
    )
    cos_yaw = 1.0 - 2.0 * (
        float(orientation.y) ** 2 + float(orientation.z) ** 2
    )
    if not math.isfinite(sin_yaw) or not math.isfinite(cos_yaw):
        raise RuntimeError("odometry orientation is invalid")
    return math.atan2(sin_yaw, cos_yaw)


def sample_from_messages(
    scan,
    odom,
    *,
    previous_scan_stamp_ns,
    previous_odom_stamp_ns,
    scan_received,
    odom_received,
    scan_advanced_at,
    odom_advanced_at,
    odom_frame,
) -> SensorSample:
    """Build a sample from one scan and one odometry message."""

    try:
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        twist = odom.twist.twist
        frame = odom.header.frame_id
        scan_stamp = _stamp_ns(scan.header.stamp)
        odom_stamp = _stamp_ns(odom.header.stamp)
        yaw = _yaw(orientation)
        sample = SensorSample(
            ranges=ranges,
            x=float(position.x),
            y=float(position.y),
            yaw=yaw,
            linear_x=float(twist.linear.x),
            linear_y=float(getattr(twist.linear, "y", 0.0)),
            angular_z=float(twist.angular.z),
            scan_stamp_ns=scan_stamp,
            previous_scan_stamp_ns=previous_scan_stamp_ns,
            odom_stamp_ns=odom_stamp,
            previous_odom_stamp_ns=previous_odom_stamp_ns,
            scan_received=float(scan_received),
            odom_received=float(odom_received),
            scan_advanced_at=None if scan_advanced_at is None else float(scan_advanced_at),
            odom_advanced_at=None if odom_advanced_at is None else float(odom_advanced_at),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("LiDAR or odometry message is incomplete") from exc
    if ranges.ndim != 1:
        raise RuntimeError("LiDAR ranges are invalid")
    if frame != odom_frame:
        raise RuntimeError("odometry frame is not the configured odom frame")
    return sample


def validate_sensor_sample(sample: SensorSample, *, now: float, freshness_s: float) -> None:
    """Require both sensors to be finite, fresh, and strictly advancing."""

    if freshness_s != FRESHNESS_S:
        raise RuntimeError("sensor freshness must be 0.5 seconds")
    if not math.isfinite(now):
        raise RuntimeError("observation clock is invalid")
    _require_advancing(
        sample.scan_stamp_ns,
        sample.previous_scan_stamp_ns,
        sample.scan_received,
        sample.scan_advanced_at,
        now,
        "LiDAR",
    )
    _require_advancing(
        sample.odom_stamp_ns,
        sample.previous_odom_stamp_ns,
        sample.odom_received,
        sample.odom_advanced_at,
        now,
        "odometry",
    )
    for label, value in (
        ("pose", sample.x),
        ("pose", sample.y),
        ("yaw", sample.yaw),
        ("linear velocity", sample.linear_x),
        ("linear velocity", sample.linear_y),
        ("angular velocity", sample.angular_z),
    ):
        if not math.isfinite(float(value)):
            raise RuntimeError(f"{label} is invalid")


def _require_advancing(stamp, previous, received, advanced_at, now, label) -> None:
    if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp < 0:
        raise RuntimeError(f"{label} timestamp is invalid")
    if previous is None or isinstance(previous, bool) or not isinstance(previous, int):
        raise RuntimeError(f"{label} is not advancing")
    if previous < 0 or stamp <= previous:
        raise RuntimeError(f"{label} is not advancing")
    if advanced_at is None or not math.isfinite(received) or not math.isfinite(advanced_at):
        raise RuntimeError(f"{label} is stale")
    age = now - received
    since_advance = now - advanced_at
    if age < 0 or age > FRESHNESS_S or since_advance < 0 or since_advance > FRESHNESS_S:
        raise RuntimeError(f"{label} is stale")


def iter_contact_pairs(message):
    """Return contact name pairs, or None when the message is unreadable."""

    if message is None or not hasattr(message, "contacts"):
        return None
    try:
        contacts = list(message.contacts)
    except TypeError:
        return None
    pairs = []
    for contact in contacts:
        try:
            first = contact.collision1.name
            second = contact.collision2.name
        except AttributeError:
            return None
        if not all(isinstance(name, str) and name.strip() for name in (first, second)):
            return None
        pairs.append((first, second))
    return pairs


def pairs_are_fault(pairs, robot_name: str) -> bool:
    """Malformed contacts and obstacle contacts are safety faults."""

    if not isinstance(pairs, list):
        return True
    for item in pairs:
        if not isinstance(item, tuple) or len(item) != 2:
            return True
        if is_robot_obstacle_collision(item[0], item[1], robot_name):
            return True
    return False


def bind_observation_subscriptions(node, config, callbacks, message_types, qos=None):
    """Subscribe to sensors and open a read-only controller client.

    The returned client is the only service handle. This function does not
    create a publisher or a timer, and it refuses the velocity command topic.
    LiDAR and odometry use the caller's QoS profiles. The live profile is
    sensor data, which is what the Gazebo bridge publishes.
    """

    lidar = config.get("lidar_topic")
    odom = config.get("odom_topic")
    contacts = config.get("contact_topic")
    namespace = config.get("namespace")
    topics = (lidar, odom, contacts)
    if not all(isinstance(topic, str) and topic.startswith("/") for topic in topics):
        raise RuntimeError("Observation topics are missing")
    if COMMAND_TOPIC in topics:
        raise RuntimeError("Refusing to use the velocity command topic as a sensor")
    if namespace != REQUIRED_NAMESPACE:
        raise RuntimeError("Stationary observation is bound to namespace 'j100_0000'")
    profiles = {"scan": 10, "odom": 10, "contacts": 10}
    if qos is not None:
        profiles.update(qos)
    node.create_subscription(message_types["scan"], lidar, callbacks["scan"], profiles["scan"])
    node.create_subscription(message_types["odom"], odom, callbacks["odom"], profiles["odom"])
    node.create_subscription(
        message_types["contacts"], contacts, callbacks["contacts"], profiles["contacts"],
    )
    service = f"/{namespace}/controller_manager/list_controllers"
    return node.create_client(message_types["controllers"], service)


def live_subscription_qos() -> dict:
    """QoS that can receive the bridge's LiDAR and odometry streams."""

    from rclpy.qos import qos_profile_sensor_data

    return {
        "scan": qos_profile_sensor_data,
        "odom": qos_profile_sensor_data,
        "contacts": 10,
    }


def load_ros_message_types() -> dict:
    """Import the ROS types used by the subscriber-only transport."""

    from controller_manager_msgs.srv import ListControllers
    from nav_msgs.msg import Odometry
    from ros_gz_interfaces.msg import Contacts
    from sensor_msgs.msg import LaserScan

    return {
        "scan": LaserScan,
        "odom": Odometry,
        "contacts": Contacts,
        "controllers": ListControllers,
    }


def default_node_factory():
    """Create the observer node after the caller has set the external partition.

    ``rclpy`` is initialized only when this factory runs. Construction of the
    session with an injected transport does not call it.
    """

    import rclpy

    started = False
    if not rclpy.ok():
        rclpy.init()
        started = True
    node = rclpy.create_node("warehouse_stationary_observer")
    node._warehouse_started_rclpy = started
    return node


class _SensorState:
    """Remember the latest stamp and the previous one so an advance is visible."""

    def __init__(self):
        self.stamp = None
        self.previous = None
        self.received = None
        self.advanced_at = None
        self.message = None
        self.fault = None

    def note(self, message, stamp: int, received: float, label: str) -> None:
        if self.fault is not None:
            return
        if self.stamp is None:
            self.stamp = stamp
            self.received = received
            self.message = message
            return
        if stamp < self.stamp:
            self.fault = f"{label} timestamp moved backwards"
            return
        if stamp > self.stamp:
            self.previous = self.stamp
            self.stamp = stamp
            self.advanced_at = received
        self.received = received
        self.message = message


class RosObservationTransport:
    """Subscriber-only ROS transport. It has no publisher and no watchdog timer."""

    def __init__(self, config, *, node, message_types, owns_node: bool = False, qos=None):
        require_external_warehouse_target(config)
        assert_process_partition()
        self.config = config
        self.node = node
        self.owns_node = owns_node
        self.robot_name = REQUIRED_NAMESPACE
        self.odom_frame = config.get("odom_frame", "odom")
        self.timeout_s = float(config["limits"].sensor_timeout_s)
        self._scan = _SensorState()
        self._odom = _SensorState()
        self._pairs = []
        self.collision_latched = False
        self.emergency_active = False
        self._request_type = message_types["controllers"]
        self._client = bind_observation_subscriptions(
            node,
            config,
            {
                "scan": self._on_scan,
                "odom": self._on_odom,
                "contacts": self._on_contacts,
            },
            message_types,
            qos,
        )

    def command_publishers(self):
        """Return nodes publishing the velocity command topic, if any.

        This is a graph query. It does not advertise the topic.
        """

        query = getattr(self.node, "get_publishers_info_by_topic", None)
        if not callable(query):
            raise RuntimeError("command publisher graph is unavailable")
        names = []
        for info in query(COMMAND_TOPIC):
            node_name = getattr(info, "node_name", None)
            names.append(node_name if isinstance(node_name, str) and node_name else "unknown")
        return names

    def _on_scan(self, message) -> None:
        try:
            stamp = _stamp_ns(message.header.stamp)
        except RuntimeError:
            self._scan.fault = "LiDAR timestamp is invalid"
            return
        self._scan.note(message, stamp, time.monotonic(), "LiDAR")

    def _on_odom(self, message) -> None:
        try:
            stamp = _stamp_ns(message.header.stamp)
        except RuntimeError:
            self._odom.fault = "odometry timestamp is invalid"
            return
        self._odom.note(message, stamp, time.monotonic(), "odometry")

    def _on_contacts(self, message) -> None:
        pairs = iter_contact_pairs(message)
        if pairs_are_fault(pairs, self.robot_name):
            self.collision_latched = True
            return
        if not self.collision_latched:
            self._pairs = pairs

    def contact_pairs(self):
        return list(self._pairs)

    def list_controllers(self, *, spin=None):
        if self._client is None:
            raise RuntimeError("controller_manager response is missing")
        if not self._client.wait_for_service(timeout_sec=self.timeout_s):
            raise RuntimeError("controller_manager list_controllers is not available")
        future = self._client.call_async(self._request_type.Request())
        deadline = time.monotonic() + self.timeout_s
        while not future.done():
            if time.monotonic() >= deadline:
                raise RuntimeError("controller_manager timed out")
            (spin or self._spin)(0.05)
        return controllers_dict(future.result())

    def wait_for_advancing_sensors(self, timeout_s: float, freshness_s: float, *, spin=None):
        if freshness_s != FRESHNESS_S:
            raise RuntimeError("sensor freshness must be 0.5 seconds")
        deadline = time.monotonic() + float(timeout_s)
        while True:
            fault = self._scan.fault or self._odom.fault
            if fault is not None:
                raise RuntimeError(fault)
            sample = self._sample()
            if sample is not None:
                try:
                    validate_sensor_sample(sample, now=time.monotonic(), freshness_s=freshness_s)
                    return sample
                except RuntimeError as exc:
                    if "invalid" in str(exc) or "incomplete" in str(exc):
                        raise
            if time.monotonic() >= deadline:
                raise TimeoutError("LiDAR and odometry did not advance before the timeout")
            (spin or self._spin)(min(0.05, max(0.0, deadline - time.monotonic())))

    def _sample(self):
        if self._scan.message is None or self._odom.message is None:
            return None
        return sample_from_messages(
            self._scan.message,
            self._odom.message,
            previous_scan_stamp_ns=self._scan.previous,
            previous_odom_stamp_ns=self._odom.previous,
            scan_received=self._scan.received,
            odom_received=self._odom.received,
            scan_advanced_at=self._scan.advanced_at,
            odom_advanced_at=self._odom.advanced_at,
            odom_frame=self.odom_frame,
        )

    def _spin(self, timeout_sec: float) -> None:
        import rclpy

        rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def close(self) -> None:
        if not self.owns_node:
            return
        started = bool(getattr(self.node, "_warehouse_started_rclpy", False))
        destroy = getattr(self.node, "destroy_node", None)
        if destroy is not None:
            destroy()
        if started:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()


class StationaryObservationSession:
    """Read one 40-D observation without commanding the robot or the simulator."""

    def __init__(
        self,
        config,
        *,
        transport=None,
        connect: bool = False,
        node_factory=None,
        message_types=None,
        subscription_qos=None,
    ):
        require_external_warehouse_target(config)
        if transport is not None and connect:
            raise RuntimeError("Pass either an observation transport or connect, not both")
        self.config = config
        self.emergency_latched = False
        self.collision_latched = False
        self._builder = ObservationBuilder()
        self._robot_name = REQUIRED_NAMESPACE
        if transport is not None:
            self.transport = transport
            return
        if not connect:
            self.transport = None
            return
        assert_process_partition()
        qos = live_subscription_qos() if subscription_qos is None else subscription_qos
        factory = node_factory or default_node_factory
        node = factory()
        types = message_types if message_types is not None else load_ros_message_types()
        self.transport = RosObservationTransport(
            config,
            node=node,
            message_types=types,
            owns_node=node_factory is None,
            qos=qos,
        )

    def reset(self, *, seed, scenario):
        del seed
        if self.emergency_latched:
            raise RuntimeError("Emergency stop is latched; reset cannot clear it")
        if self.collision_latched:
            raise RuntimeError("Collision latch is set; reset cannot clear it")
        if self.transport is None:
            raise RuntimeError(
                "No observation transport is connected. "
                "Refusing to create a command interface."
            )
        self._require_no_command_publishers()
        require_active_controllers(self.transport.list_controllers())
        timeout_s = float(self.config["limits"].sensor_timeout_s)
        try:
            sample = self.transport.wait_for_advancing_sensors(timeout_s, FRESHNESS_S)
        except TimeoutError as exc:
            raise RuntimeError("LiDAR and odometry timed out") from exc
        validate_sensor_sample(sample, now=time.monotonic(), freshness_s=FRESHNESS_S)
        self._require_stationary(sample)
        self._require_safe()
        return self._observation(sample, scenario)

    def step(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "step() is blocked on the stationary observation path. "
            "It does not publish velocity or step the simulator."
        )

    def halt(self) -> None:
        """There is no command path to halt."""

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if close is not None:
            close()

    def _require_no_command_publishers(self) -> None:
        query = getattr(self.transport, "command_publishers", None)
        if not callable(query):
            return
        names = [str(name) for name in query()]
        if names:
            raise RuntimeError(
                "A velocity command publisher is already present: " + ", ".join(names)
            )

    def _require_stationary(self, sample: SensorSample) -> None:
        linear_limit = float(self.config.get("stationary_linear_limit_mps", 0.02))
        angular_limit = float(self.config.get("stationary_angular_limit_radps", 0.05))
        linear = math.hypot(sample.linear_x, sample.linear_y)
        if linear > linear_limit or abs(sample.angular_z) > angular_limit:
            raise RuntimeError("Robot is not stationary")

    def _require_safe(self) -> None:
        reported = getattr(self.transport, "emergency_active", False)
        if callable(reported):
            reported = reported()
        if reported:
            self.emergency_latched = True
        pairs = []
        if hasattr(self.transport, "contact_pairs"):
            pairs = self.transport.contact_pairs()
        if bool(getattr(self.transport, "collision_latched", False)) or pairs_are_fault(
            pairs, self._robot_name
        ):
            self.collision_latched = True
        if self.emergency_latched:
            raise RuntimeError("Emergency stop is latched; reset cannot clear it")
        if self.collision_latched:
            raise RuntimeError("Collision latch is set; reset cannot clear it")

    def _observation(self, sample: SensorSample, scenario):
        goal = scenario.get("goal") if isinstance(scenario, dict) else None
        if goal is None or len(goal) < 2:
            raise RuntimeError("Scenario goal is missing")
        goal_x = float(goal[0])
        goal_y = float(goal[1])
        if not math.isfinite(goal_x) or not math.isfinite(goal_y):
            raise RuntimeError("Scenario goal is invalid")
        dx = goal_x - sample.x
        dy = goal_y - sample.y
        distance = math.hypot(dx, dy)
        heading = math.atan2(dy, dx) - sample.yaw
        heading = math.atan2(math.sin(heading), math.cos(heading))
        result = self._builder.build(
            sample.ranges,
            distance,
            heading,
            sample.linear_x,
            sample.angular_z,
        )
        vector = np.asarray(result.vector, dtype=np.float32)
        if (
            not result.valid
            or vector.shape != (40,)
            or vector.dtype != np.float32
            or not bool(np.all(np.isfinite(vector)))
        ):
            raise RuntimeError("Reset did not produce a valid observation")
        return vector.copy(), {
            "pose_xy": (sample.x, sample.y),
            "goal_distance": distance,
            "world_name": REQUIRED_WORLD,
            "gazebo_partition": REQUIRED_PARTITION,
            "observation_valid": True,
            "collision_state": False,
        }
