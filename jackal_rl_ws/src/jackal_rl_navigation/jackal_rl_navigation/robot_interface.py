"""ROS 2 interface for Jackal RL navigation."""

import math
import time

import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PointStamped

from tf2_ros import Buffer, TransformListener

from jackal_rl_navigation.observation_builder import ObservationBuilder
from jackal_rl_navigation.action_mapper import ActionMapper, VelocityCommand
from jackal_rl_navigation.safety_monitor import SafetyMonitor


class JackalRLInterface(Node):
    """Connect Jackal sensors, observations and safe velocity commands."""

    def __init__(self):
        super().__init__("jackal_rl_interface")

        self.declare_parameter("namespace", "j100_0000")
        self.declare_parameter("lidar_bins", 36)
        self.declare_parameter("goal_x", 3.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_frame", "odom")
        self.declare_parameter("sensor_timeout", 0.5)

        namespace = (
            self.get_parameter("namespace")
            .value.strip("/")
        )

        self.scan_topic = (
            f"/{namespace}/sensors/lidar3d_0/scan"
        )
        self.odom_topic = (
            f"/{namespace}/platform/odom"
        )
        self.declare_parameter(
            "command_topic",
            f"/{namespace}/platform/cmd_vel_unstamped",
        )

        self.command_topic = str(
            self.get_parameter("command_topic").value
        )

        self.observation_builder = ObservationBuilder(
            lidar_bins=self.get_parameter("lidar_bins").value
        )

        self.action_mapper = ActionMapper()

        self.safety_monitor = SafetyMonitor(
            sensor_timeout=self.get_parameter("sensor_timeout").value
        )

        self.declare_parameter("enable_goal_loop", False)
        self.declare_parameter("goal_controller", "")
        self._goal_pilot = None
        self._pilot_started = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.scan = None
        self.odom = None

        self.scan_received = None
        self.odom_received = None

        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.goal_frame = self.get_parameter("goal_frame").value

        self.last_command_time = None
        self.last_publish_time = None

        self.publisher = self.create_publisher(
            Twist,
            self.command_topic,
            10,
        )

        self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Odometry,
            self.odom_topic,
            self._odom_callback,
            qos_profile_sensor_data,
        )

        self.watchdog = self.create_timer(
            0.05,
            self._watchdog_callback,
        )

        self.diagnostics = self.create_timer(
            2.0,
            self._diagnostic_callback,
        )

        # The contact subscription exists before main() can enable the goal loop.
        self._initialize_collision_monitor()
        self._log_collision_preflight()

        self.get_logger().info(
            f"Jackal RL interface initialized: {namespace}"
        )

    def _scan_callback(self, msg):
        self.scan = msg
        self.scan_received = time.monotonic()

    def _odom_callback(self, msg):
        self.odom = msg
        self.odom_received = time.monotonic()

    def _sensor_ages(self):
        now = time.monotonic()

        if self.scan_received is None or self.odom_received is None:
            return None

        return (
            now - self.scan_received,
            now - self.odom_received,
        )

    def _goal_relative(self):
        """Calculate goal distance and heading in the robot frame."""

        if self.odom is None:
            return None

        odom_frame = self.odom.header.frame_id

        # Use odometry directly when the goal is in its reference frame.
        if self.goal_frame == odom_frame:
            position = self.odom.pose.pose.position
            orientation = self.odom.pose.pose.orientation

            yaw = math.atan2(
                2.0 * (
                    orientation.w * orientation.z
                    + orientation.x * orientation.y
                ),
                1.0 - 2.0 * (
                    orientation.y ** 2
                    + orientation.z ** 2
                ),
            )

            dx = self.goal_x - position.x
            dy = self.goal_y - position.y

            distance = math.hypot(dx, dy)
            heading = math.atan2(dy, dx) - yaw

            heading = math.atan2(
                math.sin(heading),
                math.cos(heading),
            )

            return distance, heading

        # Other goal frames still require TF.
        robot_frame = self.odom.child_frame_id

        if not robot_frame:
            return None

        goal = PointStamped()
        goal.header.frame_id = self.goal_frame
        goal.header.stamp = Time().to_msg()
        goal.point.x = self.goal_x
        goal.point.y = self.goal_y

        try:
            transformed = self.tf_buffer.transform(
                goal,
                robot_frame,
                timeout=Duration(seconds=0.1),
            )
        except Exception as exc:
            self.get_logger().debug(f"TF unavailable: {exc}")
            return None

        return (
            math.hypot(
                transformed.point.x,
                transformed.point.y,
            ),
            math.atan2(
                transformed.point.y,
                transformed.point.x,
            ),
        )

    def get_observation(self):
        """Return a valid 40-dimensional observation or None."""

        ages = self._sensor_ages()

        if ages is None:
            return None

        scan_age, odom_age = ages

        timeout = self.safety_monitor.sensor_timeout

        if scan_age > timeout or odom_age > timeout:
            return None

        goal = self._goal_relative()

        if goal is None:
            return None

        distance, heading = goal

        ranges = self.observation_builder.expand_partial_scan(
            self.scan.ranges,
            self.scan.angle_min,
            self.scan.angle_increment,
            self.scan.angle_max,
        )
        result = self.observation_builder.build(
            lidar_ranges=ranges,
            goal_distance=distance,
            goal_heading=heading,
            linear_velocity=self.odom.twist.twist.linear.x,
            angular_velocity=self.odom.twist.twist.angular.z,
        )

        if not result.valid:
            return None

        return result.vector

    def _publish_velocity(self, linear, angular):
        if getattr(self, "_collision_latched", False):
            linear = 0.0
            angular = 0.0

        msg = Twist()

        msg.linear.x = float(linear)
        msg.angular.z = float(angular)

        if not rclpy.ok(context=self.context):
            return

        try:
            self.publisher.publish(msg)
        except Exception as exc:
            self.get_logger().warning(
                f"Velocity publish unavailable: {exc}"
            )
            return False

        self.last_publish_time = time.monotonic()

        callback = getattr(self, "expert_publish_callback", None)
        if callback is not None:
            try:
                callback(
                    float(msg.linear.x),
                    float(msg.angular.z),
                    int(self.get_clock().now().nanoseconds),
                )
            except Exception as exc:
                self.get_logger().warning(
                    f"Expert recording callback failed: {exc}"
                )

        return True

    def stop(self):
        """Immediately publish zero velocity."""

        self.action_mapper.emergency_stop()
        self._publish_velocity(0.0, 0.0)
        self.last_command_time = None

    def _robot_name_for_collision(self):
        try:
            name = str(self.get_parameter("namespace").value).strip("/")
        except Exception:
            name = ""
        return name or "j100_0000"

    def _import_collision_dependencies(self):
        """Import the bridged contact type and the existing monitor."""
        from ros_gz_interfaces.msg import Contacts
        from jackal_rl_navigation.collision_config import CollisionConfig
        from jackal_rl_navigation.collision_monitor import CollisionMonitor
        return Contacts, CollisionConfig, CollisionMonitor

    def _initialize_collision_monitor(self):
        """Subscribe to obstacle contacts before a goal loop can start.

        Failure leaves ``collision_monitor`` unset. The goal loop must not
        start in that state. A launch file that names the bridge is not
        treated as evidence that the topic is active.
        """

        self.collision_topic = "/jackal_rl/obstacle_contacts"
        self.collision_message_type = "ros_gz_interfaces/msg/Contacts"
        self.collision_monitor = None
        self.collision_init_error = None
        self._collision_latched = False
        self._collision_contact_details = []
        try:
            contacts_type, config_type, monitor_type = (
                self._import_collision_dependencies()
            )
            if getattr(contacts_type, "__name__", "") != "Contacts":
                raise ImportError(
                    "ros_gz_interfaces/msg/Contacts is unavailable"
                )
            robot_name = self._robot_name_for_collision()
            config = config_type(
                enabled=True,
                topic=self.collision_topic,
                robot_name=robot_name,
            )
            if config.topic != self.collision_topic:
                raise RuntimeError(
                    "collision topic does not match the bridged contact topic"
                )
            monitor = monitor_type(
                self,
                topic=config.topic,
                robot_name=config.robot_name,
                on_collision=self._on_obstacle_collision,
            )
            if getattr(monitor, "subscription", None) is None:
                raise RuntimeError("contact subscription was not created")
            if getattr(monitor, "topic", None) != self.collision_topic:
                raise RuntimeError(
                    "collision monitor topic does not match the bridged contact topic"
                )
            self.collision_monitor = monitor
            self.collision_robot_name = config.robot_name
        except Exception as exc:
            self.collision_monitor = None
            self.collision_init_error = f"{type(exc).__name__}: {exc}"

    def _require_collision_monitor(self):
        """Refuse the goal loop when contact monitoring is not ready."""
        monitor = getattr(self, "collision_monitor", None)
        error = getattr(self, "collision_init_error", None)
        topic = None
        subscription = None
        if monitor is not None:
            topic = getattr(monitor, "topic", None) or getattr(
                self, "collision_topic", None
            )
            subscription = getattr(monitor, "subscription", None)
        if (
            monitor is None
            or error
            or topic != "/jackal_rl/obstacle_contacts"
            or subscription is None
        ):
            detail = error or "collision monitor is not initialized"
            if monitor is not None and topic != "/jackal_rl/obstacle_contacts":
                detail = (
                    "collision topic is not /jackal_rl/obstacle_contacts"
                )
            elif monitor is not None and subscription is None and not error:
                detail = "contact subscription was not created"
            message = (
                "Goal loop refused because collision monitoring is unavailable: "
                + str(detail)
            )
            logger = getattr(self, "get_logger", None)
            if callable(logger):
                logger().error(message)
            raise RuntimeError(message)

    def _monitor_reports_collision(self):
        monitor = getattr(self, "collision_monitor", None)
        if monitor is None:
            return False
        return bool(getattr(monitor, "collision_detected", False))

    def _capture_contact_details(self):
        monitor = getattr(self, "collision_monitor", None)
        pairs = []
        if monitor is not None:
            raw = getattr(monitor, "collision_pairs", []) or []
            pairs = [tuple(pair) for pair in raw]
        if pairs:
            self._collision_contact_details = pairs

    def _sync_collision_latch(self):
        """Latch a monitor collision once the goal pilot owns commands.

        Returns True when motion must stay at zero. The latch is not cleared
        here.
        """
        if getattr(self, "_goal_pilot", None) is None:
            return False
        if getattr(self, "_collision_latched", False):
            self._capture_contact_details()
            return True
        if self._monitor_reports_collision():
            self._collision_latched = True
            self._capture_contact_details()
            return True
        return False

    def _on_obstacle_collision(self, status=None):
        """Publish zero immediately after a robot-obstacle contact.

        Ground contacts do not reach this callback. The latch stays set for
        the rest of the episode.
        """
        pairs = []
        if isinstance(status, dict):
            pairs = [tuple(pair) for pair in (status.get("collision_pairs") or [])]
        if pairs:
            self._collision_contact_details = pairs
        if getattr(self, "_goal_pilot", None) is None:
            return
        self._collision_latched = True
        self.stop()

    @property
    def collision_contact_details(self):
        """Contact pairs remembered for this episode.

        ``JackalEpisodeRecord`` stores the collision count and termination
        reason. It has no contact-pair field, so the pairs stay beside the
        record.
        """
        return list(getattr(self, "_collision_contact_details", []) or [])

    def reset_collision_latch(self):
        """Clear collision memory only between episodes.

        Contact callbacks, the watchdog, and ``step_navigation`` do not call
        this. After a collision, reset the pilot to idle with no goal, then
        call this method before the next episode.
        """
        pilot = getattr(self, "_goal_pilot", None)
        if pilot is not None:
            from navigation.jackal_navigation_state import NavState
            manager = pilot.manager
            if manager.state is not NavState.IDLE or manager.goal is not None:
                raise RuntimeError(
                    "Collision latch cannot be cleared during an episode"
                )
        self._collision_latched = False
        self._collision_contact_details = []
        monitor = getattr(self, "collision_monitor", None)
        reset = getattr(monitor, "reset", None)
        if callable(reset):
            reset()

    def collision_preflight(self):
        """Report whether contact monitoring is initialized and receiving.

        ``receiving_valid_contacts`` becomes true only after a readable
        contact message arrives. A declared launch bridge is not used.
        ``collision_free_operation`` is never true: silence on the contact
        topic is not evidence of collision-free operation.
        """
        monitor = getattr(self, "collision_monitor", None)
        error = getattr(self, "collision_init_error", None)
        initialized = monitor is not None and not error
        status = {}
        if initialized and hasattr(monitor, "get_status"):
            try:
                status = dict(monitor.get_status())
            except Exception:
                status = {}
                initialized = False
                error = error or "collision status unavailable"

        def _count(key, attr):
            if key in status:
                return int(status[key])
            if monitor is None:
                return 0
            return int(getattr(monitor, attr, 0) or 0)

        received = _count("contact_messages_received", "messages_received")
        valid = _count("valid_contact_messages", "valid_messages")
        malformed = _count("malformed_contact_messages", "malformed_messages")
        observed = bool(status.get("collision_detected", False))
        if not observed and monitor is not None:
            observed = bool(getattr(monitor, "collision_detected", False))
        latched = bool(getattr(self, "_collision_latched", False))
        pairs = list(status.get("collision_pairs") or [])
        if not pairs and monitor is not None:
            pairs = [tuple(pair) for pair in (getattr(monitor, "collision_pairs", []) or [])]
        topic = getattr(self, "collision_topic", "/jackal_rl/obstacle_contacts")
        message_type = getattr(
            self, "collision_message_type", "ros_gz_interfaces/msg/Contacts"
        )
        if not initialized:
            stream = "not_initialized"
            interpretation = (
                "Collision monitoring is not initialized. "
                "The goal loop must not start. "
                "This is not evidence of collision-free operation. "
                "A declared launch bridge does not mean the contact topic is active."
            )
        elif valid <= 0:
            stream = "waiting_for_valid_contacts"
            interpretation = (
                f"Collision monitoring is initialized on {topic} "
                f"({message_type}), but no valid contact message has been received. "
                "This is not evidence of collision-free operation. "
                "A declared launch bridge does not mean the contact topic is active."
            )
        elif latched:
            stream = "collision_latched"
            interpretation = (
                "A robot-obstacle contact was received and the collision latch is set. "
                f"Contact pairs: {pairs}."
            )
        elif observed:
            stream = "obstacle_contact_observed"
            interpretation = (
                "A robot-obstacle contact was received. "
                f"Contact pairs: {pairs}. "
                "The motion latch is set when the goal pilot is bound."
            )
        else:
            stream = "contacts_received_no_obstacle_collision"
            interpretation = (
                "Valid contact messages have been received and none contained "
                "a robot-obstacle contact. "
                "This is not proof of collision-free operation."
            )
        return {
            "initialized": initialized,
            "init_error": error,
            "topic": topic,
            "message_type": message_type,
            "subscription_created": bool(
                initialized and getattr(monitor, "subscription", None) is not None
            ),
            "contact_messages_received": received,
            "valid_contact_messages": valid,
            "malformed_contact_messages": malformed,
            "receiving_valid_contacts": bool(initialized and valid > 0),
            "obstacle_contact_observed": bool(initialized and (observed or latched)),
            "collision_latched": bool(initialized and latched),
            "collision_pairs": pairs,
            "collision_free_operation": False if (latched or observed) else None,
            "bridge_declaration_used": False,
            "stream": stream,
            "interpretation": interpretation,
        }

    def _log_collision_preflight(self):
        report = self.collision_preflight()
        logger = self.get_logger()
        if not report["initialized"] or report["collision_latched"]:
            logger.error(report["interpretation"])
        elif not report["receiving_valid_contacts"]:
            logger.warning(report["interpretation"])
        else:
            logger.info(report["interpretation"])

    def configure_goal_pilot(self, controller="classical_baseline", start_pose=(0.0, 0.0, 0.0)):
        """Bind the classical baseline to this node's mapper, safety check, and publisher.

        The publisher created in ``__init__`` remains the only velocity publisher.
        The goal loop stays disabled when the contact monitor is unavailable.
        """

        from navigation.jackal_goal_pilot import GoalPilot, refuse_checkpoint_controller

        if self._goal_pilot is not None:
            raise RuntimeError("A goal pilot is already bound to this publisher")
        refuse_checkpoint_controller(controller)
        self._require_collision_monitor()
        self._goal_pilot = GoalPilot.classical(
            self.action_mapper,
            self.safety_monitor,
            self._publish_velocity,
            start_pose=start_pose,
        )
        if self._monitor_reports_collision():
            self._collision_latched = True
            self._capture_contact_details()
            self.stop()
        return self._goal_pilot

    def set_navigation_goal(
        self,
        x,
        y,
        scenario_id,
        map_id,
        split,
        seed,
        tolerance_m=0.30,
        timeout_s=60.0,
        # 300 steps at the 0.05 s interval ended the 4 m warehouse run
        # at 15 s. 1200 steps lets the existing 60 s timeout bind instead.
        max_steps=1200,
        yaw=None,
    ):
        """Store a goal in meters and hand it to the bound pilot."""

        from navigation.jackal_navigation_manager import GoalSpec

        if self._goal_pilot is None:
            raise RuntimeError("Goal pilot is not configured")
        self.goal_x = float(x)
        self.goal_y = float(y)
        pose = self._odom_pose()
        if pose is not None and self._goal_pilot.manager.steps == 0:
            self._goal_pilot.manager.start_pose = pose
        self._pilot_started = time.monotonic()
        return self._goal_pilot.manager.set_goal(GoalSpec(
            x=self.goal_x,
            y=self.goal_y,
            yaw=yaw,
            tolerance_m=float(tolerance_m),
            timeout_s=float(timeout_s),
            max_steps=int(max_steps),
            scenario_id=scenario_id,
            map_id=map_id,
            split=split,
            seed=int(seed),
        ))

    def _odom_pose(self):
        if self.odom is None:
            return None
        from jackal_rl_navigation.goal_frames import odom_pose_from_message
        return odom_pose_from_message(self.odom)

    def step_navigation(self, collision=False, elapsed_s=None):
        """Run one goal cycle. Mapping happens inside the pilot, then one publish."""

        from navigation.jackal_navigation_state import TERMINAL_STATES, NavState

        pilot = self._goal_pilot
        if pilot is None:
            raise RuntimeError("Goal pilot is not configured")
        if pilot.manager.goal is None:
            raise RuntimeError("Navigation goal is not set")
        if elapsed_s is None:
            if self._pilot_started is None:
                self._pilot_started = time.monotonic()
            elapsed_s = time.monotonic() - self._pilot_started

        # The default argument is not the collision source. The monitor is.
        collision_active = bool(collision) or self._sync_collision_latch()
        if collision_active:
            self._collision_latched = True
            self._capture_contact_details()

        now = time.monotonic()
        if (
            not collision_active
            and pilot.state not in TERMINAL_STATES
            and pilot.state is not NavState.IDLE
            and self.last_command_time is not None
        ):
            dt = now - self.last_command_time
            if dt < self.action_mapper.min_command_interval:
                return None
        else:
            dt = self.action_mapper.min_command_interval

        ages = self._sensor_ages()
        pose = self._odom_pose()
        goal = self._goal_relative() if ages is not None else None
        observation = None
        distance = float("inf")
        pose_xy = pose[:2] if pose is not None else (0.0, 0.0)
        scan_age = self.safety_monitor.sensor_timeout + 1.0
        odom_age = scan_age
        ranges = []
        angles = []
        if ages is not None and self.scan is not None and goal is not None:
            scan_age, odom_age = ages
            distance = float(goal[0])
            ranges = self.scan.ranges
            angles = (
                float(self.scan.angle_min)
                + np.arange(len(self.scan.ranges))
                * float(self.scan.angle_increment)
            )
            observation = self.observation_builder.build(
                lidar_ranges=ranges,
                goal_distance=goal[0],
                goal_heading=goal[1],
                linear_velocity=self.odom.twist.twist.linear.x,
                angular_velocity=self.odom.twist.twist.angular.z,
            )

        def run_cycle(is_collision):
            return pilot.cycle(
                observation,
                ranges,
                angles,
                scan_age,
                odom_age,
                distance,
                bool(is_collision),
                dt,
                pose_xy,
                float(elapsed_s),
            )

        decision, published = run_cycle(collision_active)
        if (
            getattr(self, "_collision_latched", False)
            and pilot.state not in TERMINAL_STATES
            and pilot.state is not NavState.IDLE
        ):
            self._capture_contact_details()
            decision, published = run_cycle(True)
        if decision.commanding and published is True:
            self.last_command_time = now
        if pilot.state in TERMINAL_STATES:
            self.last_command_time = None
        if getattr(self, "_collision_latched", False):
            pilot.manager.collision_contact_details = tuple(
                self.collision_contact_details
            )
        return decision

    def command_action(self, action):
        """Apply one normalized action after safety validation.

        Values are dimensionless and must lie in [-1, 1]. Physical velocities
        are not accepted here. While a goal pilot is bound, this method
        refuses so the pilot remains the only command owner.
        """

        if getattr(self, "_goal_pilot", None) is not None:
            raise RuntimeError("The goal pilot owns velocity commands")

        if getattr(self, "_collision_latched", False):
            self.last_safety_reason = "Collision"
            self.stop()
            return False

        self.last_safety_reason = ""

        if self.safety_monitor.emergency_active:
            self.last_safety_reason = "Emergency stop"
            self.stop()
            return False

        if self.safety_monitor.episode_terminated:
            self.last_safety_reason = "Episode terminated"
            self.stop()
            return False

        ages = self._sensor_ages()

        if ages is None:
            self.last_safety_reason = "Sensors unavailable"
            self.stop()
            return False

        now = time.monotonic()

        if self.last_command_time is None:
            dt = self.action_mapper.min_command_interval
        else:
            dt = now - self.last_command_time

        if dt < self.action_mapper.min_command_interval:
            return False

        try:
            try:
                from navigation.jackal_policy import validate_action
                validate_action(action)
            except ModuleNotFoundError:
                pass
            command = self.action_mapper.map_action(action, dt)
        except (ValueError, TypeError):
            self.last_safety_reason = "Invalid action"
            self.stop()
            return False

        angles = (
            float(self.scan.angle_min)
            + np.arange(len(self.scan.ranges))
            * float(self.scan.angle_increment)
        )

        safety = self.safety_monitor.evaluate(
            linear=command.linear,
            angular=command.angular,
            lidar_ranges=self.scan.ranges,
            lidar_angles=angles,
            scan_age=ages[0],
            odom_age=ages[1],
        )

        if not safety.safe:
            self.last_safety_reason = safety.reason
            self.get_logger().warning(
                f"Safety stop: {safety.reason}"
            )
            self.stop()
            return False

        if safety.reason != "Command permitted":
            # Keep the acceleration limiter aligned with the command that
            # was actually accepted. A reduced turn must not remain stored
            # as if the wheels had already reached it.
            self.last_safety_reason = safety.reason
            self.action_mapper.previous = VelocityCommand(
                float(safety.linear),
                float(safety.angular),
            )
            self.get_logger().warning(
                f"Safety reduced command: {safety.reason}"
            )

        published = self._publish_velocity(
            safety.linear, safety.angular
        )

        if published is not True:
            self.last_safety_reason = "Velocity publish failed"
            self.stop()
            return False

        self.last_command_time = now

        return True

    def emergency_stop(self):
        self.safety_monitor.emergency_stop()
        pilot = getattr(self, "_goal_pilot", None)
        if pilot is not None and pilot.manager.goal is not None:
            from navigation.jackal_navigation_state import TERMINAL_STATES, NavState
            if pilot.state is not NavState.IDLE and pilot.state not in TERMINAL_STATES:
                pilot.manager.emergency_stop()
        self.stop()

    def terminate_episode(self):
        self.safety_monitor.terminate_episode()
        self.stop()

    def _watchdog_callback(self):
        if self._sync_collision_latch():
            self.stop()
            return

        ages = self._sensor_ages()

        if ages is None:
            self.stop()
            return

        if max(ages) > self.safety_monitor.sensor_timeout:
            self.stop()
            return

        if self.last_command_time is not None:
            elapsed = time.monotonic() - self.last_command_time
            hold_s = float(getattr(self, "command_hold_s", 0.4))

            if elapsed > hold_s:
                self.stop()

    def _diagnostic_callback(self):
        self._log_collision_preflight()

        if self.scan is None:
            self.get_logger().warning("WAITING: LiDAR messages")
            return

        if self.odom is None:
            self.get_logger().warning("WAITING: Odometry messages")
            return

        ages = self._sensor_ages()

        if ages is None:
            self.get_logger().warning("WAITING: Sensor timestamps")
            return

        if max(ages) > self.safety_monitor.sensor_timeout:
            self.get_logger().warning(
                f"STALE: scan={ages[0]:.3f}s, "
                f"odom={ages[1]:.3f}s"
            )
            return

        goal = self._goal_relative()

        if goal is None:
            self.get_logger().warning(
                "TF FAILURE: "
                f"goal_frame={self.goal_frame}, "
                f"robot_frame={self.odom.child_frame_id}"
            )
            return

        result = self.observation_builder.build(
            lidar_ranges=self.scan.ranges,
            goal_distance=goal[0],
            goal_heading=goal[1],
            linear_velocity=self.odom.twist.twist.linear.x,
            angular_velocity=self.odom.twist.twist.angular.z,
        )

        if not result.valid:
            self.get_logger().warning(
                f"OBSERVATION INVALID: {result.reason}"
            )
            return

        self.get_logger().info(
            f"OBSERVATION READY: shape={result.vector.shape}, "
            f"finite={np.all(np.isfinite(result.vector))}"
        )


def main():
    rclpy.init()

    node = JackalRLInterface()

    try:
        enabled = node.get_parameter("enable_goal_loop").value
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {"1", "true", "yes"}
        else:
            enabled = enabled is True
        if not enabled:
            # Observation-only mode: no movement commands.
            rclpy.spin(node)
        else:
            node._require_collision_monitor()
            controller = str(node.get_parameter("goal_controller").value or "classical_baseline")
            node.configure_goal_pilot(controller)
            node.set_navigation_goal(
                node.goal_x,
                node.goal_y,
                scenario_id="pilot_goal",
                map_id="pilot",
                split="fixed",
                seed=0,
            )
            from navigation.jackal_navigation_state import TERMINAL_STATES
            while rclpy.ok() and node._goal_pilot.state not in TERMINAL_STATES:
                rclpy.spin_once(node, timeout_sec=0.05)
                node.step_navigation()

    except KeyboardInterrupt:
        pass

    finally:
        node.stop()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
