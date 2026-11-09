"""Static checks for the Jackal demonstration files.

The checks read source, SDF, and YAML. They do not launch ROS 2 or Gazebo,
import rclpy, or publish a velocity.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

from navigation.jackal_policy import OBSERVATION_DIM


class PreflightError(ValueError):
    """A demonstration file is missing or internally inconsistent."""


@dataclass
class PreflightReport:
    ok: bool
    checks: list[str] = field(default_factory=list)
    live_authorization_files: list[str] = field(default_factory=list)

    def add(self, message):
        self.checks.append(message)


def _repo_root():
    return Path(__file__).resolve().parents[1]


def validate_navigation_parameters(params):
    """Validate the ROS parameter dictionary stored under ros__parameters."""

    if not isinstance(params, dict):
        raise PreflightError("Navigation parameters must be a mapping")
    required = (
        "namespace",
        "lidar_bins",
        "goal_x",
        "goal_y",
        "goal_frame",
        "sensor_timeout",
    )
    missing = [key for key in required if key not in params]
    if missing:
        raise PreflightError(f"Missing navigation parameters: {missing}")
    namespace = params["namespace"]
    if not isinstance(namespace, str) or not namespace.strip("/"):
        raise PreflightError("namespace must be a non-empty string")
    if params["lidar_bins"] != 36:
        raise PreflightError("lidar_bins must stay 36 for the existing observation")
    if params["goal_frame"] not in {"odom", "map"}:
        raise PreflightError("goal_frame must be odom or map")
    for key in ("goal_x", "goal_y", "sensor_timeout"):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PreflightError(f"{key} must be numeric")
        if not math.isfinite(float(value)):
            raise PreflightError(f"{key} must be finite")
    if not 0 < float(params["sensor_timeout"]) <= 2.0:
        raise PreflightError("sensor_timeout must be in (0, 2] seconds")
    return {
        "namespace": namespace.strip("/"),
        "scan_topic": f"/{namespace.strip('/')}/sensors/lidar3d_0/scan",
        "odom_topic": f"/{namespace.strip('/')}/platform/odom",
        "command_topic": f"/{namespace.strip('/')}/platform/cmd_vel_unstamped",
        "goal_frame": params["goal_frame"],
        "observation_dim": OBSERVATION_DIM,
    }


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _models(world):
    return [child for child in list(world) if _local(child.tag) == "model"]


def _box_aabb(model):
    pose_text = model.findtext("pose") or "0 0 0 0 0 0"
    pose = [float(part) for part in pose_text.split()]
    size_node = model.find("./link/collision/geometry/box/size")
    if size_node is None or size_node.text is None:
        return None
    size = [float(part) for part in size_node.text.split()]
    return (
        pose[0] - size[0] / 2.0,
        pose[0] + size[0] / 2.0,
        pose[1] - size[1] / 2.0,
        pose[1] + size[1] / 2.0,
    )


def _inside(x, y, box, margin):
    return box[0] - margin <= x <= box[1] + margin and box[2] - margin <= y <= box[3] + margin


def validate_world(path, start_xy, goal_xy, margin=0.25):
    """Parse one Fortress SDF and reject a start or goal inside an obstacle."""

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise PreflightError(f"{path} is not valid XML: {exc}") from exc
    if _local(root.tag) != "sdf" or root.attrib.get("version") != "1.8":
        raise PreflightError(f"{path} must be an SDF 1.8 world")
    world = root.find("world")
    if world is None or not world.attrib.get("name"):
        raise PreflightError(f"{path} needs a named world")
    plugins = " ".join(node.attrib.get("filename", "") for node in root.iter("plugin"))
    if "libignition-gazebo-physics-system.so" not in plugins:
        raise PreflightError(f"{path} is missing the Fortress physics plugin")
    if "libignition-gazebo-contact-system.so" not in plugins:
        raise PreflightError(f"{path} is missing the contact plugin")
    names = [model.attrib.get("name", "") for model in _models(world)]
    if "ground_plane" not in names or "goal_marker" not in names:
        raise PreflightError(f"{path} needs a ground plane and a goal marker")
    goal = world.find("./model[@name='goal_marker']")
    if goal.find(".//collision") is not None:
        raise PreflightError(f"{path} goal marker must be visual only")
    boxes = []
    for model in _models(world):
        name = model.attrib.get("name", "")
        if name in {"ground_plane", "goal_marker"}:
            continue
        box = _box_aabb(model)
        if box is None:
            raise PreflightError(f"{path} model {name} needs a box collision")
        if model.find(".//sensor[@type='contact']") is None:
            raise PreflightError(f"{path} model {name} needs a contact sensor")
        boxes.append(box)
        if _inside(start_xy[0], start_xy[1], box, margin):
            raise PreflightError(f"{path} start pose intersects {name}")
        if _inside(goal_xy[0], goal_xy[1], box, margin):
            raise PreflightError(f"{path} goal intersects {name}")
    return {"world_name": world.attrib["name"], "obstacle_count": len(boxes)}


def _parse_launch(path):
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise PreflightError(f"{path} has a Python syntax error: {exc}") from exc
    return source


def run_static_preflight(root=None):
    """Check worlds, scenario manifest, parameters, frames, and launch syntax."""

    root = Path(root) if root is not None else _repo_root()
    report = PreflightReport(ok=True)
    parameter_file = root / "jackal_rl_ws/src/jackal_rl_navigation/config/navigation.yaml"
    loaded = yaml.safe_load(parameter_file.read_text(encoding="utf-8"))
    params = loaded["jackal_rl_interface"]["ros__parameters"]
    topics = validate_navigation_parameters(params)
    report.add(f"navigation parameters describe {topics['scan_topic']}")

    control = (root / "jackal_rl_ws/config/jackal/platform/config/control.yaml").read_text(encoding="utf-8")
    if "odom_frame_id: 'odom'" not in control or "base_frame_id: 'base_link'" not in control:
        raise PreflightError("control.yaml lost the odom or base_link frame")
    report.add("control.yaml keeps odom and base_link")

    interface = (root / "jackal_rl_ws/src/jackal_rl_navigation/jackal_rl_navigation/robot_interface.py").read_text(encoding="utf-8")
    for fragment in ("sensors/lidar3d_0/scan", "platform/odom", "platform/cmd_vel_unstamped"):
        if fragment not in interface:
            raise PreflightError(f"robot_interface.py no longer names {fragment}")
    report.add("robot interface topic suffixes match the parameter contract")

    lidar = (root / "jackal_rl_ws/config/jackal/sensors/config/lidar3d_0.yaml").read_text(encoding="utf-8")
    if "/j100_0000/sensors/lidar3d_0/scan" not in lidar:
        raise PreflightError("LiDAR bridge topic does not match the robot namespace")
    report.add("LiDAR bridge topic matches j100_0000")

    manifest_path = root / "configs/jackal_demo_scenarios.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "jackal_demo_scenarios_v1":
        raise PreflightError("Unexpected scenario schema")
    if manifest.get("evaluated") is not False:
        raise PreflightError("The scenario manifest must stay marked unevaluated")
    worlds = manifest.get("worlds") or []
    if len(worlds) != 6:
        raise PreflightError("Expected six demonstration worlds")
    unseen = manifest.get("unseen_routes") or []
    if len(unseen) < 5:
        raise PreflightError("Expected at least five unevaluated unseen routes")
    if any(route.get("evaluated") for route in unseen):
        raise PreflightError("Unseen routes must not be marked evaluated")
    names = set()
    for world in worlds:
        path = root / world["file"]
        if not path.is_file():
            raise PreflightError(f"Missing world file {path}")
        checked = validate_world(path, world["start"], world["goal"])
        if checked["world_name"] != world["world_name"]:
            raise PreflightError(f"{path} world name does not match the manifest")
        if world["world_name"] in names:
            raise PreflightError("World names must be unique")
        names.add(world["world_name"])
        report.add(f"{world['id']} parsed with {checked['obstacle_count']} obstacles")

    worlds_by_id = {world["id"]: world for world in worlds}
    for route in unseen:
        host = worlds_by_id.get(route.get("world_id"))
        if host is None:
            raise PreflightError(f"Unseen route {route.get('id')} names an unknown world")
        validate_world(root / host["file"], route["start"], route["goal"])
        report.add(f"{route['id']} is geometrically clear and unevaluated")

    rviz = (root / "jackal_rl_ws/src/jackal_rl_navigation/config/jackal_demo.rviz").read_text(encoding="utf-8")
    if "Fixed Frame: odom" not in rviz or topics["scan_topic"] not in rviz:
        raise PreflightError("RViz config lost the odom frame or LiDAR topic")
    report.add("RViz config names the LiDAR topic and odom frame")

    navigation_launch = root / "jackal_rl_ws/src/jackal_rl_navigation/launch/demo_navigation.launch.py"
    navigation_source = _parse_launch(navigation_launch)
    forbidden = ("ExecuteProcess", "subprocess", "ign gazebo", "cmd_vel")
    found = [token for token in forbidden if token in navigation_source]
    if found:
        raise PreflightError(f"demo_navigation.launch.py contains {found}")
    report.add("demo navigation launch parses and does not start Gazebo")

    sim_launch = root / "jackal_rl_ws/src/jackal_rl_bringup/launch/jackal_demo_sim.launch.py"
    sim_source = _parse_launch(sim_launch)
    if "ign" not in sim_source or "gazebo" not in sim_source:
        raise PreflightError("The demo simulation launch no longer names Gazebo")
    report.live_authorization_files.append(str(sim_launch.relative_to(root)))
    report.add("demo simulation launch parses and is marked authorization-required")
    check_open_pilot_configuration(root, report)
    return report


def check_open_pilot_configuration(root, report):
    """Compare the open-world pilot files. This does not start a process."""

    root = Path(root)
    navigation_launch = (
        root / "jackal_rl_ws/src/jackal_rl_navigation/launch/demo_navigation.launch.py"
    ).read_text(encoding="utf-8")
    required = (
        'default_value="4.0"',
        'default_value="0.0"',
        'default_value="odom"',
        'default_value="classical_baseline"',
        'default_value="false"',
        "goal_controller",
        "enable_goal_loop",
        'lidar_bins": 36',
    )
    missing = [item for item in required if item not in navigation_launch]
    if missing:
        raise PreflightError(f"demo navigation launch is missing {missing}")
    if "ExecuteProcess" in navigation_launch or "subprocess" in navigation_launch:
        raise PreflightError("demo navigation launch gained a process start")

    simulation_launch = (
        root / "jackal_rl_ws/src/jackal_rl_bringup/launch/jackal_demo_sim.launch.py"
    ).read_text(encoding="utf-8")
    for item in (
        "jackal_demo_open.sdf",
        'default_value="jackal_demo_open"',
        '"-z", "0.3"',
        "TimerAction",
        "OnProcessExit",
        "joint_state_broadcaster",
        "platform_velocity_controller",
        "/sensors/lidar3d_0/scan",
    ):
        if item not in simulation_launch:
            raise PreflightError(f"demo simulation launch is missing {item}")

    manifest = yaml.safe_load(
        (root / "configs/jackal_demo_scenarios.yaml").read_text(encoding="utf-8")
    )
    world = next(item for item in manifest["worlds"] if item["id"] == "world1_open")
    if world["goal"][:2] != [4.0, 0.0] or world["world_name"] != "jackal_demo_open":
        raise PreflightError("Open-world scenario does not match the pilot launch goal")
    if manifest.get("namespace") != "j100_0000" or manifest.get("spawn_z_m") != 0.3:
        raise PreflightError("Scenario namespace or spawn height drifted")
    checked = validate_world(root / world["file"], world["start"], world["goal"])
    if checked["obstacle_count"] != 0 or checked["world_name"] != "jackal_demo_open":
        raise PreflightError("The open pilot world is not an empty named world")

    rviz = (
        root / "jackal_rl_ws/src/jackal_rl_navigation/config/jackal_demo.rviz"
    ).read_text(encoding="utf-8")
    for item in (
        "Fixed Frame: odom",
        "/j100_0000/sensors/lidar3d_0/scan",
        "/j100_0000/platform/odom",
    ):
        if item not in rviz:
            raise PreflightError(f"RViz config is missing {item}")
    report.add("Open-world pilot files agree on goal, frame, topics, and spawn height")
