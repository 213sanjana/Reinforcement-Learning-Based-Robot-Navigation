"""World selection for house, ground, and the existing open and warehouse worlds.

These checks read YAML and SDF files. They do not launch Gazebo or load a policy.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import numpy as np

from evaluation.world_explore import (
    avoidance_action,
    exploration_goal,
    forward_sector,
    roaming_goal,
    sector_bearing,
)
from jackal_rl_navigation.warehouse_config import load_warehouse_config
from jackal_rl_navigation.world_catalog import apply_world, world_names


ROOT = Path(__file__).resolve().parents[1]


def _boxes(path: Path):
    boxes = []
    for model in ET.parse(path).getroot().findall(".//model"):
        pose = model.findtext("pose")
        size = model.findtext(".//collision/geometry/box/size")
        if not pose or not size:
            continue
        x, y = (float(value) for value in pose.split()[:2])
        sx, sy = (float(value) for value in size.split()[:2])
        boxes.append((x - sx / 2.0, y - sy / 2.0, x + sx / 2.0, y + sy / 2.0))
    return boxes


def _clear(point, boxes, margin: float) -> bool:
    x, y = point
    for min_x, min_y, max_x, max_y in boxes:
        if min_x - margin <= x <= max_x + margin and min_y - margin <= y <= max_y + margin:
            return False
    return True


def test_default_stage_stays_on_the_external_warehouse():
    config = load_warehouse_config(ROOT / "configs" / "warehouse_ppo.yaml")
    assert config["stage_id"] == "external_warehouse"
    assert config["world_name"] == "industrial-warehouse"
    assert "route" not in config


@pytest.mark.parametrize("name,world_name", [("house", "jackal_house"), ("ground", "jackal_ground"), ("open", "jackal_demo_open")])
def test_named_world_selects_its_fortress_file_and_route(name, world_name):
    config = apply_world(load_warehouse_config(ROOT / "configs" / "warehouse_ppo.yaml"), name)
    world_file = ROOT / config["world_file"]
    text = world_file.read_text(encoding="utf-8")
    world = ET.parse(world_file).getroot().find("world")
    assert config["selected_world"] == name
    assert config["world_name"] == world_name
    assert config["gazebo_partition"] == "jackal_rl"
    assert world is not None and world.get("name") == world_name
    assert "libignition-gazebo-physics-system.so" in text
    assert "gz-sim" not in text
    assert "route" not in config
    if name == "house":
        assert "model://cafe" in text
        assert (ROOT / "jackal_rl_ws/worlds/models/cafe/meshes/cafe.dae").is_file()
    boxes = _boxes(world_file)
    for scenario in config["scenarios"]:
        assert _clear(scenario["start"][:2], boxes, 0.30)
        assert _clear(scenario["goal"], boxes, 0.30)


def test_warehouse_name_keeps_the_external_world():
    config = apply_world(load_warehouse_config(ROOT / "configs" / "warehouse_ppo.yaml"), "warehouse")
    assert config["stage_id"] == "external_warehouse"
    assert config["world_name"] == "industrial-warehouse"
    assert "route" not in config


def test_house_session_targets_the_house_partition(monkeypatch):
    pytest.importorskip("rclpy")
    import jackal_rl_navigation.gazebo_env as gazebo_env
    from jackal_rl_navigation.warehouse_env import GazeboWarehouseSession

    captured = {}

    class FakeEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop before the command node")

    monkeypatch.setattr(gazebo_env, "JackalGazeboEnv", FakeEnv)
    config = apply_world(load_warehouse_config(ROOT / "configs" / "warehouse_ppo.yaml"), "house")
    with pytest.raises(RuntimeError, match="stop before the command node"):
        GazeboWarehouseSession(config)
    assert captured["world"] == "jackal_house"
    assert captured["partition"] == "jackal_rl"

    config["gazebo_partition"] = "jackal_rl_external"
    with pytest.raises(RuntimeError, match="jackal_house"):
        GazeboWarehouseSession(config)
    assert captured["partition"] == "jackal_rl"


def test_exploration_goal_follows_the_open_lidar_direction():
    observation = np.ones(40, dtype=np.float32)
    observation[:36] = 0.2
    observation[0] = 0.8
    choice = exploration_goal(observation, (1.0, 2.0, 0.0))
    assert choice["sector"] == 0
    assert choice["goal"][0] < 1.0
    assert choice["goal"][1] == pytest.approx(2.0, abs=0.3)


def test_avoidance_turns_toward_the_open_side_and_stops_when_the_front_is_clear():
    blocked = np.ones(40, dtype=np.float32)
    blocked[15:21] = 0.4 / 12.0
    blocked[9:15] = 0.4 / 12.0
    blocked[21:27] = 3.0 / 12.0
    action = avoidance_action(blocked)
    assert action is not None
    assert float(action[0]) == 0.0
    assert float(action[1]) > 0.0

    blocked[9:15] = 4.0 / 12.0
    blocked[21:27] = 0.5 / 12.0
    action = avoidance_action(blocked)
    assert float(action[1]) < 0.0

    clear = np.ones(40, dtype=np.float32)
    assert avoidance_action(clear) is None

    beside = np.ones(40, dtype=np.float32)
    beside[21:24] = 0.5 / 12.0
    action = avoidance_action(beside)
    assert action is not None
    assert float(action[0]) > 0.0
    assert float(action[1]) < 0.0


def test_opened_front_keeps_the_forward_goal():
    observation = np.full(40, 0.2, dtype=np.float32)
    observation[15:21] = 2.0 / 12.0
    observation[0] = 0.9
    assert forward_sector(observation) == 15
    choice = exploration_goal(
        observation,
        (0.0, 0.0, 0.0),
        min_range_m=0.9,
        preferred_sector=forward_sector(observation),
    )
    assert choice["sector"] == 15
    assert choice["goal"][0] > 0.5


def test_roaming_goal_leaves_the_area_already_driven():
    observation = np.full(40, 0.25, dtype=np.float32)
    observation[0] = 0.9
    observation[18] = 0.5
    choice = roaming_goal(observation, (0.0, 0.0, 0.0), recent=[(-3.0, 0.0)])
    assert abs(sector_bearing(choice["sector"])) < math.radians(100)
    assert choice["goal"][0] > 1.0


def test_unknown_world_is_rejected():
    config = load_warehouse_config(ROOT / "configs" / "warehouse_ppo.yaml")
    with pytest.raises(ValueError, match="Unknown world"):
        apply_world(config, "hospital")
    assert set(world_names()) == {"house", "ground", "open", "warehouse"}
