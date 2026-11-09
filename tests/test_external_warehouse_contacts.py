"""Offline synthetic contacts only: no node, publisher, Gazebo or checkpoint."""
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import pytest
from ros_gz_interfaces.msg import Contact, Contacts, Entity
from jackal_rl_navigation.collision_filter import (
    SUPPORT_COLLISIONS, WHEEL_COLLISIONS, is_robot_obstacle_collision,
)
from jackal_rl_navigation.collision_monitor import CollisionMonitor

ROOT = Path(__file__).resolve().parents[1]
FLOOR = ('aws_robomaker_warehouse_GroundB_01_001::'
         'aws_robomaker_warehouse_GroundB_01::ground_link::collision')
GROUND = 'ground_plane::ground_link::collision'
CAFE_FLOOR = 'cafe::link::main_floor'
CHASSIS = 'j100_0000::base_link::base_link_fixed_joint_lump__chassis_link_collision'
SHELF = 'aws_robomaker_warehouse_ShelfF_01_001::aws_robomaker_warehouse_ShelfF_01::link::collision'
WALL = 'aws_robomaker_warehouse_WallB_01_001::aws_robomaker_warehouse_WallB_01::link::collision'


def message(*pairs):
    return Contacts(contacts=[Contact(collision1=Entity(name=a), collision2=Entity(name=b))
                              for a, b in pairs])


@pytest.mark.parametrize('wheel', sorted(WHEEL_COLLISIONS))
@pytest.mark.parametrize('prefix', ['j100_0000::', 'j100_0000/robot::'])
@pytest.mark.parametrize('support', [GROUND, FLOOR, CAFE_FLOOR])
@pytest.mark.parametrize('reverse', [False, True])
def test_known_wheel_support_only_is_benign(wheel, prefix, support, reverse):
    pair = (prefix + wheel, support)
    assert not is_robot_obstacle_collision(*(pair[::-1] if reverse else pair))


@pytest.mark.parametrize('robot', [CHASSIS, 'j100_0000::unknown_link::collision',
    'j100_0000::front_left_wheel_link::unknown_collision',
    'j100_0000::front_left_wheel_link::front_left_wheel_link_collision'])
@pytest.mark.parametrize('external', [SHELF, WALL, 'unknown::link::collision',
    'floor::ground_link::collision', FLOOR + 'extra'])
@pytest.mark.parametrize('reverse', [False, True])
def test_obstacles_and_unknown_entities_fail_closed(robot, external, reverse):
    pair = (robot, external)
    assert is_robot_obstacle_collision(*(pair[::-1] if reverse else pair))


@pytest.mark.parametrize('support', [GROUND, FLOOR, CAFE_FLOOR])
@pytest.mark.parametrize('robot', [CHASSIS, 'j100_0000::unknown_link::collision'])
@pytest.mark.parametrize('reverse', [False, True])
def test_chassis_and_unknown_robot_support_is_a_fault(support, robot, reverse):
    pair = (robot, support)
    assert is_robot_obstacle_collision(*(pair[::-1] if reverse else pair))


def test_identifiers_match_actual_sdf_and_sensor_configuration():
    external = ROOT / 'jackal_rl_ws/worlds/external_warehouse'
    world = ET.parse(external / 'jackal_industrial_warehouse.sdf')
    outer = world.find(".//model[@name='aws_robomaker_warehouse_GroundB_01_001']")
    model_uri = outer.findtext('include/uri').removeprefix('model://')
    model = ET.parse(external / 'models' / model_uri / 'model.sdf').find('model')
    link = model.find('link'); collision = link.find('collision')
    assert '::'.join((outer.get('name'), model.get('name'), link.get('name'), collision.get('name'))) == FLOOR
    assert SUPPORT_COLLISIONS == {FLOOR, GROUND, CAFE_FLOOR}
    house = ET.parse(ROOT / 'jackal_rl_ws/worlds/house/jackal_house.sdf')
    cafe_include = next(
        element for element in house.findall('.//include')
        if element.findtext('name') == 'cafe'
    )
    cafe_model = ET.parse(ROOT / 'jackal_rl_ws/worlds/models/cafe/model.sdf').find('model')
    cafe_link = cafe_model.find('link')
    cafe_floor = cafe_link.find("collision[@name='main_floor']")
    assert cafe_floor is not None
    assert '::'.join((
        cafe_include.findtext('name'),
        cafe_link.get('name'),
        cafe_floor.get('name'),
    )) == CAFE_FLOOR
    for name in ('jackal_navigation_world.sdf', 'jackal_demo_open.sdf'):
        sdf = ET.parse(ROOT / 'jackal_rl_ws/worlds/demo' / name)
        assert sdf.find(".//model[@name='ground_plane']/link[@name='ground_link']/collision[@name='collision']") is not None
    robot = ET.parse(ROOT / 'jackal_rl_ws/config/jackal/robot.urdf.xacro')
    ns = {'x': 'http://www.ros.org/wiki/xacro'}
    sensors = robot.findall('x:jackal_contact_sensor', ns)
    assert {f"{s.get('link_name')}::{s.get('collision_name')}" for s in sensors if 'wheel' in s.get('link_name')} == WHEEL_COLLISIONS


def test_mixed_contacts_latch_and_later_support_cannot_clear_collision():
    callback = Mock(); monitor = CollisionMonitor(Mock(), on_collision=callback)
    wheel = 'j100_0000::front_left_wheel_link::front_left_wheel_link_collision'
    monitor._contact_callback(message())
    assert not monitor.collision_detected
    monitor._contact_callback(message((wheel, FLOOR), (WALL, wheel), (CHASSIS, SHELF)))
    assert monitor.collision_pairs == [(WALL, wheel), (CHASSIS, SHELF)]
    assert callback.call_count == 1
    monitor._contact_callback(message((FLOOR, wheel)))
    monitor._contact_callback(message())
    assert monitor.collision_detected
    assert len(monitor.collision_pairs) == 2


@pytest.mark.parametrize('bad', [None, S(), S(contacts=None), S(contacts='bad'),
    S(contacts=[S()]), S(contacts=[S(collision1=S(name=''), collision2=S(name=FLOOR))]),
    S(contacts=[S(collision1=S(name=123), collision2=S(name=FLOOR))])])
def test_malformed_messages_latch_safety_fault(bad):
    callback = Mock(); monitor = CollisionMonitor(Mock(), on_collision=callback)
    monitor._contact_callback(bad)
    assert monitor.collision_detected
    assert monitor.malformed_messages + monitor.malformed_contacts == 1
    assert callback.call_count == 1


def test_unknown_robot_entity_is_not_ignored_as_self_contact():
    assert is_robot_obstacle_collision(CHASSIS, 'j100_0000::unknown::collision')
    assert is_robot_obstacle_collision('j100_0000::unknown::collision', CHASSIS)
