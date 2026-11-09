
import ast
from pathlib import Path

LAUNCH = Path(
    "jackal_rl_ws/src/jackal_rl_bringup/"
    "launch/jackal_sim.launch.py"
)

def test_collision_world_selected():
    source = LAUNCH.read_text()
    assert '"jackal_obstacle_test_collision.sdf"' in source

def test_contact_bridge_configured():
    source = LAUNCH.read_text()
    assert "/jackal_rl/obstacle_contacts" in source
    assert "ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts" in source

def test_contact_bridge_in_launch_description():
    source = LAUNCH.read_text()
    tree = ast.parse(source)
    assert tree is not None
    assert "        contact_bridge," in source

def test_single_velocity_controller_definition():
    source = LAUNCH.read_text()
    assert source.count("velocity_controller = Node(") == 1
