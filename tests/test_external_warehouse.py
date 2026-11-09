"""Static checks for the external industrial warehouse. No simulator is started."""

import hashlib
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "jackal_rl_ws/worlds/external_warehouse"
ORIGINAL = EXTERNAL / "original/industrial-warehouse.sdf"
INTEGRATION = EXTERNAL / "jackal_industrial_warehouse.sdf"
LAUNCH = (
    ROOT
    / "jackal_rl_ws/src/jackal_rl_bringup/launch/jackal_external_warehouse.launch.py"
)
NAVIGATION = ROOT / "jackal_rl_ws/worlds/demo/jackal_navigation_world.sdf"
OPEN_WORLD = ROOT / "jackal_rl_ws/worlds/demo/jackal_demo_open.sdf"
ORIGINAL_SHA256 = "a80de9f6d76e6b589a8b080ccdbce87bd1b88cce9b88cf9de6c81af10602c161"
NAVIGATION_SHA256 = "564a600181d298cd127b4dbdd63f721247058f8daed03017044c0ee0104e4bfc"
OPEN_SHA256 = "1257816e504b25fa68ed0b324c2524b36735cd6d57642fd8637422441332824e"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_existing_worlds_are_unchanged():
    assert _sha(ORIGINAL) == ORIGINAL_SHA256
    assert _sha(NAVIGATION) == NAVIGATION_SHA256
    assert _sha(OPEN_WORLD) == OPEN_SHA256


def test_integration_keeps_every_original_obstacle_pose():
    original = ORIGINAL.read_text(encoding="utf-8")
    integration = INTEGRATION.read_text(encoding="utf-8")
    assert 'world name="industrial-warehouse"' in integration
    names = re.findall(r'<model name="([^"]+)"', integration)
    assert names[-1] == "goal_marker"
    assert all("jackal" not in name.lower() and "j100" not in name for name in names)
    marker = integration.split('<model name="goal_marker">', 1)[1].split("</model>", 1)[0]
    assert "<collision" not in marker
    for line in original.splitlines():
        if "<pose>" in line or 'model name=' in line:
            assert line.strip() in integration
    assert "fuel.ignitionrobotics.org" not in integration.split("<world", 1)[1]
    assert integration.count("<uri>model://aws_robomaker_warehouse_") == 25


def test_downloaded_models_have_separate_collision_meshes():
    models = EXTERNAL / "models"
    assert len(list(models.iterdir())) == 12
    for model in models.iterdir():
        text = (model / "model.sdf").read_text(encoding="utf-8")
        assert "<collision" in text
        assert "<visual" in text
        assert "_collision." in text or "collision." in text.lower()


def test_fortress_parser_accepts_the_integration_world():
    env = os.environ.copy()
    env["SDF_PATH"] = str(EXTERNAL / "models")
    result = subprocess.run(
        ["ign", "sdf", "-k", str(INTEGRATION)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "Valid." in result.stderr or "Valid." in result.stdout


def test_external_launch_stays_stationary_and_refuses_a_second_server():
    text = LAUNCH.read_text(encoding="utf-8")
    assert 'default_value="simulation_only"' in text
    assert 'default_value="industrial-warehouse"' in text
    assert "jackal_rl_external" in text
    assert "does not start a second server" in text
    assert "Do not use killall." in text
    assert "initialized_ppo.zip" not in text
    assert "rl_navigation does not load a checkpoint" in text
    assert '"-z", "0.3"' in text
    compile(text, str(LAUNCH), "exec")
