"""Static checks for the Gazebo launch modes. This file starts no process."""

from pathlib import Path

LAUNCH = (
    Path(__file__).resolve().parents[1]
    / "jackal_rl_ws/src/jackal_rl_bringup/launch/jackal_demo_sim.launch.py"
)


def test_launch_defaults_stay_stationary_on_the_open_world():
    text = LAUNCH.read_text(encoding="utf-8")
    assert "jackal_demo_open.sdf" in text
    assert 'default_value="jackal_demo_open"' in text
    assert 'DeclareLaunchArgument("mode", default_value="simulation_only")' in text
    assert "No navigation node is started" in text
    assert "not publish velocity." in text


def test_moving_modes_are_explicit_and_rl_loads_no_checkpoint():
    text = LAUNCH.read_text(encoding="utf-8")
    assert "classical_navigation" in text
    assert 'enable_goal_loop", "true"' in text
    assert "rl_navigation does not load a checkpoint" in text
    assert "initialized_ppo.zip" not in text
    assert "evaluation mode does not start a moving episode" in text


def test_launch_still_refuses_a_second_gazebo_server():
    text = LAUNCH.read_text(encoding="utf-8")
    assert "A Gazebo server is already running" in text
    assert "does not start a second server" in text
    assert "Do not use killall" in text
    compile(text, str(LAUNCH), "exec")
