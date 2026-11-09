
import json
import subprocess
import sys

import pytest

from navigation.collect_expert_demonstrations import (
    allocate_episode_ids,
    existing_episode_ids,
    parse_args,
)


def test_default_preview_mode():
    args = parse_args([])
    assert args.mode == "preview"
    assert args.episodes == 1


def test_reject_zero_episodes():
    with pytest.raises(SystemExit):
        parse_args(["--episodes", "0"])


def test_reject_negative_episodes():
    with pytest.raises(SystemExit):
        parse_args(["--episodes", "-2"])


def test_reject_nonfinite_start():
    with pytest.raises(SystemExit):
        parse_args([
            "--start", "nan", "0", "0"
        ])


def test_reject_nonfinite_goal():
    with pytest.raises(SystemExit):
        parse_args([
            "--goal", "inf", "0"
        ])


def test_reject_negative_inflation():
    with pytest.raises(SystemExit):
        parse_args([
            "--inflation-radius", "-0.1"
        ])


def test_reject_zero_duration():
    with pytest.raises(SystemExit):
        parse_args([
            "--duration", "0"
        ])


def test_missing_world_rejected(tmp_path):
    with pytest.raises(SystemExit):
        parse_args([
            "--world",
            str(tmp_path / "missing.sdf"),
        ])


def test_empty_output_allocates_zero(tmp_path):
    assert allocate_episode_ids(tmp_path, 3) == [0, 1, 2]


def test_existing_npz_protected(tmp_path):
    (tmp_path / "expert_episode_0000.npz").touch()

    assert existing_episode_ids(tmp_path) == {0}
    assert allocate_episode_ids(tmp_path, 2) == [1, 2]


def test_existing_json_protected(tmp_path):
    (tmp_path / "expert_episode_0001.json").touch()

    assert existing_episode_ids(tmp_path) == {1}
    assert allocate_episode_ids(tmp_path, 3) == [0, 2, 3]


def test_existing_pair_protected(tmp_path):
    (tmp_path / "expert_episode_0002.npz").touch()
    (tmp_path / "expert_episode_0002.json").touch()

    assert existing_episode_ids(tmp_path) == {2}
    assert allocate_episode_ids(tmp_path, 2) == [0, 1]


def test_preview_does_not_create_output_directory(tmp_path):
    output = tmp_path / "new_dataset"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "navigation.collect_expert_demonstrations",
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0

    config = json.loads(result.stdout)

    assert config["movement_enabled"] is False
    assert config["simulation_reset_enabled"] is False
    assert config["collision_monitoring_enabled"] is True
    assert not output.exists()


def test_execute_rejects_invalid_plan_without_creating_output(tmp_path):
    world = tmp_path / "world.sdf"
    world.write_text("<sdf version='1.7'></sdf>")
    output = tmp_path / "blocked_dataset"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "navigation.collect_expert_demonstrations",
            "--mode",
            "execute",
            "--world",
            str(world),
            "--output-dir",
            str(output),
            "--inflation-radius",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode != 0
    assert "Inflation radius must be positive" in result.stderr
    assert "No robot commands or simulation resets were issued." in result.stdout
    assert "EXECUTE: PASS" not in result.stdout
    assert not output.exists()
