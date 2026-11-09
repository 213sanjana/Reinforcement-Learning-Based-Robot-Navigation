import json
from pathlib import Path

import pytest

from navigation import collect_expert_demonstrations as collector
from navigation.expert_collection_plan import build_collection_plan


def valid_args(tmp_path):
    world = tmp_path / "test_world.sdf"
    world.write_text("<sdf version='1.7'></sdf>")

    return collector.parse_args([
        "--world", str(world),
        "--output-dir", str(tmp_path / "output"),
        "--episodes", "2",
    ])


def test_preview_keeps_motion_disabled(tmp_path, capsys):
    args = valid_args(tmp_path)

    collector.preview(args)

    config = json.loads(capsys.readouterr().out)

    assert config["movement_enabled"] is False
    assert config["simulation_reset_enabled"] is False
    assert config["collision_monitoring_enabled"] is True


def test_preview_does_not_create_dataset(tmp_path, capsys):
    args = valid_args(tmp_path)

    collector.preview(args)
    capsys.readouterr()

    assert not args.output_dir.exists()


def test_preview_allocates_unique_ids(tmp_path, capsys):
    args = valid_args(tmp_path)

    collector.preview(args)

    config = json.loads(capsys.readouterr().out)

    assert len(config["episode_ids"]) == 2
    assert len(set(config["episode_ids"])) == 2


def test_preview_preserves_existing_episode(tmp_path, capsys):
    args = valid_args(tmp_path)

    args.output_dir.mkdir(parents=True)
    existing = args.output_dir / "expert_episode_0000.npz"
    existing.write_bytes(b"protected")

    collector.preview(args)
    config = json.loads(capsys.readouterr().out)

    assert existing.read_bytes() == b"protected"
    assert 0 not in config["episode_ids"]


def test_invalid_execute_plan_skips_preflight(monkeypatch, tmp_path, capsys):
    calls = []

    def forbidden():
        calls.append("graph")
        raise AssertionError("Preflight must not run")

    import navigation.expert_collection_preflight as graph

    monkeypatch.setattr(graph, "run_preflight", forbidden)

    world = tmp_path / "test_world.sdf"
    world.write_text("<sdf version='1.7'></sdf>")

    code = collector.main([
        "--mode",
        "execute",
        "--world",
        str(world),
        "--output-dir",
        str(tmp_path / "output"),
        "--inflation-radius",
        "0",
    ])

    captured = capsys.readouterr()

    assert code == 1
    assert calls == []
    assert "EXECUTE: FAIL" in captured.out
    assert "No robot commands or simulation resets were issued." in captured.out
    assert not (tmp_path / "output").exists()


def test_execute_preflight_failure_does_not_initialize_environment(
    monkeypatch, tmp_path, capsys
):
    import jackal_rl_navigation.gazebo_env as environment
    import navigation.expert_collection_preflight as graph
    import navigation.expert_live_runner_factory as factory

    def fail_graph():
        return type("Result", (), {
            "passed": False,
            "checks": {"velocity_no_competing_publisher": False},
        })()

    def forbidden(*args, **kwargs):
        raise AssertionError("Environment must not initialize")

    monkeypatch.setattr(graph, "run_preflight", fail_graph)
    monkeypatch.setattr(environment, "JackalGazeboEnv", forbidden)
    monkeypatch.setattr(factory, "create_live_runner", forbidden)

    world = tmp_path / "test_world.sdf"
    world.write_text("<sdf version='1.7'></sdf>")

    code = collector.main([
        "--mode",
        "execute",
        "--world",
        str(world),
        "--output-dir",
        str(tmp_path / "output"),
    ])

    captured = capsys.readouterr()

    assert code == 1
    assert "EXECUTE: BLOCKED" in captured.out
    assert "RUNTIME PREFLIGHT: SKIPPED" in captured.out
    assert "No robot commands or simulation resets were issued." in captured.out
    assert not (tmp_path / "output").exists()


def test_missing_world_rejected_before_plan(tmp_path):
    args = valid_args(tmp_path)
    args.world = tmp_path / "missing.sdf"

    with pytest.raises(FileNotFoundError):
        build_collection_plan(args, [0, 1])


def test_duplicate_episode_ids_rejected(tmp_path):
    args = valid_args(tmp_path)

    with pytest.raises(ValueError, match="Duplicate"):
        build_collection_plan(args, [1, 1])


def test_collision_monitoring_required_in_plan(tmp_path):
    args = valid_args(tmp_path)

    plan = build_collection_plan(args, [0, 1])

    assert plan.collision_config.enabled is True
    assert plan.collision_config.topic == (
        "/jackal_rl/obstacle_contacts"
    )


def test_plan_never_enables_motion(tmp_path):
    args = valid_args(tmp_path)

    plan = build_collection_plan(args, [0, 1])

    assert plan.movement_enabled is False
    assert plan.simulation_reset_enabled is False
