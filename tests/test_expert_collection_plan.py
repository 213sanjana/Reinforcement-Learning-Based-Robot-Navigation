from pathlib import Path
from types import SimpleNamespace

import pytest

from navigation.expert_collection_plan import build_collection_plan


def arguments(tmp_path):
    world = tmp_path / "world.sdf"
    world.write_text("<sdf version='1.7'></sdf>")

    return SimpleNamespace(
        world=world,
        output_dir=tmp_path / "output",
        episodes=2,
        start=(0.0, 0.0, 0.0),
        goal=(3.0, 0.0),
        duration=45.0,
        inflation_radius=0.95,
    )


def test_valid_plan(tmp_path):
    plan = build_collection_plan(arguments(tmp_path), [0, 1])

    assert len(plan.episodes) == 2
    assert plan.collision_config.enabled
    assert not plan.movement_enabled
    assert not plan.simulation_reset_enabled


def test_missing_world(tmp_path):
    args = arguments(tmp_path)
    args.world = tmp_path / "missing.sdf"

    with pytest.raises(FileNotFoundError):
        build_collection_plan(args, [0, 1])


def test_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        build_collection_plan(arguments(tmp_path), [0, 0])


def test_negative_id(tmp_path):
    with pytest.raises(ValueError):
        build_collection_plan(arguments(tmp_path), [-1, 1])


def test_id_count_mismatch(tmp_path):
    with pytest.raises(ValueError):
        build_collection_plan(arguments(tmp_path), [0])


def test_nonfinite_goal(tmp_path):
    args = arguments(tmp_path)
    args.goal = (float("nan"), 0.0)

    with pytest.raises(ValueError):
        build_collection_plan(args, [0, 1])


def test_invalid_duration(tmp_path):
    args = arguments(tmp_path)
    args.duration = 0.0

    with pytest.raises(ValueError):
        build_collection_plan(args, [0, 1])


def test_invalid_inflation(tmp_path):
    args = arguments(tmp_path)
    args.inflation_radius = -1.0

    with pytest.raises(ValueError):
        build_collection_plan(args, [0, 1])


def test_no_output_directory_created(tmp_path):
    args = arguments(tmp_path)
    plan = build_collection_plan(args, [0, 1])

    assert not plan.output_dir.exists()


def test_plan_is_immutable(tmp_path):
    from dataclasses import FrozenInstanceError

    plan = build_collection_plan(arguments(tmp_path), [0, 1])

    with pytest.raises(FrozenInstanceError):
        plan.movement_enabled = True
