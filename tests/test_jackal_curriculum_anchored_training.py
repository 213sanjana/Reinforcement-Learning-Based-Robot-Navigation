"""Configuration and scripted-evaluator tests.

Rejected-resume and illegal-output tests call ``run`` only up to the
pre-write gate. They do not call ``learn``, start ROS 2 or Gazebo, or load
the Phase 6G.1 checkpoint.
"""

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import gymnasium as gym

from navigation.expert_demonstration_schema import jackal_policy_spaces
from navigation import jackal_ppo_diagnostic as diagnostic
from navigation.jackal_curriculum_train import (
    ANCHORED_OUTPUT_NAMESPACE,
    load_config,
    run,
    validate_anchored_output,
)
from navigation.jackal_ppo_diagnostic import (
    OFFICIAL_CHECKPOINT,
    OFFICIAL_CHECKPOINT_SHA256,
    _evaluate_episode,
)


ROOT = Path(__file__).resolve().parents[1]
ROOT_CONFIG = ROOT / "configs/jackal_curriculum_bc_anchored.yaml"
HISTORICAL_CONFIG = ROOT / "configs/jackal_curriculum_6h.yaml"


def test_anchored_config_names_the_verified_checkpoint_and_gates():
    config = load_config(ROOT_CONFIG)
    historical = load_config(HISTORICAL_CONFIG)
    assert config["phase"] == "6h_bc_anchored"
    assert config["seed"] == historical["seed"] == 642
    assert config["total_timesteps"] == historical["total_timesteps"] == 10240
    assert config["required_passes"] == historical["required_passes"] == 2
    assert config["min_level_steps"] == historical["min_level_steps"] == 2048
    assert config["actor_anchoring"]["enabled"] is True
    assert config["actor_anchoring"]["coefficient"] == pytest.approx(0.1)
    assert config["actor_anchoring"]["checkpoint"].endswith(
        "artifacts/ppo/jackal_6g1/run_20261022_230534/initialized_ppo.zip"
    )
    assert config["actor_anchoring"]["expected_sha256"] == OFFICIAL_CHECKPOINT_SHA256
    assert OFFICIAL_CHECKPOINT.name == "initialized_ppo.zip"
    assert config["efficiency_gate"]["enabled"] is True
    assert config["efficiency_gate"]["threshold"] == pytest.approx(0.5)
    assert config["efficiency_gate"]["definition"] == "net_goal_progress_over_path_length"
    assert "actor_anchoring" not in historical
    assert "efficiency_gate" not in historical
    assert historical["output_root"] == "artifacts/ppo/jackal_6h/curriculum"
    assert config["output_root"] == "artifacts/ppo/jackal_6h_bc_anchored/curriculum"
    assert config["output_root"] != historical["output_root"]


def test_config_rejects_bad_anchor_and_historical_overrides(tmp_path):
    text = ROOT_CONFIG.read_text(encoding="utf-8")
    bad_hash = tmp_path / "hash.yaml"
    bad_hash.write_text(text.replace(OFFICIAL_CHECKPOINT_SHA256, "0" * 64))
    with pytest.raises(ValueError, match="hash"):
        load_config(bad_hash)
    bad_definition = tmp_path / "definition.yaml"
    bad_definition.write_text(text.replace("net_goal_progress_over_path_length", "snapshot_ratio"))
    with pytest.raises(ValueError, match="definition"):
        load_config(bad_definition)
    historical = HISTORICAL_CONFIG.read_text(encoding="utf-8")
    tainted = tmp_path / "historical.yaml"
    tainted.write_text(historical + "\nactor_anchoring:\n  enabled: true\n")
    with pytest.raises(ValueError, match="Historical phase 6h"):
        load_config(tainted)


class _Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _Odom:
    def __init__(self, x, y):
        self.pose = SimpleNamespace(pose=SimpleNamespace(position=_Position(x, y)))


class _Robot:
    def __init__(self, x, y):
        self.odom = _Odom(x, y)
        self.stops = 0

    def stop(self):
        self.stops += 1


class _DistanceEnv(gym.Env):
    def __init__(self, start_distance=0.75, reported_start=0.75, final_distance=0.30):
        super().__init__()
        self.observation_space, self.action_space = jackal_policy_spaces()
        self.max_episode_steps = 300
        self.robot = _Robot(0.0, 0.0)
        self.previous_goal_distance = None
        self.start_distance = start_distance
        self.reported_start = reported_start
        self.final_distance = final_distance
        self.goal_odom = (self.start_distance, 0.0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.robot.odom = _Odom(0.0, 0.0)
        self.previous_goal_distance = self.reported_start
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        return observation, {"goal_odom": self.goal_odom}

    def step(self, action):
        moved = self.start_distance - self.final_distance
        self.robot.odom = _Odom(moved, 0.0)
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "goal_reached": True,
            "collision_detected": False,
            "time_limit_reached": False,
            "termination_reason": "goal",
            "current_distance": self.final_distance,
        }
        return observation, 1.0, True, False, info


class _Predict:
    def predict(self, observation, deterministic=True):
        return np.zeros(2, dtype=np.float32), None


def _scenario():
    return SimpleNamespace(
        scenario_id="A_0",
        category="development",
        start=(0.0, -1.8, 0.0),
        goal=(0.75, -1.8),
    )


def test_scripted_evaluator_records_consistent_goal_distances():
    episode = _evaluate_episode(_Predict(), _DistanceEnv(), _scenario(), 642, 0, True)
    assert episode["status"] == "completed"
    assert episode["start_goal_distance_m"] == pytest.approx(0.75)
    assert episode["final_goal_distance_m"] == pytest.approx(0.30)
    assert episode["goal_distance_frame"] == "odom"
    assert episode["goal_distance_consistent"] is True
    assert episode["odom_point_count"] == 2
    assert episode["odom_coverage_complete"] is True
    assert episode["path_length_odom_m"] == pytest.approx(0.45)
    assert episode["net_goal_progress_m"] == pytest.approx(0.45)
    assert episode["path_efficiency"] == pytest.approx(1.0)
    inconsistent = _evaluate_episode(
        _Predict(),
        _DistanceEnv(reported_start=1.2),
        _scenario(),
        642,
        0,
        True,
    )
    assert inconsistent["start_goal_distance_m"] is None
    assert inconsistent["goal_distance_consistent"] is False
    assert inconsistent["odom_coverage_complete"] is False
    assert inconsistent["path_efficiency"] is None
    assert inconsistent["path_length_odom_m"] == pytest.approx(0.45)


def _snapshot(root):
    root = Path(root)
    if not root.exists():
        return None
    records = {".": ("dir", root.stat().st_mtime_ns)}
    for path in sorted(root.rglob("*")):
        stat = path.lstat()
        relative = str(path.relative_to(root))
        if path.is_symlink():
            records[relative] = ("symlink", str(path.readlink()), stat.st_mtime_ns)
        elif path.is_dir():
            records[relative] = ("dir", stat.st_mtime_ns)
        elif path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records[relative] = ("file", stat.st_size, stat.st_mtime_ns, digest)
        else:
            records[relative] = ("other", stat.st_mtime_ns)
    return records


def _direct_signature(root):
    root = Path(root)
    rows = {".": ("dir", root.stat().st_mtime_ns)}
    for path in sorted(root.iterdir()):
        stat = path.lstat()
        if path.is_file() and not path.is_symlink():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows[path.name] = ("file", stat.st_size, stat.st_mtime_ns, digest)
        else:
            rows[path.name] = ("entry", stat.st_mtime_ns)
    return rows


def _forbid_execution(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("rejected path or resume continued into execution")

    monkeypatch.setattr(diagnostic, "verify_source_hashes", boom)
    monkeypatch.setattr(diagnostic, "create_model", boom)
    monkeypatch.setattr(diagnostic, "preflight_simulator", boom)


def _remove_empty_created(path, created):
    if not created:
        return
    current = Path(path)
    while current != ROOT and current.exists() and not any(current.iterdir()):
        current.rmdir()
        current = current.parent


def test_anchored_output_rejects_historical_ancestor_descendant_and_unrelated(tmp_path, monkeypatch):
    _forbid_execution(monkeypatch)
    config = load_config(ROOT_CONFIG)
    namespace = ANCHORED_OUTPUT_NAMESPACE
    historical_root = ROOT / "artifacts" / "ppo" / "jackal_6h"
    historical_curriculum = historical_root / "curriculum"
    descendant = historical_curriculum / "audit_must_not_create"
    ancestor = ROOT / "artifacts" / "ppo"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("same")
    protected = (
        diagnostic.OFFICIAL_CHECKPOINT.parent,
        ROOT / "artifacts" / "behavior_cloning",
        ROOT / "artifacts" / "ppo" / "jackal_6g2",
    )
    cases = (
        (historical_root, "historical root"),
        (historical_curriculum, "historical descendant"),
        (descendant, "historical descendant"),
        (ancestor, "historical ancestor"),
        (namespace / ".." / ".." / "jackal_6h" / "curriculum" / "nested", "historical descendant"),
        (unrelated, "outside"),
        (namespace, "outside"),
    )
    for path, match in cases:
        with pytest.raises(ValueError, match=match):
            validate_anchored_output(namespace, path)
    for path in protected:
        with pytest.raises(ValueError, match="historical"):
            validate_anchored_output(namespace, path)
    validate_anchored_output(namespace, namespace / "run_ok")

    text = ROOT_CONFIG.read_text(encoding="utf-8")
    original = "output_root: artifacts/ppo/jackal_6h_bc_anchored/curriculum"
    for replacement, match in (
        ("output_root: artifacts/ppo/jackal_6h/curriculum", "historical descendant"),
        ("output_root: artifacts/ppo/jackal_6h", "historical root"),
        ("output_root: artifacts/ppo", "historical ancestor"),
        ("output_root: artifacts/behavior_cloning", "historical root"),
        ("output_root: /tmp/not-anchored", "outside"),
    ):
        path = tmp_path / f"{match.replace(' ', '_')}.yaml"
        path.write_text(text.replace(original, replacement))
        with pytest.raises(ValueError, match=match):
            load_config(path)

    assert not descendant.exists()
    before_root = _direct_signature(historical_root)
    before_curriculum = _direct_signature(historical_curriculum)
    before_ancestor = _direct_signature(ancestor)
    before_unrelated = _snapshot(unrelated)
    with pytest.raises(ValueError, match="historical root"):
        run(config, historical_root, mode="pilot", resume=True, environment_factory=lambda: pytest.fail("env"))
    with pytest.raises(ValueError, match="historical descendant"):
        run(config, descendant, mode="pilot", environment_factory=lambda: pytest.fail("env"))
    with pytest.raises(ValueError, match="historical ancestor"):
        run(config, ancestor, mode="pilot", environment_factory=lambda: pytest.fail("env"))
    with pytest.raises(ValueError, match="outside"):
        run(config, unrelated, mode="pilot", resume=True, environment_factory=lambda: pytest.fail("env"))
    assert not descendant.exists()
    assert _direct_signature(historical_root) == before_root
    assert _direct_signature(historical_curriculum) == before_curriculum
    assert _direct_signature(ancestor) == before_ancestor
    assert _snapshot(unrelated) == before_unrelated


def test_rejected_resume_writes_nothing(monkeypatch):
    _forbid_execution(monkeypatch)
    config = load_config(ROOT_CONFIG)
    namespace = ANCHORED_OUTPUT_NAMESPACE
    created_namespace = not namespace.exists()
    target = namespace / "audit_rejected_resume_zero_write"
    if target.exists():
        raise AssertionError(f"refusing to reuse unexpected directory {target}")
    namespace.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    try:
        (target / "marker.txt").write_text("audit-zero-write")
        (target / "summary.json").write_text('{"status": "previous"}\n')
        (target / "latest.json").write_text(json.dumps({
            "config_sha256": "0" * 64,
            "checkpoint": "does-not-matter.zip",
            "sha256": "1" * 64,
        }))
        before = _snapshot(target)
        result = run(
            config, target, mode="pilot", resume=True,
            environment_factory=lambda: pytest.fail("env"),
        )
        assert result["status"] == "failed"
        assert "mismatch" in result["error"]["message"]
        assert result["checkpoint"] is None
        assert _snapshot(target) == before
        assert not (target / "failures.jsonl").exists()
    finally:
        shutil.rmtree(target)
        _remove_empty_created(namespace, created_namespace)
