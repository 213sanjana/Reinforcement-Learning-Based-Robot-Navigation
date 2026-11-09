"""Path-efficiency gate tests. No simulator, checkpoint, or PPO update."""

import json
import logging

import pytest

from curriculum.jackal_curriculum import JackalCurriculum, path_efficiency, stages


def _batch(level=0, **overrides):
    pool = stages()[level]
    rows = []
    for index in range(12):
        row = dict(
            scenario_id=pool[index % len(pool)].scenario_id,
            status="completed",
            success=True,
            collision=False,
            start_goal_distance_m=1.0,
            final_goal_distance_m=0.0,
            path_length_odom_m=2.0,
            odom_coverage_complete=True,
        )
        row.update(overrides)
        rows.append(row)
    return rows


def _ready(level=0, **overrides):
    curriculum = JackalCurriculum(efficiency_enabled=True, efficiency_threshold=0.5)
    curriculum.charge(2048)
    return curriculum, _batch(level, **overrides)


def test_path_efficiency_rejects_invalid_lengths_without_dividing():
    assert path_efficiency(1.0, 0.0, 2.0) == (0.5, None)
    for value, reason in (
        (None, "invalid_path_length"),
        (float("nan"), "invalid_path_length"),
        (float("inf"), "invalid_path_length"),
        (0.0, "nonpositive_path_length"),
        (-1.0, "nonpositive_path_length"),
    ):
        efficiency, rejection = path_efficiency(1.0, 0.0, value)
        assert efficiency is None
        assert rejection == reason
    efficiency, rejection = path_efficiency(0.3, 1.0, 1.0)
    assert rejection is None
    assert efficiency == pytest.approx(-0.7)


def test_inefficient_success_is_rejected_and_efficient_success_can_pass():
    curriculum, inefficient = _ready(path_length_odom_m=3.4, start_goal_distance_m=0.75, final_goal_distance_m=0.30)
    assert curriculum.observe(inefficient) == "retain"
    assert curriculum.passes == 0
    assert curriculum.history[-1]["promotion"] == "rejected"
    assert curriculum.history[-1]["efficiency_episodes"][0]["efficiency_gate"] == "fail"
    curriculum.charge(2048)
    assert curriculum.observe(_batch()) == "retain"
    assert curriculum.passes == 1
    curriculum.charge(2048)
    assert curriculum.observe(_batch()) == "promote"
    assert curriculum.level == 1
    assert curriculum.history[-1]["promotion"] == "accepted"
    assert curriculum.history[-1]["efficiency_pass"] is True


def test_threshold_boundary_and_two_pass_reset():
    exact, rows = _ready()
    assert exact.observe(rows) == "retain"
    assert exact.passes == 1
    below = _batch(path_length_odom_m=2.01)
    exact.charge(2048)
    assert exact.observe(below) == "retain"
    assert exact.passes == 0
    assert exact.history[-1]["promotion"] == "rejected"
    exact.charge(2048)
    assert exact.observe(_batch()) == "retain"
    exact.charge(2048)
    assert exact.observe(_batch()) == "promote"


def test_collision_and_failure_still_block_promotion():
    collided, rows = _ready(collision=True)
    assert collided.observe(rows) == "retain"
    collided.charge(2048)
    assert collided.observe(_batch(collision=True)) == "retain"
    assert collided.level == 0 and collided.passes == 0
    assert collided.history[-1]["promotion"] == "not_applied"
    failed, failed_rows = _ready(success=False)
    assert failed.observe(failed_rows) == "retain"
    failed.charge(2048)
    assert failed.observe(_batch(success=False)) == "retain"
    assert failed.level == 0 and failed.passes == 0


def test_invalid_telemetry_cannot_pass_and_disabled_gate_keeps_old_behavior():
    invalid, rows = _ready(path_length_odom_m=float("nan"))
    assert invalid.observe(rows) == "retain"
    assert invalid.passes == 0
    assert invalid.history[-1]["efficiency_episodes"][0]["rejection"] == "invalid_path_length"
    for bad in (0.0, float("inf"), -0.2, None):
        curriculum, batch = _ready(path_length_odom_m=bad)
        assert curriculum.observe(batch) == "retain"
        assert curriculum.passes == 0
    disabled = JackalCurriculum()
    disabled.charge(2048)
    assert disabled.observe(_batch(path_length_odom_m=float("nan"), odom_coverage_complete=False)) == "retain"
    disabled.charge(2048)
    assert disabled.observe(_batch(path_length_odom_m=0.0)) == "promote"
    assert disabled.history[-1]["efficiency_gate"] is False
    assert disabled.history[-1]["efficiency_pass"] is None
    with pytest.raises(ValueError, match="development"):
        disabled.observe(_batch(), source="fixed")


def test_efficiency_state_round_trip_and_bad_threshold():
    curriculum, rows = _ready()
    curriculum.observe(rows)
    restored = JackalCurriculum(**json.loads(json.dumps(curriculum.state_dict())))
    assert restored == curriculum
    with pytest.raises(ValueError):
        JackalCurriculum(efficiency_enabled=True, efficiency_threshold=float("nan"))
    with pytest.raises(ValueError):
        JackalCurriculum(efficiency_enabled=True, efficiency_threshold=0)


def test_efficiency_contract_covers_success_failure_and_invalid_telemetry():
    ready = JackalCurriculum(efficiency_enabled=True, efficiency_threshold=0.5)
    ready.charge(2048)
    assert ready.observe(_batch()) == "retain"
    assert ready.passes == 1
    assert ready.history[-1]["efficiency_pass"] is True
    ready.charge(2048)
    assert ready.observe(_batch()) == "promote"
    assert ready.level == 1
    assert ready.history[-1]["promotion"] == "accepted"

    one_short = _batch()
    one_short[0]["path_length_odom_m"] = 10.0
    below = JackalCurriculum(efficiency_enabled=True, efficiency_threshold=0.5)
    below.charge(2048)
    assert below.observe(one_short) == "retain"
    assert below.passes == 0
    assert below.history[-1]["success_rate"] == pytest.approx(1.0)
    assert below.history[-1]["promotion"] == "rejected"
    assert below.history[-1]["efficiency_episodes"][0]["rejection"] == "below_threshold"
    assert below.history[-1]["efficiency_episodes"][1]["efficiency_gate"] == "pass"

    invalid = _batch()
    invalid[4]["path_length_odom_m"] = float("nan")
    blocked = JackalCurriculum(efficiency_enabled=True, efficiency_threshold=0.5)
    blocked.charge(2048)
    assert blocked.observe(invalid) == "retain"
    assert blocked.passes == 0
    assert blocked.history[-1]["success_rate"] == pytest.approx(1.0)
    assert blocked.history[-1]["efficiency_pass"] is False
    assert blocked.history[-1]["promotion"] == "rejected"
    assert blocked.history[-1]["efficiency_episodes"][4]["rejection"] == "invalid_path_length"

    low_failures = _batch()
    for index in (0, 1):
        low_failures[index]["success"] = False
        low_failures[index]["path_length_odom_m"] = 20.0
        low_failures[index]["final_goal_distance_m"] = 0.9
    tolerated = JackalCurriculum(efficiency_enabled=True, efficiency_threshold=0.5)
    tolerated.charge(2048)
    assert tolerated.observe(low_failures) == "retain"
    assert tolerated.passes == 1
    assert tolerated.history[-1]["success_rate"] == pytest.approx(10 / 12)
    assert tolerated.history[-1]["collision_rate"] == pytest.approx(0.0)
    assert tolerated.history[-1]["efficiency_pass"] is True
    assert tolerated.history[-1]["efficiency_episodes"][0]["efficiency_gate"] == "pass"
    assert tolerated.history[-1]["efficiency_episodes"][0]["rejection"] is None
    assert tolerated.history[-1]["promotion"] == "not_applied"
    tolerated.charge(2048)
    assert tolerated.observe(low_failures) == "promote"
    assert tolerated.history[-1]["promotion"] == "accepted"

    collided = JackalCurriculum(efficiency_enabled=True, efficiency_threshold=0.5)
    collided.charge(2048)
    assert collided.observe(_batch(collision=True)) == "retain"
    assert collided.passes == 0
    assert collided.history[-1]["promotion"] == "not_applied"

    failed = JackalCurriculum(efficiency_enabled=True, efficiency_threshold=0.5)
    failed.charge(2048)
    assert failed.observe(_batch(success=False)) == "retain"
    assert failed.passes == 0
    assert failed.failures == 1
    assert failed.history[-1]["efficiency_pass"] is True
    assert failed.history[-1]["promotion"] == "not_applied"

    disabled = JackalCurriculum()
    disabled.charge(2048)
    assert disabled.observe(_batch(path_length_odom_m=float("nan"))) == "retain"
    disabled.charge(2048)
    assert disabled.observe(_batch(path_length_odom_m=0.0)) == "promote"
    assert disabled.history[-1]["efficiency_gate"] is False
    assert disabled.history[-1]["efficiency_pass"] is None

    for bad, reason in (
        (0.0, "nonpositive_path_length"),
        (-1.0, "nonpositive_path_length"),
        (float("nan"), "invalid_path_length"),
        (float("inf"), "invalid_path_length"),
    ):
        curriculum = JackalCurriculum(efficiency_enabled=True, efficiency_threshold=0.5)
        curriculum.charge(2048)
        batch = _batch()
        batch[2]["path_length_odom_m"] = bad
        assert curriculum.observe(batch) == "retain"
        assert curriculum.passes == 0
        assert curriculum.history[-1]["efficiency_pass"] is False
        assert curriculum.history[-1]["efficiency_episodes"][2]["rejection"] == reason


def test_decision_log_names_pass_fail_and_promotion(caplog):
    caplog.set_level(logging.INFO)
    curriculum, rows = _ready(path_length_odom_m=3.4, start_goal_distance_m=0.75, final_goal_distance_m=0.30)
    curriculum.observe(rows)
    assert "efficiency_gate=fail" in caplog.text
    assert "promotion=rejected" in caplog.text
    assert "path_efficiency=" in caplog.text
