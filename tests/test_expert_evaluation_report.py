import json

import pytest

from navigation.expert_evaluation_plan import diagnostic_scenarios
from navigation.expert_evaluation_report import (
    assess_episode,
    goal_distance,
    max_cross_track_error,
    path_length,
    plot_planned_versus_executed,
    save_episode,
    summarize_benchmark,
)


def scenario(name):
    return next(
        item for item in diagnostic_scenarios()
        if item.scenario_id == name
    )


def test_path_metrics_use_recorded_points_only():
    assert path_length(None) is None
    assert path_length([]) is None
    assert path_length([(0.0, 0.0)]) == 0.0
    assert path_length([(0.0, 0.0), (3.0, 4.0)]) == pytest.approx(5.0)
    assert goal_distance([(1.0, 1.0, 0.0)], (1.0, 2.0)) == pytest.approx(1.0)
    assert max_cross_track_error(
        [(0.0, 0.0), (2.0, 0.0)],
        [(0.0, 0.0, 0.0), (1.0, 0.4, 0.0)],
    ) == pytest.approx(0.4)


def test_goal_success_requires_tracking_and_a_zero_stop():
    measured = {
        "status": "GOAL_REACHED",
        "goal_distance_m": 0.1,
        "max_cross_track_error_m": 0.2,
        "stop_verified": True,
    }

    passed, explanation = assess_episode(scenario("low_speed_single"), measured)

    assert passed
    assert explanation == ""


def test_goal_report_without_a_stop_is_not_success():
    passed, explanation = assess_episode(
        scenario("low_speed_single"),
        {
            "status": "GOAL_REACHED",
            "goal_distance_m": 0.1,
            "max_cross_track_error_m": 0.1,
            "stop_verified": False,
        },
    )

    assert not passed
    assert "zero" in explanation


def test_large_deviation_is_explained():
    passed, explanation = assess_episode(
        scenario("reachable_north"),
        {
            "status": "GOAL_REACHED",
            "goal_distance_m": 0.1,
            "max_cross_track_error_m": 1.5,
            "stop_verified": True,
        },
    )

    assert not passed
    assert "corridor" in explanation


def test_expected_planning_failure_passes_only_with_a_stop():
    passed, _ = assess_episode(
        scenario("unreachable_goal"),
        {
            "status": "PLANNING_FAILED",
            "goal_distance_m": None,
            "max_cross_track_error_m": None,
            "stop_verified": True,
        },
    )

    assert passed


def test_empty_benchmark_does_not_invent_rates():
    summary = summarize_benchmark([])

    assert summary["episodes_recorded"] == 0
    assert summary["success_rate"] is None
    assert summary["collision_count"] is None
    assert summary["timeout_rate"] is None
    assert summary["mean_path_length_m"] is None
    assert summary["mean_navigation_time_s"] is None


def test_benchmark_counts_real_outcomes():
    records = [
        {
            "scenario_id": "bench_00",
            "status": "GOAL_REACHED",
            "passed": True,
            "explanation": "",
            "path_length_m": 2.0,
            "navigation_time_s": 10.0,
        },
        {
            "scenario_id": "bench_01",
            "status": "SAFETY_STOP",
            "passed": False,
            "explanation": "Collision monitoring stopped the robot before another command.",
            "path_length_m": 0.4,
            "navigation_time_s": 3.0,
        },
        {
            "scenario_id": "bench_02",
            "status": "TIME_LIMIT",
            "passed": False,
            "explanation": "The episode wall-clock limit expired before the goal.",
            "path_length_m": None,
            "navigation_time_s": 45.0,
        },
    ]

    summary = summarize_benchmark(records)

    assert summary["episodes_recorded"] == 3
    assert summary["success_rate"] == pytest.approx(1 / 3)
    assert summary["collision_count"] == 1
    assert summary["timeout_rate"] == pytest.approx(1 / 3)
    assert summary["mean_path_length_m"] == pytest.approx(1.2)
    assert summary["mean_navigation_time_s"] == pytest.approx(58 / 3)
    assert [item["scenario_id"] for item in summary["failures"]] == [
        "bench_01",
        "bench_02",
    ]


def test_plot_and_record_are_not_overwritten(tmp_path):
    record = {
        "scenario_id": "low_speed_single",
        "status": "GOAL_REACHED",
        "start": [0.0, -1.5, 0.0],
        "goal": [2.5, -1.5],
        "planned_path": [[0.0, -1.5], [2.5, -1.5]],
        "executed_poses": [[0.0, -1.5, 0.0], [2.5, -1.5, 0.0]],
    }
    plot_path = tmp_path / "episode_low_speed_single.png"
    plot_planned_versus_executed(record, plot_path, (1.5, 0.0, 0.5, 0.5))
    original = plot_path.read_bytes()

    with pytest.raises(FileExistsError):
        plot_planned_versus_executed(record, plot_path, (1.5, 0.0, 0.5, 0.5))

    assert plot_path.read_bytes() == original

    save_episode(record, tmp_path)
    saved = (tmp_path / "episode_low_speed_single.json").read_text()

    with pytest.raises(FileExistsError):
        save_episode(record, tmp_path)

    assert (tmp_path / "episode_low_speed_single.json").read_text() == saved
    assert json.loads(saved)["status"] == "GOAL_REACHED"
