import json

from navigation import evaluate_expert_navigation as evaluation
from tests.test_expert_evaluation_runner import FakeRunner, probe


def world_file(tmp_path):
    path = tmp_path / "jackal_obstacle_test_collision.sdf"
    path.write_text("<sdf version='1.7'></sdf>")
    return path


def argv(tmp_path, mode, suite="diagnostics"):
    return [
        "--mode", mode,
        "--suite", suite,
        "--world", str(world_file(tmp_path)),
        "--output-dir", str(tmp_path / "out"),
    ]


def test_preview_prints_plan_without_creating_output(tmp_path, capsys):
    code = evaluation.main(argv(tmp_path, "preview"))
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["movement_enabled"] is False
    assert payload["simulation_reset_enabled"] is False
    assert payload["episodes"] == 7
    assert not (tmp_path / "out").exists()


def test_preview_rejects_a_different_world(tmp_path):
    other = tmp_path / "other.sdf"
    other.write_text("<sdf></sdf>")

    try:
        evaluation.parse_args([
            "--world", str(other),
        ])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("A different world must be rejected")


def test_execute_is_blocked_before_a_runner(monkeypatch, tmp_path, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Runner must not be created")

    monkeypatch.setattr(
        "navigation.collect_expert_demonstrations.run_collection_preflight",
        lambda: False,
    )
    monkeypatch.setattr(
        "navigation.expert_live_runner_factory.create_live_runner",
        forbidden,
    )

    code = evaluation.main(argv(tmp_path, "execute"))
    output = capsys.readouterr().out

    assert code == 1
    assert "EXECUTE: BLOCKED" in output
    assert "No robot commands or simulation resets were issued." in output
    assert not (tmp_path / "out").exists()


def test_execute_persists_real_fake_runner_outcomes(monkeypatch, tmp_path, capsys):
    created = {}

    def preflight():
        print("PREFLIGHT: PASS")
        return True

    def factory(_scenario):
        runner = FakeRunner()
        created["runner"] = runner
        return runner

    monkeypatch.setattr(
        "navigation.collect_expert_demonstrations.run_collection_preflight",
        preflight,
    )
    monkeypatch.setattr(
        evaluation,
        "_publisher_count",
        lambda: 0,
    )
    monkeypatch.setattr(evaluation, "_collision_probe", probe)

    code = evaluation.execute(
        evaluation.parse_args(argv(tmp_path, "execute", "benchmark")),
        runner_factory=factory,
    )
    captured = capsys.readouterr()
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())

    assert code == 0
    assert created["runner"].closed
    assert created["runner"].robot.stopped == 20
    assert summary["live_executed"] is True
    assert summary["episodes_recorded"] == 20
    assert summary["success_rate"] == 1.0
    assert summary["collision_count"] == 0
    assert summary["timeout_rate"] == 0.0
    assert summary["publisher_count_before"] == 0
    assert summary["publisher_count_after"] == 0
    assert summary["stale_publisher"] is False
    assert summary["mean_path_length_m"] > 0
    assert "EXECUTE: PASS" in captured.out
    assert (tmp_path / "out" / "episode_bench_00_low_speed_single.json").is_file()
    assert (tmp_path / "out" / "episode_bench_02_reachable_around.png").is_file()


def test_nonzero_publisher_blocks_before_motion(monkeypatch, tmp_path, capsys):
    def factory(_scenario):
        raise AssertionError("Runner must not be created")

    monkeypatch.setattr(
        "navigation.collect_expert_demonstrations.run_collection_preflight",
        lambda: True,
    )
    monkeypatch.setattr(evaluation, "_publisher_count", lambda: 1)

    code = evaluation.execute(
        evaluation.parse_args(argv(tmp_path, "execute")),
        runner_factory=factory,
    )

    assert code == 1
    assert "EXECUTE: BLOCKED" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()
