"""Fail-closed execute path and applied-action persistence."""

import json
from types import SimpleNamespace

import numpy as np

from navigation import collect_expert_demonstrations as collector
from navigation import expert_collection_preflight as graph
from navigation import expert_runtime_preflight as runtime
def preflight_result(passed):
    return SimpleNamespace(
        passed=passed,
        checks={"test_check": passed},
    )


def write_world(tmp_path):
    world = tmp_path / "world.sdf"
    world.write_text("<sdf version='1.7'></sdf>")
    return world


def execute_argv(tmp_path, episodes="1"):
    world = write_world(tmp_path)
    return [
        "--mode",
        "execute",
        "--world",
        str(world),
        "--output-dir",
        str(tmp_path / "output"),
        "--episodes",
        episodes,
        "--inflation-radius",
        "0.45",
        "--duration",
        "5",
        "--start",
        "0",
        "0",
        "0",
        "--goal",
        "3",
        "0",
    ]


def pass_preflight(monkeypatch):
    monkeypatch.setattr(
        graph, "run_preflight", lambda: preflight_result(True)
    )
    monkeypatch.setattr(
        runtime,
        "run_runtime_preflight",
        lambda timeout=5.0: preflight_result(True),
    )


def make_result(
    status="GOAL_REACHED",
    rejected=0,
    applied=0.4,
    requested=0.5,
):
    transition = SimpleNamespace(
        observation=np.zeros(40, dtype=np.float32),
        action=np.array([requested, 0.0], dtype=np.float32),
        applied_action=np.array([applied, 0.0], dtype=np.float32),
        published_linear=0.1,
        published_angular=0.0,
        simulation_time_ns=1000,
        pose=(0.0, 0.0, 0.0),
    )

    return SimpleNamespace(
        status=status,
        transitions=[transition],
        accepted_commands=1,
        rejected_commands=rejected,
        planned_path=[(0.0, 0.0), (1.0, 0.0)],
        executed_poses=[(0.0, 0.0, 0.0), (0.4, 0.0, 0.0)],
    )


class FakeRunner:
    def __init__(self, result=None, events=None, prepare_error=None):
        self.result = make_result() if result is None else result
        self.events = events if events is not None else []
        self.prepare_error = prepare_error
        self.closed = False

    def prepare(self, start, goal):
        self.events.append(("prepare", tuple(start), tuple(goal)))
        if self.prepare_error is not None:
            raise self.prepare_error

    def run(self, duration):
        self.events.append(("run", duration))
        return self.result

    def close(self):
        self.events.append("close")
        self.closed = True


def summary_from_output(output):
    start = output.index("{")
    end = output.rindex("}") + 1
    return json.loads(output[start:end])


def test_graph_exception_blocks_before_runner(monkeypatch, tmp_path, capsys):
    def explode():
        raise RuntimeError("Controller manager unavailable")

    def forbidden(*args, **kwargs):
        raise AssertionError("Runner must not be created")

    import navigation.expert_live_runner_factory as factory

    monkeypatch.setattr(graph, "run_preflight", explode)
    monkeypatch.setattr(runtime, "run_runtime_preflight", forbidden)
    monkeypatch.setattr(factory, "create_live_runner", forbidden)

    code = collector.main(execute_argv(tmp_path))
    captured = capsys.readouterr()

    assert code == 1
    assert "ROS GRAPH PREFLIGHT ERROR" in captured.err
    assert "EXECUTE: BLOCKED" in captured.out
    assert "EXECUTE: PASS" not in captured.out
    assert not (tmp_path / "output").exists()


def test_runtime_exception_blocks_before_runner(monkeypatch, tmp_path, capsys):
    def explode(timeout=5.0):
        raise RuntimeError("Sensor stream unavailable")

    def forbidden(*args, **kwargs):
        raise AssertionError("Runner must not be created")

    import navigation.expert_live_runner_factory as factory

    monkeypatch.setattr(graph, "run_preflight", lambda: preflight_result(True))
    monkeypatch.setattr(runtime, "run_runtime_preflight", explode)
    monkeypatch.setattr(factory, "create_live_runner", forbidden)

    code = collector.main(execute_argv(tmp_path))
    captured = capsys.readouterr()

    assert code == 1
    assert "RUNTIME PREFLIGHT ERROR" in captured.err
    assert "EXECUTE: BLOCKED" in captured.out
    assert not (tmp_path / "output").exists()


def test_successful_execute_persists_applied_actions(
    monkeypatch, tmp_path, capsys
):
    events = []
    authorizations = []

    import navigation.expert_collection_orchestrator as orchestrator
    import navigation.expert_dataset_export as exporter
    import navigation.expert_live_runner_factory as factory

    real_collect = orchestrator.collect_plan
    real_export = exporter.export_expert_episode

    def spy_collect(*args, **kwargs):
        authorizations.append(kwargs["allow_execution"])
        return real_collect(*args, **kwargs)

    def spy_export(result, output_dir, episode_id):
        events.append("export")
        return real_export(result, output_dir, episode_id)

    def make_runner(episode, **kwargs):
        events.append(
            (
                "factory",
                kwargs["allow_live"],
                kwargs["inflation_radius"],
                str(kwargs["world_path"]),
            )
        )
        return FakeRunner(events=events)

    pass_preflight(monkeypatch)
    monkeypatch.setattr(orchestrator, "collect_plan", spy_collect)
    monkeypatch.setattr(exporter, "export_expert_episode", spy_export)
    monkeypatch.setattr(factory, "create_live_runner", make_runner)

    code = collector.main(execute_argv(tmp_path))
    captured = capsys.readouterr()

    assert code == 0
    assert authorizations == [True]
    assert events[0][1] is True
    assert events[0][2] == 0.45
    assert "close" in events
    assert events.index("close") < events.index("export")

    output_dir = tmp_path / "output"
    npz_path = output_dir / "expert_episode_0000.npz"
    json_path = output_dir / "expert_episode_0000.json"
    assert npz_path.is_file()
    assert json_path.is_file()

    dataset = np.load(npz_path)
    assert dataset["applied_actions"].shape == (1, 2)
    assert np.allclose(dataset["applied_actions"], [[0.4, 0.0]])
    assert np.allclose(dataset["requested_actions"], [[0.5, 0.0]])

    metadata = json.loads(json_path.read_text())
    assert metadata["training_action_key"] == "applied_actions"
    assert metadata["status"] == "GOAL_REACHED"
    assert metadata["planned_path"] == [[0.0, 0.0], [1.0, 0.0]]

    summary = summary_from_output(captured.out)
    assert summary["execution_authorized"] is True
    assert summary["outcomes"][0]["exported"] is True
    assert "EXECUTE: PASS" in captured.out


def test_second_execute_does_not_overwrite(monkeypatch, tmp_path, capsys):
    import navigation.expert_live_runner_factory as factory

    pass_preflight(monkeypatch)
    monkeypatch.setattr(
        factory,
        "create_live_runner",
        lambda *args, **kwargs: FakeRunner(),
    )

    assert collector.main(execute_argv(tmp_path)) == 0
    original = (tmp_path / "output" / "expert_episode_0000.npz").read_bytes()

    capsys.readouterr()
    assert collector.main(execute_argv(tmp_path)) == 0

    assert (tmp_path / "output" / "expert_episode_0000.npz").read_bytes() == original
    assert (tmp_path / "output" / "expert_episode_0001.npz").is_file()
    assert (tmp_path / "output" / "expert_episode_0001.json").is_file()


def test_collision_status_persists_nothing(monkeypatch, tmp_path, capsys):
    import navigation.expert_live_runner_factory as factory

    pass_preflight(monkeypatch)
    monkeypatch.setattr(
        factory,
        "create_live_runner",
        lambda *args, **kwargs: FakeRunner(
            result=make_result(status="SAFETY_STOP")
        ),
    )

    code = collector.main(execute_argv(tmp_path, episodes="2"))
    captured = capsys.readouterr()
    summary = summary_from_output(captured.out)

    assert code == 1
    assert "EXECUTE: FAIL" in captured.out
    assert summary["episodes_requested"] == 2
    assert summary["episodes_completed"] == 1
    assert summary["outcomes"][0]["status"] == "SAFETY_STOP"
    assert summary["outcomes"][0]["exported"] is False
    output = tmp_path / "output"
    assert not output.exists() or list(output.iterdir()) == []


def test_rejected_commands_are_not_persisted(monkeypatch, tmp_path, capsys):
    import navigation.expert_live_runner_factory as factory

    pass_preflight(monkeypatch)
    monkeypatch.setattr(
        factory,
        "create_live_runner",
        lambda *args, **kwargs: FakeRunner(
            result=make_result(rejected=1)
        ),
    )

    code = collector.main(execute_argv(tmp_path))
    summary = summary_from_output(capsys.readouterr().out)

    assert code == 1
    assert summary["outcomes"][0]["status"] == "REJECTED_COMMANDS"
    assert summary["outcomes"][0]["exported"] is False
    assert not (tmp_path / "output").exists()


def test_runner_error_persists_nothing_and_closes(monkeypatch, tmp_path, capsys):
    runner = FakeRunner(prepare_error=RuntimeError("Odometry unavailable"))

    import navigation.expert_live_runner_factory as factory

    pass_preflight(monkeypatch)
    monkeypatch.setattr(
        factory,
        "create_live_runner",
        lambda *args, **kwargs: runner,
    )

    code = collector.main(execute_argv(tmp_path))
    captured = capsys.readouterr()
    summary = summary_from_output(captured.out)

    assert code == 1
    assert "EXECUTE: FAIL" in captured.out
    assert runner.closed
    assert summary["outcomes"][0]["status"] == "ERROR"
    assert "Odometry unavailable" in summary["outcomes"][0]["error"]
    assert summary["outcomes"][0]["exported"] is False
    assert not (tmp_path / "output").exists()


def test_cleanup_failure_does_not_export(monkeypatch, tmp_path, capsys):
    class DirtyRunner(FakeRunner):
        def close(self):
            super().close()
            raise RuntimeError("Stop failed")

    import navigation.expert_live_runner_factory as factory

    pass_preflight(monkeypatch)
    monkeypatch.setattr(
        factory,
        "create_live_runner",
        lambda *args, **kwargs: DirtyRunner(),
    )

    code = collector.main(execute_argv(tmp_path))
    summary = summary_from_output(capsys.readouterr().out)

    assert code == 1
    assert summary["outcomes"][0]["status"] == "CLEANUP_FAILED"
    assert summary["outcomes"][0]["exported"] is False
    assert not (tmp_path / "output").exists()


def test_export_failure_does_not_claim_success(monkeypatch, tmp_path, capsys):
    import navigation.expert_dataset_export as exporter
    import navigation.expert_live_runner_factory as factory

    def fail_export(*args, **kwargs):
        raise OSError("Disk full")

    pass_preflight(monkeypatch)
    monkeypatch.setattr(factory, "create_live_runner", lambda *args, **kwargs: FakeRunner())
    monkeypatch.setattr(exporter, "export_expert_episode", fail_export)

    code = collector.main(execute_argv(tmp_path))
    captured = capsys.readouterr()
    summary = summary_from_output(captured.out)

    assert code == 1
    assert "EXECUTE: PASS" not in captured.out
    assert "EXECUTE: FAIL" in captured.out
    assert summary["outcomes"][0]["status"] == "ERROR"
    assert summary["outcomes"][0]["exported"] is False
    assert "Disk full" in summary["outcomes"][0]["error"]
