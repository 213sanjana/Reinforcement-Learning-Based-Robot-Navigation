
from types import SimpleNamespace

from navigation import collect_expert_demonstrations as collector
from navigation import expert_collection_preflight as graph
from navigation import expert_runtime_preflight as runtime


def result(passed):
    return SimpleNamespace(
        passed=passed,
        checks={"test_check": passed},
    )


def test_graph_failure_skips_runtime(monkeypatch, capsys):
    calls = []

    def graph_fail():
        calls.append("graph")
        return result(False)

    def runtime_check(timeout=5.0):
        calls.append("runtime")
        return result(True)

    monkeypatch.setattr(graph, "run_preflight", graph_fail)
    monkeypatch.setattr(runtime, "run_runtime_preflight", runtime_check)

    assert collector.main(["--mode", "preflight"]) == 1
    assert calls == ["graph"]

    output = capsys.readouterr().out
    assert "RUNTIME PREFLIGHT: SKIPPED" in output
    assert "PREFLIGHT: FAIL" in output


def test_runtime_failure_rejected(monkeypatch, capsys):
    calls = []

    def graph_check():
        calls.append("graph")
        return result(True)

    def runtime_check(timeout=5.0):
        calls.append("runtime")
        return result(False)

    monkeypatch.setattr(graph, "run_preflight", graph_check)
    monkeypatch.setattr(runtime, "run_runtime_preflight", runtime_check)

    assert collector.main(["--mode", "preflight"]) == 1
    assert calls == ["graph", "runtime"]

    output = capsys.readouterr().out
    assert "RUNTIME PREFLIGHT: FAIL" in output
    assert "PREFLIGHT: FAIL" in output


def test_all_checks_pass(monkeypatch, capsys):
    monkeypatch.setattr(
        graph,
        "run_preflight",
        lambda: result(True),
    )

    monkeypatch.setattr(
        runtime,
        "run_runtime_preflight",
        lambda timeout=5.0: result(True),
    )

    assert collector.main(["--mode", "preflight"]) == 0

    output = capsys.readouterr().out
    assert "ROS GRAPH PREFLIGHT: PASS" in output
    assert "RUNTIME PREFLIGHT: PASS" in output
    assert "PREFLIGHT: PASS" in output


def test_runtime_exception_does_not_report_success(monkeypatch, capsys):
    monkeypatch.setattr(
        graph,
        "run_preflight",
        lambda: result(True),
    )

    def runtime_error(timeout=5.0):
        raise RuntimeError("Sensor stream unavailable")

    monkeypatch.setattr(
        runtime,
        "run_runtime_preflight",
        runtime_error,
    )

    assert collector.main(["--mode", "preflight"]) == 1

    captured = capsys.readouterr()

    assert "RUNTIME PREFLIGHT ERROR" in captured.err
    assert "Sensor stream unavailable" in captured.err
    assert "PREFLIGHT: FAIL" in captured.out.splitlines()
    assert "PREFLIGHT: PASS" not in captured.out.splitlines()


def test_execute_runtime_failure_blocks_runner(monkeypatch, tmp_path, capsys):
    calls = []

    def graph_check():
        calls.append("graph")
        return result(True)

    def runtime_check(timeout=5.0):
        calls.append("runtime")
        return result(False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Runner must not be created")

    import navigation.expert_live_runner_factory as factory

    monkeypatch.setattr(graph, "run_preflight", graph_check)
    monkeypatch.setattr(runtime, "run_runtime_preflight", runtime_check)
    monkeypatch.setattr(factory, "create_live_runner", forbidden)

    world = tmp_path / "world.sdf"
    world.write_text("<sdf version='1.7'></sdf>")

    code = collector.main([
        "--mode",
        "execute",
        "--world",
        str(world),
        "--output-dir",
        str(tmp_path / "output"),
    ])

    assert code == 1
    assert calls == ["graph", "runtime"]

    output = capsys.readouterr().out
    assert "EXECUTE: BLOCKED" in output
    assert "PREFLIGHT: FAIL" in output
    assert not (tmp_path / "output").exists()


def test_preview_does_not_run_preflight(monkeypatch, capsys):
    def forbidden():
        raise AssertionError("Preflight must not run during preview")

    monkeypatch.setattr(graph, "run_preflight", forbidden)

    assert collector.main(["--mode", "preview"]) == 0

    output = capsys.readouterr().out
    assert '"movement_enabled": false' in output
    assert '"simulation_reset_enabled": false' in output


def test_graph_exception_fails_closed(monkeypatch, capsys):
    def graph_error():
        raise RuntimeError("Controller manager unavailable")

    monkeypatch.setattr(graph, "run_preflight", graph_error)

    assert collector.main(["--mode", "preflight"]) == 1

    captured = capsys.readouterr()

    assert "ROS GRAPH PREFLIGHT ERROR" in captured.err
    assert "Controller manager unavailable" in captured.err
    assert "PREFLIGHT: FAIL" in captured.out.splitlines()
    assert "PREFLIGHT: PASS" not in captured.out.splitlines()
