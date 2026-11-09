"""Mocked subprocess tests; these do not contact Gazebo or ROS."""
import json
import subprocess
from types import SimpleNamespace

import pytest

from jackal_rl_navigation import gazebo_controller as module


def result(code=0, stdout="data: true\n", stderr=""):
    return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr)


def controller(tmp_path, monkeypatch, outcomes):
    calls, sleeps = [], []
    outcomes = iter(outcomes)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        outcome = next(outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(module.GazeboController, "_execute_command", lambda self, command, **kwargs: run(command, **kwargs))
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    client = module.GazeboController(diagnostics_path=tmp_path / "attempts.jsonl")
    return client, calls, sleeps


def records(client):
    return [json.loads(line) for line in client.diagnostics_path.read_text().splitlines()]


@pytest.mark.parametrize("outcome,classification", [
    (result(0, "data: false\n"), "boolean_false"),
    (result(2, "", "bad request"), "command_failure"),
    (result(0, "unexpected"), "unexpected_response"),
    (FileNotFoundError("ign"), "execution_error"),
])
def test_nonretryable_failures_persist_once(tmp_path, monkeypatch, outcome, classification):
    client, calls, sleeps = controller(tmp_path, monkeypatch, [outcome])
    with pytest.raises(module.GazeboControlError) as failure:
        client.step(100)
    rows = records(client)
    assert len(calls) == len(rows) == 1 and not sleeps
    assert rows[0]["classification"] == classification
    assert failure.value.attempts == rows
    assert failure.value.diagnostics_path == str(client.diagnostics_path)


def test_all_attempts_survive_mixed_timeout_failure(tmp_path, monkeypatch):
    expired = subprocess.TimeoutExpired("ign", 7, output=b"partial", stderr=b"slow")
    client, calls, sleeps = controller(tmp_path, monkeypatch, [
        expired, result(1, "", "Service call timed out"), result(1, "", "network unreachable"),
    ])
    with pytest.raises(module.GazeboControlError) as failure:
        client.pause()
    rows = records(client)
    assert [r["classification"] for r in rows] == ["subprocess_timeout", "service_timeout", "transport_error"]
    assert [r["attempt"] for r in rows] == [1, 2, 3]
    assert rows[0]["stdout"] == "partial" and rows[0]["stderr"] == "slow"
    assert rows[0]["returncode"] is None and rows[1]["returncode"] == 1
    assert len({r["request_id"] for r in rows}) == 1
    assert failure.value.attempts == rows
    assert sleeps == [0.4, 0.4]
    for command, kwargs in calls:
        assert kwargs["timeout"] == 7.0
        assert command[command.index("--timeout") + 1] == "5000"
        assert command[-1] == "pause: true"
        assert kwargs["env"]["IGN_PARTITION"] == "jackal_rl"
    assert all(r["duration_s"] >= 0 and r["operation"] == "control" for r in rows)


def test_success_after_retry_and_append_preserve_existing_log(tmp_path, monkeypatch):
    client, calls, sleeps = controller(tmp_path, monkeypatch, [
        result(1, "", "Service call timed out"), result(), result(),
    ])
    assert client.pause() is True
    first = records(client)
    assert [r["classification"] for r in first] == ["service_timeout", "success"]
    assert client.pause() is True
    rows = records(client)
    assert rows[:2] == first and len(rows) == 3
    assert rows[2]["request_id"] != rows[0]["request_id"]
    assert rows[2]["attempt"] == 1 and sleeps == [0.4]


def test_diagnostic_io_failure_does_not_change_success(tmp_path, monkeypatch, caplog):
    client, _, _ = controller(tmp_path, monkeypatch, [result()])
    client.diagnostics_path = tmp_path  # Opening a directory must fail.
    assert client.step(100) is True
    assert "Cannot persist control diagnostics" in caplog.text


def test_interruption_is_logged_without_retry(tmp_path, monkeypatch):
    client, calls, sleeps = controller(tmp_path, monkeypatch, [KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        client.step(100)
    row = records(client)[0]
    assert row["classification"] == "interrupted"
    assert "KeyboardInterrupt" in row["exception"]
    assert len(calls) == 1 and not sleeps


def test_normal_step_success_is_single_submission(tmp_path, monkeypatch):
    client, calls, sleeps = controller(tmp_path, monkeypatch, [result()])
    assert client.step(100) is True
    assert len(calls) == 1 and not sleeps
    assert records(client)[0]["classification"] == "success"


@pytest.mark.parametrize("progress", [0.0, 0.1, 0.2, 0.05])
@pytest.mark.parametrize("failure", ["service_timeout", "subprocess_timeout", "transport_error"])
def test_ambiguous_step_never_resends_regardless_of_apparent_progress(tmp_path, monkeypatch, progress, failure):
    # Simulate no execution, intended advance, excess advance, or partial
    # advancement. A clock value cannot correlate execution to this request.
    simulated_clock = {"seconds": 10.0}
    calls = []
    sleeps = []

    def run(command, **kwargs):
        calls.append(command)
        simulated_clock["seconds"] += progress
        if failure == "subprocess_timeout":
            raise subprocess.TimeoutExpired(command, 7)
        detail = "Service call timed out" if failure == "service_timeout" else "Host unreachable"
        return result(0, "", detail)

    monkeypatch.setattr(module.GazeboController, "_execute_command", lambda self, command, **kwargs: run(command, **kwargs))
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    client = module.GazeboController(diagnostics_path=tmp_path / "step.jsonl")
    with pytest.raises(module.AmbiguousSimulationStep, match="execution is unknown") as failure_info:
        client.step(100)
    assert simulated_clock["seconds"] == pytest.approx(10.0 + progress)
    assert len(calls) == 1 and not sleeps
    row = records(client)[0]
    assert row["classification"] == failure
    assert row["execution_status"] == "unknown" and row["retry_suppressed"] is True
    assert failure_info.value.attempts == [row]
