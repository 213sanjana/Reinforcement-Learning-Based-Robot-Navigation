"""Local pipe mocks only: no Gazebo, transport sockets, or physics."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import subprocess

from jackal_rl_navigation import gazebo_transport as transport


class FakeHelper:
    def __init__(self, statuses):
        read, self.write_fd = os.pipe()
        self.stdout = os.fdopen(read, "rb", buffering=0)
        self.statuses = iter(statuses)
        self.requests = []
        self.pid = 321
        self.exited = False
        self.stdin = SimpleNamespace(write=self.write, close=lambda: None)

    def write(self, data):
        fields = data.decode().strip().split("\t")
        self.requests.append(fields)
        status = next(self.statuses)
        if status is not None:
            request_id = "wrong" if status == "WRONG_ID" else fields[0]
            os.write(self.write_fd, f"{request_id}\t{status}\n".encode())
        return len(data)

    def poll(self):
        return 0 if self.exited else None

    def wait(self, timeout):
        self.exited = True
        os.close(self.write_fd)
        return 0


def client(tmp_path, monkeypatch, statuses):
    helper = FakeHelper(statuses)
    calls = []

    def start(*args, **kwargs):
        calls.append((args, kwargs))
        return helper

    monkeypatch.setattr(transport.subprocess, "Popen", start)
    native = transport.PersistentGazeboClient(
        {"IGN_PARTITION": "jackal_rl"}, tmp_path / "client.stderr", binary=Path("/fake/client"),
    )
    return native, helper, calls


def request(native, timeout=1):
    return native.request("/world/jackal_obstacle_test/control", "ignition.msgs.WorldControl",
                          "pause: true, multi_step: 100", 5000, timeout)


def test_reuses_one_client_and_correlates_each_response(tmp_path, monkeypatch):
    native, helper, calls = client(tmp_path, monkeypatch, ["OK_TRUE", "OK_FALSE", "TIMEOUT", "OK_TRUE"])
    try:
        assert request(native).stdout == "data: true\n"
        assert request(native).stdout == "data: false\n"
        assert request(native).stderr == "Service call timed out"
        assert request(native).stdout == "data: true\n"
        assert len(calls) == 1
        assert len({r[0] for r in helper.requests}) == 4
        assert all(r[1] == "5000" for r in helper.requests)
    finally:
        native.close()
    assert helper.exited and native.process is None and native.stderr_handle is None
    native.close()


@pytest.mark.parametrize("status", [None, "WRONG_ID"])
def test_missing_or_uncorrelated_reply_closes_owned_client(tmp_path, monkeypatch, status):
    native, helper, calls = client(tmp_path, monkeypatch, [status])
    error = subprocess.TimeoutExpired if status is None else transport.TransportExchangeError
    with pytest.raises(error):
        request(native, timeout=0.01)
    assert helper.exited and native.process is None
    assert len(helper.requests) == len(calls) == 1
