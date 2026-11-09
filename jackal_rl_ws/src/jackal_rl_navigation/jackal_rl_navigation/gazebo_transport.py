"""Owned persistent Ignition Transport 11 client; no ROS/Gazebo process ownership.

The installed SDK is used through a small compiled helper, avoiding a new
reply endpoint for every physics step. No automatic CLI fallback is used.
The helper has a 5s SDK request deadline (from the existing controller),
and the Python pipe wait retains the existing 7s outer deadline.
"""
import hashlib
import os
from pathlib import Path
import selectors
import shlex
import subprocess
import tempfile
import time
import uuid


class TransportExchangeError(OSError):
    execution_unknown = True


def build_transport_helper():
    source = Path(__file__).with_name("gazebo_transport_client.cpp")
    flags = subprocess.run(
        ["pkg-config", "--cflags", "--libs", "ignition-transport11", "ignition-msgs8"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    digest = hashlib.sha256(source.read_bytes() + flags.encode()).hexdigest()
    directory = Path(tempfile.gettempdir()) / "jackal_transport_cache" / digest
    binary = directory / "client"
    if not binary.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / ("client_" + uuid.uuid4().hex)
        subprocess.run(["g++", "-O2", str(source), "-o", str(temporary), *shlex.split(flags)],
                       capture_output=True, text=True, check=True, timeout=60)
        os.replace(temporary, binary)
    return binary


class PersistentGazeboClient:
    def __init__(self, env, stderr_path, binary=None):
        self.env = env
        self.stderr_path = Path(stderr_path)
        self.binary = binary
        self.process = None
        self.stderr_handle = None
        self.last_pid = None

    @property
    def pid(self):
        return None if self.process is None else self.process.pid

    def _start(self):
        if self.process is not None:
            if self.process.poll() is not None:
                raise OSError("Persistent Gazebo transport client exited")
            return
        if self.binary is None:
            self.binary = build_transport_helper()
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_handle = self.stderr_path.open("ab", buffering=0)
        self.process = subprocess.Popen(
            [str(self.binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr_handle, bufsize=0, env=self.env,
        )
        self.last_pid = self.process.pid

    def request(self, service, request_type, data, timeout_ms, outer_timeout):
        self._start()
        request_id = uuid.uuid4().hex
        fields = [request_id, str(timeout_ms), service, request_type, data]
        if any("\n" in value or "\t" in value for value in fields):
            raise ValueError("Transport protocol fields cannot contain tabs/newlines")
        deadline = time.monotonic() + outer_timeout
        try:
            self.process.stdin.write(("\t".join(fields) + "\n").encode())
            reply = bytearray()
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while b"\n" not in reply:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise subprocess.TimeoutExpired("persistent Gazebo request", outer_timeout)
                    chunk = os.read(self.process.stdout.fileno(), 4096)
                    if not chunk:
                        raise OSError("Persistent Gazebo transport reply pipe closed")
                    reply.extend(chunk)
            parts = reply.decode().strip().split("\t")
            if len(parts) != 2 or parts[0] != request_id:
                raise OSError("Persistent Gazebo transport reply ID mismatch")
            status = parts[1]
            if status == "OK_TRUE":
                return subprocess.CompletedProcess("persistent", 0, "data: true\n", "")
            if status == "OK_FALSE":
                return subprocess.CompletedProcess("persistent", 0, "data: false\n", "")
            detail = "Service call timed out" if status == "TIMEOUT" else f"Transport reply: {status}"
            return subprocess.CompletedProcess("persistent", 1, "", detail)
        except OSError as exc:
            self.close()
            raise TransportExchangeError(str(exc)) from exc
        except BaseException:
            # Never let an uncorrelated late response satisfy a future call.
            self.close()
            raise

    def close(self):
        process, self.process = self.process, None
        try:
            if process is not None:
                try:
                    process.stdin.close()
                    process.wait(timeout=1)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1)
                finally:
                    process.stdout.close()
        finally:
            if self.stderr_handle is not None:
                self.stderr_handle.close()
                self.stderr_handle = None
