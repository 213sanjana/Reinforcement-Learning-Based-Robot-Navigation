"""Bounded Gazebo Fortress simulation-control interface."""

import math
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from jackal_rl_navigation.gazebo_transport import PersistentGazeboClient


# Keep the Jackal world off the default transport so another Gazebo
# process cannot inject entities into this scene.
JACKAL_GAZEBO_PARTITION = "jackal_rl"


def gazebo_process_env(partition=None):
    """Environment for ign/gz tools that must reach one Jackal world.

    The default partition remains ``jackal_rl``. Callers of the external
    warehouse pass ``jackal_rl_external`` explicitly. This function does not
    read a world name and substitute one.
    """

    chosen = JACKAL_GAZEBO_PARTITION if partition is None else partition
    if not isinstance(chosen, str) or not chosen.strip():
        raise ValueError("Gazebo partition must be an explicit name")
    env = os.environ.copy()
    env["IGN_PARTITION"] = chosen
    env["GZ_PARTITION"] = chosen
    # Multicast discovery dies when the host interface reports
    # "network unreachable". Keep the Jackal graph on localhost.
    env["IGN_IP"] = "127.0.0.1"
    env["GZ_IP"] = "127.0.0.1"
    return env


class GazeboControlError(RuntimeError):
    """Raised when a Gazebo service fails or times out."""


class AmbiguousSimulationStep(GazeboControlError):
    """A submitted step may have executed; it must not be blindly repeated."""


class GazeboController:
    """Control an existing Gazebo Fortress world."""

    def __init__(
        self,
        world="jackal_obstacle_test",
        robot="j100_0000",
        timeout=5.0,
        diagnostics_path=None,
        partition=None,
    ):
        self.world = world
        self.partition = partition
        self.robot = robot
        self.timeout = float(timeout)

        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")

        self.prefix = f"/world/{world}"
        # Callers can place this append-only ledger in their run directory.
        # Default to a unique file, never a shared or overwritten log.
        self.diagnostics_path = Path(diagnostics_path) if diagnostics_path is not None else (
            Path(tempfile.gettempdir()) / f"jackal_control_{os.getpid()}_{uuid.uuid4().hex}.jsonl"
        )
        self.last_attempts = []
        self._transport = None

    def _execute_command(self, command, **options):
        """Use a stable response endpoint for this controller's lifetime."""
        if self._transport is None:
            self._transport = PersistentGazeboClient(
                env=options["env"],
                stderr_path=self.diagnostics_path.with_suffix(".transport.stderr"),
            )
        return self._transport.request(
            command[command.index("-s") + 1],
            command[command.index("--reqtype") + 1],
            command[command.index("-r") + 1],
            int(command[command.index("--timeout") + 1]), options["timeout"],
        )

    def close(self):
        """Release only the transport helper owned by this controller."""
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def _persist_attempt(self, record):
        """Flush every attempt; diagnostic I/O must not change control results."""
        try:
            self.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            with self.diagnostics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            logging.getLogger(__name__).warning("Cannot persist control diagnostics: %s", exc)

    def _request(self, service, request_type, response_type, data, *, retry_ambiguous=True):
        command = [
            "ign", "service",
            "-s", service,
            "--reqtype", request_type,
            "--reptype", response_type,
            "--timeout", str(int(self.timeout * 1000)),
            "-r", data,
        ]

        last_error = None
        self.last_attempts = []
        request_id = uuid.uuid4().hex

        def error(message, ambiguous=False):
            if ambiguous:
                message += "\nAmbiguous simulation transition: execution is unknown; multi_step was not retried."
            exc = (AmbiguousSimulationStep if ambiguous else GazeboControlError)(message)
            exc.attempts = self.last_attempts
            exc.diagnostics_path = str(self.diagnostics_path)
            return exc

        for attempt in range(3):
            started = time.monotonic()
            record = {
                "request_id": request_id, "operation": service.rsplit("/", 1)[-1],
                "service": service, "request": data, "request_type": request_type,
                "response_type": response_type, "attempt": attempt + 1,
                "wall_time": time.time(), "returncode": None, "stdout": "", "stderr": "",
                "classification": "interrupted",
            }
            try:
                result = self._execute_command(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout + 2.0,
                    check=False,
                    env=gazebo_process_env(self.partition),
                )
            except subprocess.TimeoutExpired as exc:
                def decoded(value):
                    return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
                record.update(classification="subprocess_timeout", stdout=decoded(exc.output), stderr=decoded(exc.stderr))
                last_error = error(
                    f"Gazebo service timed out: {service}", ambiguous=not retry_ambiguous,
                )
                if not retry_ambiguous:
                    record.update(execution_status="unknown", retry_suppressed=True)
                last_error.__cause__ = exc
            except OSError as exc:
                record.update(classification="execution_error", exception=f"{type(exc).__name__}: {exc}")
                raise error(
                    f"Cannot execute Gazebo service: {exc}",
                    ambiguous=not retry_ambiguous and getattr(exc, "execution_unknown", False),
                ) from exc
            except BaseException as exc:
                record["exception"] = f"{type(exc).__name__}: {exc}"
                raise
            else:
                record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
                detail = f"{result.stdout}\n{result.stderr}"

                if result.returncode == 0 and result.stdout.strip() == "data: true":
                    record["classification"] = "success"
                    return True

                last_error = error(
                    f"Gazebo rejected request: {service}\n{detail}"
                )
                lowered = detail.lower()
                if "timed out" in lowered:
                    record["classification"] = "service_timeout"
                elif "unreachable" in lowered or "getifaddrs" in lowered:
                    record["classification"] = "transport_error"
                elif result.stdout.strip() == "data: false":
                    record["classification"] = "boolean_false"
                else:
                    record["classification"] = "command_failure" if result.returncode != 0 else "unexpected_response"

                if not retry_ambiguous and record["classification"] != "boolean_false":
                    record.update(execution_status="unknown", retry_suppressed=True)
                    raise error(
                        f"Gazebo rejected request: {service}\n{detail}", ambiguous=True,
                    )
                if not any(
                    token in lowered
                    for token in ("timed out", "unreachable", "getifaddrs")
                ):
                    raise last_error
            finally:
                record["duration_s"] = time.monotonic() - started
                record["backend"] = "persistent_ignition_transport11"
                record["helper_pid"] = None if self._transport is None else self._transport.last_pid
                self.last_attempts.append(record)
                self._persist_attempt(record)

            if not retry_ambiguous:
                raise last_error
            if attempt < 2:
                time.sleep(0.4)

        raise last_error

    def resume(self):
        """Resume Gazebo physics after a paused reset."""
        return self._request(
            f"{self.prefix}/control",
            "ignition.msgs.WorldControl",
            "ignition.msgs.Boolean",
            "pause: false",
        )

    def pause(self):
        """Pause simulation physics."""
        return self._request(
            f"{self.prefix}/control",
            "ignition.msgs.WorldControl",
            "ignition.msgs.Boolean",
            "pause: true",
        )

    def step(self, iterations=100):
        """Submit an advance once; uncertain execution fails closed.

        ROS clock and sensor freshness do not prove a request's exact
        execution count. No automatic lost-acknowledgement recovery is used.
        """
        if not isinstance(iterations, int) or not 1 <= iterations <= 100000:
            raise ValueError("iterations must be between 1 and 100000")

        return self._request(
            f"{self.prefix}/control",
            "ignition.msgs.WorldControl",
            "ignition.msgs.Boolean",
            f"pause: true, multi_step: {iterations}",
            retry_ambiguous=False,
        )

    def set_pose(self, x, y, yaw=0.0, z=0.3):
        """Set the simulated robot pose."""
        values = (x, y, yaw, z)

        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Pose values must be finite")

        qz = math.sin(float(yaw) / 2.0)
        qw = math.cos(float(yaw) / 2.0)

        request = (
            f'name: "{self.robot}", '
            f'position: {{x: {float(x)}, y: {float(y)}, '
            f'z: {float(z)}}}, '
            f'orientation: {{z: {qz}, w: {qw}}}'
        )

        return self._request(
            f"{self.prefix}/set_pose",
            "ignition.msgs.Pose",
            "ignition.msgs.Boolean",
            request,
        )
