"""Progress and timeout monitoring for autonomous navigation."""

import math
import time


class NavigationProgressWatchdog:
    """Detect lack of motion progress and navigation timeout."""

    def __init__(
        self,
        timeout=120.0,
        stall_timeout=10.0,
        min_progress=0.10,
    ):
        if timeout <= 0 or stall_timeout <= 0 or min_progress <= 0:
            raise ValueError("Watchdog parameters must be positive")

        self.timeout = float(timeout)
        self.stall_timeout = float(stall_timeout)
        self.min_progress = float(min_progress)

        self.started_at = None
        self.progress_at = None
        self.reference_pose = None

    def reset(self, pose, now=None):
        """Start monitoring from the current robot pose."""

        if not all(math.isfinite(v) for v in pose[:2]):
            raise ValueError("Invalid robot position")

        now = time.monotonic() if now is None else float(now)

        self.started_at = now
        self.progress_at = now
        self.reference_pose = tuple(pose[:2])

    def evaluate(self, pose, now=None):
        """Return (safe, reason) based on movement and elapsed time."""

        now = time.monotonic() if now is None else float(now)

        if not all(math.isfinite(v) for v in pose[:2]):
            return False, "Invalid robot position"

        if self.started_at is None:
            self.reset(pose, now)
            return True, "Watchdog initialized"

        if now < self.started_at:
            return False, "Invalid watchdog time"

        if now - self.started_at > self.timeout:
            return False, "Navigation timeout"

        distance = math.hypot(
            pose[0] - self.reference_pose[0],
            pose[1] - self.reference_pose[1],
        )

        if distance >= self.min_progress:
            self.reference_pose = tuple(pose[:2])
            self.progress_at = now

        if now - self.progress_at > self.stall_timeout:
            return False, "Navigation stalled"

        return True, "Navigation progress acceptable"
