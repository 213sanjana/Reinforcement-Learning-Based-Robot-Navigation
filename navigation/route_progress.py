"""Monitor forward progress along an A* route."""

import math
import time


class RouteProgressWatchdog:
    """Detect navigation that fails to advance along a path."""

    def __init__(
        self,
        path,
        stall_timeout=12.0,
        min_progress=0.10,
    ):
        if len(path) < 2:
            raise ValueError("At least two waypoints required")

        if stall_timeout <= 0 or min_progress <= 0:
            raise ValueError("Invalid watchdog configuration")

        self.path = [
            (float(x), float(y))
            for x, y in path
        ]

        self.stall_timeout = float(stall_timeout)
        self.min_progress = float(min_progress)

        self.cumulative = [0.0]

        for a, b in zip(self.path[:-1], self.path[1:]):
            self.cumulative.append(
                self.cumulative[-1]
                + math.hypot(
                    b[0] - a[0],
                    b[1] - a[1],
                )
            )

        self.best_progress = 0.0
        self.reference_progress = 0.0
        self.last_progress_time = None

    def project(self, x, y):
        """Project robot position onto the nearest path segment."""

        candidates = []

        for i, (a, b) in enumerate(
            zip(self.path[:-1], self.path[1:])
        ):
            dx = b[0] - a[0]
            dy = b[1] - a[1]

            length_sq = dx * dx + dy * dy

            if length_sq <= 1e-12:
                t = 0.0
            else:
                t = max(
                    0.0,
                    min(
                        1.0,
                        (
                            (x - a[0]) * dx
                            + (y - a[1]) * dy
                        ) / length_sq,
                    ),
                )

            px = a[0] + t * dx
            py = a[1] + t * dy

            distance = math.hypot(x - px, y - py)

            segment_length = math.sqrt(length_sq)

            progress = (
                self.cumulative[i]
                + t * segment_length
            )

            candidates.append((distance, progress))

        return min(candidates, key=lambda item: item[0])

    def reset(self, pose, now=None):
        now = time.monotonic() if now is None else float(now)

        if not all(math.isfinite(v) for v in pose[:2]):
            raise ValueError("Invalid robot position")

        _, progress = self.project(*pose[:2])

        self.best_progress = progress
        self.reference_progress = progress
        self.last_progress_time = now

    def evaluate(self, pose, now=None):
        now = time.monotonic() if now is None else float(now)

        if not all(math.isfinite(v) for v in pose[:2]):
            return False, "Invalid robot position", 0.0

        if self.last_progress_time is None:
            self.reset(pose, now)

        if now < self.last_progress_time:
            return False, "Invalid watchdog time", self.best_progress

        distance, progress = self.project(*pose[:2])

        self.best_progress = max(
            self.best_progress,
            progress,
        )

        if (
            self.best_progress - self.reference_progress
            >= self.min_progress
        ):
            self.reference_progress = self.best_progress
            self.last_progress_time = now

        if now - self.last_progress_time > self.stall_timeout:
            return (
                False,
                "No forward progress along A* route",
                self.best_progress,
            )

        return (
            True,
            "Route progress acceptable",
            self.best_progress,
        )
