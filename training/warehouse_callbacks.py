"""Training callbacks that record episodes and stop on safety faults."""

from __future__ import annotations

import csv
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from jackal_rl_navigation.warehouse_episode import FATAL_REASONS


EPISODE_FIELDS = (
    "timestep",
    "episode_index",
    "scenario_id",
    "split",
    "termination_reason",
    "success",
    "collision",
    "timeout",
    "fatal",
    "episode_return",
    "episode_length",
    "goal_distance",
    "path_length_m",
    "duration_s",
    "time_to_goal_s",
    "mean_linear_speed_mps",
    "mean_angular_speed_radps",
    "min_clearance_m",
    "rotations",
    "progress_reward",
    "goal_reward",
    "collision_penalty",
    "step_penalty",
    "proximity_penalty",
    "angular_penalty",
    "total_reward",
)

PROGRESS_FIELDS = (
    "timestep",
    "episodes",
    "mean_return",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "mean_episode_length",
    "mean_goal_distance",
    "mean_duration_s",
    "mean_min_clearance_m",
)


class WarehouseTrainingCallback(BaseCallback):
    """Write episode logs and refuse to continue after an infrastructure fault."""

    def __init__(self, directory: Path, checkpoint_interval: int, min_episodes_for_best: int = 5):
        super().__init__()
        self.directory = Path(directory)
        self.checkpoint_interval = int(checkpoint_interval)
        self.min_episodes_for_best = int(min_episodes_for_best)
        self.episodes = []
        self.best_success = -1.0
        self.stop_reason = None
        self._episode_file = None
        self._progress_file = None
        self._episode_writer = None
        self._progress_writer = None

    def _on_training_start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._episode_file = (self.directory / "episodes.csv").open("w", encoding="utf-8", newline="")
        self._progress_file = (self.directory / "progress.csv").open("w", encoding="utf-8", newline="")
        self._episode_writer = csv.DictWriter(self._episode_file, fieldnames=EPISODE_FIELDS)
        self._progress_writer = csv.DictWriter(self._progress_file, fieldnames=PROGRESS_FIELDS)
        self._episode_writer.writeheader()
        self._progress_writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        for info in infos:
            if not isinstance(info, dict) or info.get("termination_reason") is None:
                continue
            row = self._episode_row(info)
            self.episodes.append(row)
            self._episode_writer.writerow(row)
            self._episode_file.flush()
            if info.get("fatal") or info.get("termination_reason") in FATAL_REASONS:
                self.stop_reason = "safety fault: " + str(info.get("termination_reason"))
                self._write_progress()
                return False
        if self.checkpoint_interval > 0 and self.n_calls % self.checkpoint_interval == 0:
            self._save_periodic()
            self._write_progress()
        self._save_best()
        return True

    def _episode_row(self, info: dict) -> dict:
        components = info.get("reward_components") or {}
        return {
            "timestep": int(self.num_timesteps),
            "episode_index": len(self.episodes),
            "scenario_id": info.get("scenario_id", ""),
            "split": info.get("split", ""),
            "termination_reason": info.get("termination_reason"),
            "success": bool(info.get("success")),
            "collision": bool(info.get("collision")),
            "timeout": bool(info.get("timeout")),
            "fatal": bool(info.get("fatal")),
            "episode_return": info.get("episode_return", info.get("total_reward")),
            "episode_length": info.get("episode_length"),
            "goal_distance": info.get("goal_distance"),
            "path_length_m": info.get("path_length_m"),
            "duration_s": info.get("duration_s"),
            "time_to_goal_s": info.get("time_to_goal_s"),
            "mean_linear_speed_mps": info.get("mean_linear_speed_mps"),
            "mean_angular_speed_radps": info.get("mean_angular_speed_radps"),
            "min_clearance_m": info.get("min_clearance_m"),
            "rotations": info.get("rotations"),
            "progress_reward": components.get("progress_reward", info.get("progress_reward")),
            "goal_reward": components.get("goal_reward", info.get("goal_reward")),
            "collision_penalty": components.get("collision_penalty", info.get("collision_penalty")),
            "step_penalty": components.get("step_penalty", info.get("step_penalty")),
            "proximity_penalty": components.get("proximity_penalty", info.get("proximity_penalty")),
            "angular_penalty": components.get("angular_penalty", info.get("angular_penalty")),
            "total_reward": info.get("total_reward"),
        }

    def _window_mean(self, key: str):
        values = []
        for episode in self.episodes:
            value = episode.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        if not values:
            return None
        return sum(values) / len(values)

    def _rate(self, key: str) -> float:
        if not self.episodes:
            return 0.0
        return sum(bool(episode[key]) for episode in self.episodes) / len(self.episodes)

    def _write_progress(self) -> None:
        if self._progress_writer is None:
            return
        self._progress_writer.writerow(
            {
                "timestep": int(self.num_timesteps),
                "episodes": len(self.episodes),
                "mean_return": self._window_mean("episode_return"),
                "success_rate": self._rate("success"),
                "collision_rate": self._rate("collision"),
                "timeout_rate": self._rate("timeout"),
                "mean_episode_length": self._window_mean("episode_length"),
                "mean_goal_distance": self._window_mean("goal_distance"),
                "mean_duration_s": self._window_mean("duration_s"),
                "mean_min_clearance_m": self._window_mean("min_clearance_m"),
            }
        )
        self._progress_file.flush()

    def _save_periodic(self) -> None:
        target = self.directory / "checkpoints" / f"step_{int(self.num_timesteps)}"
        target.mkdir(parents=True, exist_ok=True)
        self.model.save(str(target / "ppo_model"))

    def _save_best(self) -> None:
        if len(self.episodes) < self.min_episodes_for_best:
            return
        success = self._rate("success")
        if success <= self.best_success:
            return
        self.best_success = success
        target = self.directory / "best_model"
        target.mkdir(parents=True, exist_ok=True)
        self.model.save(str(target / "ppo_model"))

    def _on_training_end(self) -> None:
        self._write_progress()
        latest = self.directory / "latest"
        latest.mkdir(parents=True, exist_ok=True)
        self.model.save(str(latest / "ppo_model"))
        for handle in (self._episode_file, self._progress_file):
            if handle is not None:
                handle.close()
        if self.stop_reason is not None:
            (self.directory / "stop_reason.txt").write_text(self.stop_reason + "\n", encoding="utf-8")
