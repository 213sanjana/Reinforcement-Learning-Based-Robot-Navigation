"""Plots and comparison report for Phase 6G.3 diagnostic PPO runs.

The figures use the recorded rows only. A line between validation checkpoints
connects measured evaluations; it does not fill in missing runs.
"""

import csv
import json
from pathlib import Path

import numpy as np

from navigation.jackal_ppo_diagnostic import (
    COMPARABILITY_NOTES,
    OFFICIAL_BC_SHA256,
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_SMOKE_CHECKPOINT,
    OFFICIAL_SMOKE_SHA256,
    SELECTION_RULE,
    VALIDATION_SCENARIOS,
    _reject_output,
    _resolve,
    file_sha256,
)


ROLLING_EPISODE_WINDOW = 10
ROLLING_LOSS_WINDOW = 5
_ROOT = Path(__file__).resolve().parents[1]


def write_comparison(bc_directory, scratch_directory, output_directory):
    """Write plots and a markdown report for two completed or partial runs."""

    bc_directory = _resolve(bc_directory)
    scratch_directory = _resolve(scratch_directory)
    output = _resolve(output_directory)
    _reject_output(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if bc_directory == scratch_directory:
        raise ValueError("BC and scratch runs must be different directories")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    bc = _load_run(bc_directory)
    scratch = _load_run(scratch_directory)
    _write_series(output, bc, scratch)
    plot_paths = _write_plots(output, bc, scratch)
    report = _report_text(bc, scratch, plot_paths)
    path = output / "comparison_report.md"
    path.write_text(report, encoding="utf-8")
    summary = {
        "output_directory": str(output),
        "bc_directory": str(bc_directory),
        "scratch_directory": str(scratch_directory),
        "bc_status": bc["summary"].get("status"),
        "scratch_status": scratch["summary"].get("status"),
        "plots": [str(item) for item in plot_paths],
        "report": str(path),
        "source_hashes": _source_hashes(),
    }
    (output / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_run(directory):
    directory = Path(directory)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {
        "status": "missing",
        "output_directory": str(directory),
    }
    episodes = _read_json(directory / "episodes.json", [])
    evaluations = _read_json(directory / "evaluations.json", {"records": []}).get("records") or []
    updates = _read_jsonl(directory / "updates.jsonl")
    if not updates and (directory / "optimization.json").is_file():
        updates = json.loads((directory / "optimization.json").read_text(encoding="utf-8")).get("updates") or []
    return {
        "directory": directory,
        "summary": summary,
        "episodes": episodes,
        "evaluations": evaluations,
        "updates": updates,
        "label": summary.get("experiment") or directory.name,
    }


def _write_series(output, bc, scratch):
    _write_episode_series(output / "episode_series.csv", (("bc_initialized", bc), ("scratch", scratch)))
    _write_update_series(output / "update_series.csv", (("bc_initialized", bc), ("scratch", scratch)))
    _write_validation_series(output / "validation_series.csv", (("bc_initialized", bc), ("scratch", scratch)))
    _write_scenario_table(output / "per_scenario_final.csv", (("bc_initialized", bc), ("scratch", scratch)))


def _write_episode_series(path, runs):
    rows = ["experiment,environment_timesteps,episode_index,status,episode_return,success,collision,timeout,rolling_return,rolling_success_rate,rolling_collision_rate,rolling_timeout_rate"]
    for name, run in runs:
        completed = [episode for episode in run["episodes"] if episode.get("status") == "completed"]
        returns = []
        successes = []
        collisions = []
        timeouts = []
        for episode in run["episodes"]:
            if episode.get("status") == "completed":
                returns.append(float(episode["episode_return"]))
                successes.append(1.0 if episode.get("success") else 0.0)
                collisions.append(1.0 if episode.get("collision") else 0.0)
                timeouts.append(1.0 if episode.get("timeout") else 0.0)
                rolling_return = _trailing_mean(returns, ROLLING_EPISODE_WINDOW)
                rolling_success = _trailing_mean(successes, ROLLING_EPISODE_WINDOW)
                rolling_collision = _trailing_mean(collisions, ROLLING_EPISODE_WINDOW)
                rolling_timeout = _trailing_mean(timeouts, ROLLING_EPISODE_WINDOW)
            else:
                rolling_return = rolling_success = rolling_collision = rolling_timeout = ""
            rows.append(",".join(str(value) for value in (
                name,
                episode.get("environment_steps_completed"),
                episode.get("episode_index"),
                episode.get("status"),
                episode.get("episode_return"),
                episode.get("success"),
                episode.get("collision"),
                episode.get("timeout"),
                rolling_return,
                rolling_success,
                rolling_collision,
                rolling_timeout,
            )))
        del completed
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_update_series(path, runs):
    rows = ["experiment,environment_timesteps,policy_loss,value_loss,entropy_loss,approx_kl,explained_variance,learning_rate,log_std_0,log_std_1,std_0,std_1"]
    for name, run in runs:
        for update in run["updates"]:
            metrics = update.get("metrics") or {}
            log_std = list(update.get("log_std") or [None, None])
            std = list(update.get("std") or [None, None])
            rows.append(",".join(str(value) for value in (
                name,
                update.get("timesteps"),
                metrics.get("train/policy_gradient_loss"),
                metrics.get("train/value_loss"),
                metrics.get("train/entropy_loss"),
                metrics.get("train/approx_kl"),
                metrics.get("train/explained_variance"),
                update.get("learning_rate"),
                log_std[0] if log_std else None,
                log_std[1] if len(log_std) > 1 else None,
                std[0] if std else None,
                std[1] if len(std) > 1 else None,
            )))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_validation_series(path, runs):
    rows = ["experiment,environment_timesteps,success_rate,collision_rate,timeout_rate,mean_return,median_return,mean_length,mean_elapsed_s,mean_path_length_odom_m,complete"]
    for name, run in runs:
        for record in sorted(run["evaluations"], key=lambda item: int(item.get("timesteps") or 0)):
            rows.append(",".join(str(value) for value in (
                name,
                record.get("timesteps"),
                record.get("success_rate"),
                record.get("collision_rate"),
                record.get("timeout_rate"),
                record.get("mean_return"),
                record.get("median_return"),
                record.get("mean_length"),
                record.get("mean_elapsed_s"),
                record.get("mean_path_length_odom_m"),
                record.get("complete"),
            )))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_scenario_table(path, runs):
    rows = ["experiment,scenario_id,success,collision,timeout,episode_return,length,elapsed_s,path_length_odom_m,status"]
    for name, run in runs:
        final = _final_evaluation(run)
        for episode in (final or {}).get("episodes") or []:
            rows.append(",".join(str(value) for value in (
                name,
                episode.get("scenario_id"),
                episode.get("success"),
                episode.get("collision"),
                episode.get("timeout"),
                episode.get("episode_return"),
                episode.get("length"),
                episode.get("elapsed_s"),
                episode.get("path_length_odom_m"),
                episode.get("status"),
            )))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_plots(output, bc, scratch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    runs = (("BC-initialized", bc, "#1f4e79"), ("Scratch", scratch, "#b85c38"))
    paths.append(_plot_episode_returns(output, runs, plt))
    paths.append(_plot_rolling_outcomes(output, runs, plt))
    paths.append(_plot_losses(output, runs, plt))
    paths.append(_plot_entropy_kl(output, runs, plt))
    paths.append(_plot_validation(output, runs, plt))
    paths.append(_plot_wall_clock(output, runs, plt))
    paths.append(_plot_scenarios(output, runs, plt))
    return paths


def _plot_episode_returns(output, runs, plt):
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for label, run, color in runs:
        completed = [episode for episode in run["episodes"] if episode.get("status") == "completed"]
        if not completed:
            continue
        xs = [int(episode["environment_steps_completed"]) for episode in completed]
        ys = [float(episode["episode_return"]) for episode in completed]
        axis.scatter(xs, ys, s=12, alpha=0.35, color=color, label=f"{label} raw")
        rolled = [_trailing_mean(ys[: index + 1], ROLLING_EPISODE_WINDOW) for index in range(len(ys))]
        axis.plot(xs, rolled, color=color, label=f"{label} trailing {ROLLING_EPISODE_WINDOW}")
        _mark_incomplete(axis, run, max(xs))
    axis.set_xlabel("Environment timesteps")
    axis.set_ylabel("Episode return")
    axis.set_title("Training episode return")
    axis.legend(fontsize=8)
    return _save_figure(figure, output / "episode_return.png", plt)


def _plot_rolling_outcomes(output, runs, plt):
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for label, run, color in runs:
        completed = [episode for episode in run["episodes"] if episode.get("status") == "completed"]
        if not completed:
            continue
        xs = [int(episode["environment_steps_completed"]) for episode in completed]
        success = [1.0 if episode.get("success") else 0.0 for episode in completed]
        collision = [1.0 if episode.get("collision") else 0.0 for episode in completed]
        timeout = [1.0 if episode.get("timeout") else 0.0 for episode in completed]
        axis.plot(xs, [_trailing_mean(success[:i + 1], ROLLING_EPISODE_WINDOW) for i in range(len(xs))], color=color, label=f"{label} success")
        axis.plot(xs, [_trailing_mean(collision[:i + 1], ROLLING_EPISODE_WINDOW) for i in range(len(xs))], color=color, linestyle="--", label=f"{label} collision")
        axis.plot(xs, [_trailing_mean(timeout[:i + 1], ROLLING_EPISODE_WINDOW) for i in range(len(xs))], color=color, linestyle=":", label=f"{label} timeout")
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("Environment timesteps")
    axis.set_ylabel(f"Trailing rate, window {ROLLING_EPISODE_WINDOW}")
    axis.set_title("Completed-episode outcome rates")
    axis.legend(fontsize=7, ncol=2)
    return _save_figure(figure, output / "rolling_outcomes.png", plt)


def _plot_losses(output, runs, plt):
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for label, run, color in runs:
        xs, policy, value = [], [], []
        for update in run["updates"]:
            metrics = update.get("metrics") or {}
            if "train/policy_gradient_loss" not in metrics or "train/value_loss" not in metrics:
                continue
            xs.append(int(update["timesteps"]))
            policy.append(float(metrics["train/policy_gradient_loss"]))
            value.append(float(metrics["train/value_loss"]))
        if not xs:
            continue
        axis.plot(xs, policy, color=color, alpha=0.35)
        axis.plot(xs, [_trailing_mean(policy[:i + 1], ROLLING_LOSS_WINDOW) for i in range(len(xs))], color=color, label=f"{label} policy")
        axis.plot(xs, [_trailing_mean(value[:i + 1], ROLLING_LOSS_WINDOW) for i in range(len(xs))], color=color, linestyle="--", label=f"{label} value")
    axis.set_xlabel("Environment timesteps")
    axis.set_ylabel("Loss")
    axis.set_title(f"Policy and value loss, trailing window {ROLLING_LOSS_WINDOW}")
    axis.legend(fontsize=8)
    return _save_figure(figure, output / "losses.png", plt)


def _plot_entropy_kl(output, runs, plt):
    figure, axis = plt.subplots(figsize=(8, 4.5))
    kl_axis = axis.twinx()
    for label, run, color in runs:
        xs, entropy, kl = [], [], []
        for update in run["updates"]:
            metrics = update.get("metrics") or {}
            if "train/entropy_loss" not in metrics or "train/approx_kl" not in metrics:
                continue
            xs.append(int(update["timesteps"]))
            entropy.append(-float(metrics["train/entropy_loss"]))
            kl.append(float(metrics["train/approx_kl"]))
        if not xs:
            continue
        axis.plot(xs, entropy, color=color, label=f"{label} entropy")
        kl_axis.plot(xs, kl, color=color, linestyle="--", label=f"{label} KL")
    axis.set_xlabel("Environment timesteps")
    axis.set_ylabel("Entropy (-entropy_loss)")
    kl_axis.set_ylabel("Approximate KL")
    axis.set_title("Entropy and approximate KL")
    lines, labels = axis.get_legend_handles_labels()
    other_lines, other_labels = kl_axis.get_legend_handles_labels()
    axis.legend(lines + other_lines, labels + other_labels, fontsize=8)
    return _save_figure(figure, output / "entropy_kl.png", plt)


def _plot_validation(output, runs, plt):
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for label, run, color in runs:
        points = [
            record for record in run["evaluations"]
            if record.get("complete") and record.get("success_rate") is not None
        ]
        points = sorted(points, key=lambda item: int(item["timesteps"]))
        if not points:
            continue
        xs = [int(record["timesteps"]) for record in points]
        ys = [float(record["success_rate"]) for record in points]
        axis.plot(xs, ys, marker="o", color=color, label=label)
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel("Environment timesteps")
    axis.set_ylabel("Validation success rate")
    axis.set_title("Measured fixed-validation success rate")
    axis.legend(fontsize=8)
    return _save_figure(figure, output / "validation_success.png", plt)


def _plot_wall_clock(output, runs, plt):
    figure, axis = plt.subplots(figsize=(8, 4.5))
    labels = []
    training = []
    evaluation = []
    for label, run, _color in runs:
        labels.append(label)
        training.append(float(run["summary"].get("training_wall_time_s") or 0.0) / 3600.0)
        evaluation.append(float(run["summary"].get("evaluation_wall_time_s") or 0.0) / 3600.0)
    positions = np.arange(len(labels))
    axis.bar(positions - 0.15, training, width=0.3, label="Training")
    axis.bar(positions + 0.15, evaluation, width=0.3, label="Evaluation")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Hours")
    axis.set_title("Recorded wall-clock time")
    axis.legend(fontsize=8)
    return _save_figure(figure, output / "wall_clock.png", plt)


def _plot_scenarios(output, runs, plt):
    figure, axis = plt.subplots(figsize=(9, 4.8))
    scenarios = list(VALIDATION_SCENARIOS)
    width = 0.35
    positions = np.arange(len(scenarios))
    for offset, (label, run, color) in enumerate(runs):
        final = _final_evaluation(run) or {}
        by_id = {episode.get("scenario_id"): episode for episode in final.get("episodes") or []}
        values = [
            np.nan if scenario not in by_id else (1.0 if by_id[scenario].get("success") else 0.0)
            for scenario in scenarios
        ]
        axis.bar(positions + (offset - 0.5) * width, values, width=width, color=color, label=label)
    axis.set_xticks(positions, scenarios, rotation=15)
    axis.set_ylim(-0.05, 1.05)
    axis.set_ylabel("Final success (1) or no success (0)")
    axis.set_title("Per-scenario final validation outcome")
    axis.legend(fontsize=8)
    return _save_figure(figure, output / "per_scenario_final.png", plt)


def _save_figure(figure, path, plt):
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def _mark_incomplete(axis, run, last_x):
    status = run["summary"].get("status")
    if status not in (None, "completed"):
        axis.axvline(last_x, color="gray", linestyle=":", linewidth=1)
        axis.text(last_x, axis.get_ylim()[1], f" {status}", fontsize=8, color="gray")


def _report_text(bc, scratch, plot_paths):
    smoke = _smoke_context()
    lines = [
        "# Phase 6G.3 diagnostic comparison",
        "",
        "This report compares one BC-initialized PPO run with one PPO run from scratch.",
        "It is a single matched-seed diagnostic, not a statistical claim of general superiority.",
        "",
        "## Method",
        "",
        f"Selection rule, declared before training: {SELECTION_RULE}",
        "",
        "Comparability notes:",
        "",
    ]
    for note in COMPARABILITY_NOTES:
        lines.append(f"- {note}")
    lines.extend([
        "",
        "Phase 6G.2 recorded a 1,024-step smoke test. Finite losses there were not navigation improvement.",
        smoke,
        "",
        "## Run status",
        "",
        _run_paragraph("BC-initialized", bc),
        "",
        _run_paragraph("Scratch", scratch),
        "",
        "## Validation",
        "",
        _validation_paragraph("BC-initialized", bc),
        "",
        _validation_paragraph("Scratch", scratch),
        "",
        "## Interpretation",
        "",
    ])
    lines.extend(_interpretation(bc, scratch))
    lines.extend([
        "",
        "## Figures",
        "",
    ])
    for path in plot_paths:
        lines.append(f"- `{path.name}`")
    lines.extend([
        "",
        "Raw series are stored beside this report in `episode_series.csv`, `update_series.csv`, `validation_series.csv`, and `per_scenario_final.csv`.",
        f"Trailing episode statistics use a window of {ROLLING_EPISODE_WINDOW} completed episodes.",
        f"Trailing loss statistics use a window of {ROLLING_LOSS_WINDOW} updates.",
        "Partial training episodes are excluded from success, collision, and timeout rates.",
        "Missing validation checkpoints are omitted rather than interpolated.",
        "",
        "## Source checkpoint hashes",
        "",
        f"- Phase 6G.1 `{OFFICIAL_CHECKPOINT_SHA256}`",
        f"- Phase 6F.2 `{OFFICIAL_BC_SHA256}`",
        f"- Phase 6G.2 `{OFFICIAL_SMOKE_SHA256}`",
        "",
    ])
    current = _source_hashes()
    for label, digest in current.items():
        lines.append(f"- Current {label}: `{digest['sha256']}` unchanged={digest['unchanged']}")
    lines.append("")
    return "\n".join(lines)


def _interpretation(bc, scratch):
    lines = []
    for label, run in (("BC initialization", bc), ("Scratch PPO", scratch)):
        initial = _evaluation_at_start(run)
        final = _final_evaluation(run)
        if initial is None or final is None or initial.get("success_rate") is None or final.get("success_rate") is None:
            lines.append(f"{label} does not have both an initial and a final measured validation success rate.")
            continue
        lines.append(
            f"{label} validation success rate was {initial['success_rate']:.3f} at timestep {initial['timesteps']} "
            f"and {final['success_rate']:.3f} at timestep {final['timesteps']}."
        )
        if float(final["success_rate"]) == 0.0:
            lines.append(f"{label} final fixed-validation success rate is zero.")
        elif float(final["success_rate"]) > float(initial["success_rate"]):
            lines.append(f"{label} final success rate is higher than its own initial success rate on this seed.")
        else:
            lines.append(f"{label} final success rate is not higher than its own initial success rate on this seed.")
        if initial.get("mean_return") is not None and final.get("mean_return") is not None:
            lines.append(
                f"{label} mean validation return changed from {initial['mean_return']:.4f} to {final['mean_return']:.4f}. "
                "A return change without a goal success is not navigation success."
            )
    bc_final = _final_evaluation(bc)
    scratch_final = _final_evaluation(scratch)
    if bc_final and scratch_final and bc_final.get("success_rate") is not None and scratch_final.get("success_rate") is not None:
        if float(bc_final["success_rate"]) == float(scratch_final["success_rate"]) == 0.0:
            lines.append(
                "Neither approach produced a successful goal in the final fixed validation set. "
                "This budget does not show successful navigation learning for either initialization."
            )
        elif float(bc_final["success_rate"]) > float(scratch_final["success_rate"]):
            lines.append("On this seed, the BC-initialized final success rate is higher than the scratch final success rate.")
        elif float(scratch_final["success_rate"]) > float(bc_final["success_rate"]):
            lines.append("On this seed, the scratch final success rate is higher than the BC-initialized final success rate.")
        else:
            lines.append("On this seed, the two final validation success rates are equal.")
    lines.append("This single matched seed does not establish statistical significance or general superiority.")
    for label, run in (("BC initialization", bc), ("Scratch PPO", scratch)):
        updates = run["updates"]
        if not updates:
            lines.append(f"{label} has no recorded PPO updates.")
            continue
        kl_values = [
            float(update["metrics"]["train/approx_kl"])
            for update in updates
            if "train/approx_kl" in (update.get("metrics") or {})
        ]
        variance = [
            float(update["metrics"]["train/explained_variance"])
            for update in updates
            if "train/explained_variance" in (update.get("metrics") or {}) and _finite(update["metrics"]["train/explained_variance"])
        ]
        if kl_values:
            lines.append(f"{label} maximum approximate KL was {max(kl_values):.6f}.")
        if variance:
            lines.append(
                f"{label} explained variance ranged from {min(variance):.4f} to {max(variance):.4f}; "
                f"the last recorded value was {variance[-1]:.4f}."
            )
        completed = [episode for episode in run["episodes"] if episode.get("status") == "completed"]
        if completed:
            timeouts = sum(1 for episode in completed if episode.get("timeout"))
            lines.append(
                f"{label} completed {len(completed)} training episodes, of which {timeouts} timed out, "
                f"{sum(1 for episode in completed if episode.get('success'))} reached the goal, and "
                f"{sum(1 for episode in completed if episode.get('collision'))} collided."
            )
        partial = [episode for episode in run["episodes"] if episode.get("status") == "partial"]
        if partial:
            lines.append(
                f"{label} recorded {len(partial)} partial training episodes. Partial episodes are not counted as timeouts."
            )
        fraction = (run["summary"].get("action_summary") or {}).get("saturated_action_fraction")
        if fraction is not None:
            lines.append(f"{label} fraction of training steps with an action component at least 0.99 in absolute value was {float(fraction):.3f}.")
        if run["summary"].get("losses_finite") is False or run["summary"].get("gradients_finite") is False:
            lines.append(f"{label} recorded a non-finite loss or gradient.")
    lines.append(
        "If neither method reaches goals, the useful next step is a separate multi-seed diagnostic or a later phase. "
        "This report does not start that work."
    )
    return lines


def _run_paragraph(label, run):
    summary = run["summary"]
    if summary.get("status") == "missing":
        return f"{label}: no summary.json was found. Metrics from this run were not measured."
    return (
        f"{label} status `{summary.get('status')}`, trained timesteps {summary.get('trained_timesteps')}, "
        f"requested {summary.get('requested_timesteps')}, rollouts {summary.get('rollouts_completed')}, "
        f"n_updates {summary.get('n_updates')}, gradient steps {summary.get('gradient_steps')}, "
        f"training hours {_hours(summary.get('training_wall_time_s'))}, "
        f"evaluation hours {_hours(summary.get('evaluation_wall_time_s'))}. "
        f"Completed episodes {summary.get('completed_episodes')}, partial episodes {summary.get('partial_episodes')}, "
        f"successes {summary.get('successes')}, collisions {summary.get('collisions')}, timeouts {summary.get('timeouts')}. "
        f"Uncommitted steps {summary.get('uncommitted_steps')}, orphaned steps {summary.get('orphaned_steps')}."
    )


def _validation_paragraph(label, run):
    if not run["evaluations"]:
        return f"{label}: no fixed-validation records were measured."
    parts = []
    for record in sorted(run["evaluations"], key=lambda item: int(item.get("timesteps") or 0)):
        parts.append(
            f"timestep {record.get('timesteps')}: success {record.get('success_rate')}, "
            f"collision {record.get('collision_rate')}, timeout {record.get('timeout_rate')}, "
            f"mean return {record.get('mean_return')}"
        )
    return f"{label} measured validation points: " + "; ".join(parts) + "."


def _evaluation_at_start(run):
    for record in run["evaluations"]:
        if int(record.get("timesteps", -1)) == 0:
            return record
    return None


def _final_evaluation(run):
    if not run["evaluations"]:
        return None
    return max(run["evaluations"], key=lambda item: int(item.get("timesteps") or 0))


def _smoke_context():
    path = OFFICIAL_SMOKE_CHECKPOINT.parent / "summary.json"
    if not path.is_file():
        return "The Phase 6G.2 summary.json was not available while writing this report."
    summary = json.loads(path.read_text(encoding="utf-8"))
    return (
        f"Recorded Phase 6G.2 summary: timesteps {summary.get('environment_timesteps')}, "
        f"rollouts {summary.get('rollouts_completed')}, n_updates {summary.get('n_updates')}, "
        f"gradient steps {summary.get('gradient_steps')}, successes {summary.get('successes')}, "
        f"collisions {summary.get('collisions')}, timeouts {summary.get('timeouts')}, "
        f"partial episodes {summary.get('partial_episodes')}, n_epochs {summary.get('n_epochs')}."
    )


def _source_hashes():
    from navigation.jackal_ppo_diagnostic import (
        OFFICIAL_BC_CHECKPOINT,
        OFFICIAL_CHECKPOINT,
        OFFICIAL_SMOKE_CHECKPOINT,
    )
    items = {
        "phase_6g1": OFFICIAL_CHECKPOINT,
        "phase_6f2": OFFICIAL_BC_CHECKPOINT,
        "phase_6g2": OFFICIAL_SMOKE_CHECKPOINT,
    }
    expected = {
        "phase_6g1": OFFICIAL_CHECKPOINT_SHA256,
        "phase_6f2": OFFICIAL_BC_SHA256,
        "phase_6g2": OFFICIAL_SMOKE_SHA256,
    }
    report = {}
    for label, path in items.items():
        digest = file_sha256(path) if Path(path).is_file() else None
        report[label] = {"sha256": digest, "unchanged": digest == expected[label]}
    return report


def _trailing_mean(values, window):
    chunk = values[-int(window):]
    if not chunk:
        return None
    return float(np.mean(chunk))


def _hours(seconds):
    if seconds is None:
        return "not measured"
    return f"{float(seconds) / 3600.0:.3f}"


def _finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _read_json(path, default):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
