"""Dependency-injected expert collection orchestration.

This module does not create ROS nodes or instantiate a robot.
Callers must explicitly provide a runner factory and exporter.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeOutcome:
    episode_id: int
    status: str
    exported: bool
    output_files: tuple = ()
    error: str = ""




def collect_episode(
    episode,
    runner_factory,
    exporter,
    output_dir,
):
    """Close the runner successfully before exporting a dataset."""

    runner = None
    result = None
    status = "ERROR"
    error = ""
    files = ()
    exported = False

    try:
        runner = runner_factory(episode)

        runner.prepare(
            start=episode.start,
            goal=episode.goal,
        )

        result = runner.run(duration=episode.duration)
        status = result.status

        if status == "GOAL_REACHED":
            if result.rejected_commands != 0:
                status = "REJECTED_COMMANDS"

    except Exception as exc:
        status = "ERROR"
        error = f"{type(exc).__name__}: {exc}"

    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception as exc:
                cleanup_error = (
                    f"Cleanup failure: "
                    f"{type(exc).__name__}: {exc}"
                )

                error = (
                    f"{error}; {cleanup_error}"
                    if error else cleanup_error
                )

                status = "CLEANUP_FAILED"

    if status == "GOAL_REACHED" and result is not None:
        try:
            files = tuple(
                exporter(
                    result,
                    output_dir,
                    episode.episode_id,
                )
            )
            exported = True

        except Exception as exc:
            status = "ERROR"
            error = f"{type(exc).__name__}: {exc}"
            exported = False
            files = ()

    return EpisodeOutcome(
        episode_id=episode.episode_id,
        status=status,
        exported=exported,
        output_files=files,
        error=error,
    )


def collect_plan(
    plan,
    runner_factory,
    exporter,
    *,
    allow_execution=False,
):
    """Require explicit authorization before invoking any runner."""

    if not allow_execution:
        raise RuntimeError(
            "Collection execution is disabled. "
            "No robot commands or simulation resets were issued."
        )

    outcomes = []

    for episode in plan.episodes:
        outcome = collect_episode(
            episode,
            runner_factory,
            exporter,
            plan.output_dir,
        )

        outcomes.append(outcome)

        if not outcome.exported:
            break

    return tuple(outcomes)
