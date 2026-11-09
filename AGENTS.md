# Shared AI coding instructions

## Project and environment
- Purpose: reinforcement-learning-based robot navigation, including GridWorld and Jackal robotics work.
- Verified environment: Ubuntu 22.04, ROS 2 Humble, Gazebo Fortress, Python 3.10.
  Evidence: host OS metadata, `.venv/pyvenv.cfg`, Humble setup file,
  installed `libignition-gazebo6`, and bringup package metadata.
- Use the project's actual configuration and dependencies in `configs/`,
  `requirements/`, and relevant ROS package manifests. Do not substitute
  ROS 2 Jazzy or Gazebo Harmonic. Do not assume README commands are current.

## Bounded, efficient work
- Start with a short plan; inspect existing instructions and only relevant metadata.
- Implement one bounded task at a time. Prefer minimal diffs and modular code.
- Inspect relevant functions and immediate dependencies; avoid dumping large files
  or bulk-reading Python modules, URDFs, Xacros, and world files.
- Run focused, task-permitted tests first; broaden only when justified.
- Do not repeat successful work or regenerate valid evidence without a reason.
- Stop and report blockers instead of looping through speculative fixes.
- Report changed files, commands executed, test results (or tests not run), and
  unresolved issues for each task.
- Never invent test results, performance metrics, file paths, or successful integrations.
- Keep implementation and independent review separate; do not present an
  implementer's own checks as independent review.

## Preservation and phase authorization
- Preserve existing instructions, checkpoints, historical evidence, evidence
  packages, and immutable diagnostic candidates; do not overwrite or regenerate them.
- Follow the active task's phase-specific restrictions and relevant phase documents.
  Obtain explicit authorization before restricted operations or starting another phase.
- Phase 6H Part 1 is a numeric-validation diagnostic correction. Its restrictions
  remain authoritative; these instructions do not authorize production integration
  or any prohibited runtime activity. Historical execution commands are not permission.
- Never run training, inference, ROS 2, Gazebo, simulation, or navigation when the
  task forbids them. Do not bypass restrictions through tests, imports, or helpers
  that trigger those activities. Choose static or offline checks only when permitted.
- Preserve the separate live-authorization gates in `docs/phase6h1_diagnostic_pilot.md`
  and `docs/phase6h2_diagnostic_matrix.md`; do not infer approval from prior runs.
