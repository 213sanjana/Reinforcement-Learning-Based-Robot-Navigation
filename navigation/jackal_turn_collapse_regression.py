"""Offline regression for the recorded Phase 6H.2 Jackal right-turn collapse.

Reads the saved matrix JSON and JSONL evidence. It does not load a policy,
start a simulator, or write into the evidence directory.

Two angular thresholds are separate. A sample is a right-turn command when
``actor_mean[1] < -0.01``. An episode mean is saturated when it is at or
below ``-0.84``. The per-sample bound ``-0.8`` is also counted. On this
matrix every final episode mean meets ``-0.84`` and every final sample is
right of ``-0.01``, while some final samples are not below ``-0.8``.

``progress_path_ratio`` is net goal-distance reduction divided by the aligned
snapshot path, with negative progress clipped to zero. It is not SPL.
``aligned_snapshot_path_m`` and ``evaluator_pose_path_m`` stay separate.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping


class RegressionError(ValueError):
    """Saved matrix evidence is missing, malformed, or fails a regression check."""


SIGN_THRESHOLD = 0.01
SAMPLE_SATURATION_THRESHOLD = -0.8
EPISODE_MEAN_SATURATION_THRESHOLD = -0.84
FINAL_PROGRESS_PATH_RATIO_MAX = 0.21
PROGRESS_EXCEPTION_SCENARIO = "heading_2.7"
INITIAL_OPEN_SCENARIOS = (
    "anchor",
    "distance_1p5",
    "distance_2p5",
    "translation_south",
    "open_south",
)
INITIAL_OPEN_RATIO_MIN = 0.988
OBSTACLE_SCENARIOS = (
    "obstacle_around",
    "obstacle_west_east",
    "obstacle_south_north",
)
FINAL_OBSTACLE_RATIO_MAX = 0.08
INITIAL_OBSTACLE_RATIO_MIN = 0.69
HORIZON_STEPS = 300
EXPECTED_EPISODES = 52
EXPECTED_PER_POLICY = 26
EXPECTED_INITIAL_ROWS = 3461
EXPECTED_FINAL_ROWS = 5878
POLICIES = ("initial", "final")
SEEDS = (642, 643)
PILOT_ORDER = (
    ("initial", 642),
    ("final", 642),
    ("final", 643),
    ("initial", 643),
)
EXPECTED_SCENARIOS = (
    "anchor",
    "distance_1p5",
    "distance_2p5",
    "translation_south",
    "translation_north",
    "heading_-2.7",
    "heading_-0.4",
    "heading_0.4",
    "heading_2.7",
    "open_south",
    "obstacle_around",
    "obstacle_west_east",
    "obstacle_south_north",
)
BOTH_SEED_LOSS_SCENARIOS = (
    "distance_1p5",
    "distance_2p5",
    "translation_north",
    "open_south",
)
REQUIRED_FILES = (
    "manifest.json",
    "summary.json",
    "analysis.json",
    "episodes.jsonl",
    "telemetry.jsonl",
)
_NUMBER_TOLERANCE = 1e-9


@dataclass(frozen=True)
class EpisodeRecord:
    """One matrix episode after the JSON records have been cross-checked."""

    episode_id: str
    policy: str
    scenario: str
    seed: int
    success: bool
    collision: bool
    timeout: bool
    termination_reason: str
    control_steps: int
    net_progress_m: float
    aligned_snapshot_path_m: float
    evaluator_pose_path_m: float
    progress_path_ratio: float
    analysis_angular_mean: float
    analysis_right_action_fraction: float


@dataclass(frozen=True)
class MatrixLoad:
    """Structurally valid matrix records, before the historical contract."""

    root: Path
    training_steps: int
    instrumentation_gate_passed: bool
    episodes: tuple[EpisodeRecord, ...]


@dataclass(frozen=True)
class InventoryResult:
    episode_count: int
    initial_episodes: int
    final_episodes: int
    training_steps: int
    instrumentation_gate_passed: bool
    episode_ids: tuple[str, ...]


@dataclass
class EpisodeTelemetry:
    """Angular and terminal-path totals for one episode. Rows are not retained."""

    rows: int = 0
    angular_sum: float = 0.0
    right_sign_rows: int = 0
    below_sample_saturation_rows: int = 0
    min_angular: float = math.inf
    max_angular: float = -math.inf
    terminal_step: int = 0
    terminal_net_progress_m: float = 0.0
    terminal_aligned_path_m: float = 0.0
    steps: set[int] | None = None

    def __post_init__(self) -> None:
        if self.steps is None:
            self.steps = set()

    @property
    def angular_mean(self) -> float:
        if self.rows == 0:
            raise RegressionError("cannot average an episode with no telemetry rows")
        return self.angular_sum / self.rows

    @property
    def right_action_fraction(self) -> float:
        if self.rows == 0:
            raise RegressionError("cannot score an episode with no telemetry rows")
        return self.right_sign_rows / self.rows


@dataclass(frozen=True)
class TelemetryResult:
    by_episode: Mapping[str, EpisodeTelemetry]
    rows_by_policy: Mapping[str, int]
    angular_sum_by_policy: Mapping[str, float]

    @property
    def recomputed_angular_mean_initial(self) -> float:
        return self.angular_sum_by_policy["initial"] / self.rows_by_policy["initial"]

    @property
    def recomputed_angular_mean_final(self) -> float:
        return self.angular_sum_by_policy["final"] / self.rows_by_policy["final"]


@dataclass(frozen=True)
class SaturationResult:
    telemetry_rows_initial: int
    telemetry_rows_final: int
    recomputed_angular_mean_initial: float
    recomputed_angular_mean_final: float
    sign_threshold: float
    sample_saturation_threshold: float
    episode_mean_saturation_threshold: float
    every_final_right_action_fraction_is_one: bool
    every_final_episode_mean_within_saturation: bool
    every_final_sample_below_sign_threshold: bool
    every_final_row_below_sample_saturation: bool
    final_rows_below_sample_saturation: int
    final_rows_not_below_sample_saturation: int
    least_negative_final_angular: float


@dataclass(frozen=True)
class ProgressResult:
    final_episodes_subject_to_ratio_cap: int
    heading_exception_episodes: int
    initial_open_episodes: int
    translation_north_final_net_progress_m: tuple[float, ...]
    translation_north_final_ratios: tuple[float, ...]
    paths_differ_on_at_least_one_episode: bool
    progress_path_ratio_is_spl: bool


@dataclass(frozen=True)
class ObstacleResult:
    episode_count: int
    final_ratio_max: float
    initial_ratio_min: float


@dataclass(frozen=True)
class ComparisonResult:
    scenario_count: int
    pairs: int
    final_does_not_exceed_initial: bool
    both_seed_loss_scenarios: tuple[str, ...]


@dataclass(frozen=True)
class RegressionResult:
    inventory: InventoryResult
    saturation: SaturationResult
    progress: ProgressResult
    obstacles: ObstacleResult
    comparison: ComparisonResult


def default_run_directory() -> Path:
    """Return the saved Phase 6H.2 matrix directory."""

    return (
        Path(__file__).resolve().parents[1]
        / "artifacts/ppo/jackal_6h/diagnostics/run_20261026_172838UTC_6h2_matrix"
    )


def run_regression(root: Path | None = None) -> RegressionResult:
    """Cross-check the matrix and return the five regression results.

    Raises RegressionError when evidence is missing, malformed, or the
    recorded collapse checks fail. A final sample that is not below -0.8
    is counted in the saturation result; episode-mean saturation and the
    -0.01 sign split remain the pass conditions for that check.
    """

    loaded = load_matrix(root or default_run_directory())
    inventory = validate_inventory(loaded)
    telemetry = stream_telemetry(
        loaded.root / "telemetry.jsonl",
        {episode.episode_id: episode.policy for episode in loaded.episodes},
    )
    _require_telemetry_coverage(loaded.episodes, telemetry)
    saturation = evaluate_saturation(loaded.episodes, telemetry)
    progress = evaluate_progress(loaded.episodes)
    obstacles = evaluate_obstacles(loaded.episodes)
    comparison = evaluate_comparison(loaded.episodes)
    return RegressionResult(
        inventory=inventory,
        saturation=saturation,
        progress=progress,
        obstacles=obstacles,
        comparison=comparison,
    )


def load_matrix(root: Path) -> MatrixLoad:
    """Load and cross-check manifest, summary, analysis, and episodes.jsonl."""

    root = Path(root)
    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            raise RegressionError(f"missing evidence file: {name}")
    manifest = _read_object(root / "manifest.json")
    summary = _read_object(root / "summary.json")
    analysis = _read_object(root / "analysis.json")
    jsonl_rows = _read_jsonl_objects(root / "episodes.jsonl")
    order = _episode_order(manifest)
    identities = [_identity(item, f"manifest.json episode_order[{index}]") for index, item in enumerate(order)]
    _reject_duplicate_ids(episode_id for episode_id, _, _, _ in identities)
    training_steps = _integer(summary.get("training_steps"), "summary.json training_steps")
    gate = summary_gate(analysis)
    summary_rows = _episode_list(summary, "summary.json")
    analysis_rows = _episode_list(analysis, "analysis.json")
    _require_same_order(
        [row.get("episode_id") for row in summary_rows],
        [episode_id for episode_id, _, _, _ in identities],
        "summary.json episodes",
    )
    _require_same_order(
        [row.get("episode_id") for row in analysis_rows],
        [episode_id for episode_id, _, _, _ in identities],
        "analysis.json episodes",
    )
    _require_same_order(
        [row.get("episode_id") for row in jsonl_rows],
        [episode_id for episode_id, _, _, _ in identities],
        "episodes.jsonl",
    )
    episodes = tuple(
        _merge_episode(identity, summary_row, analysis_row, jsonl_row)
        for identity, summary_row, analysis_row, jsonl_row in zip(
            identities, summary_rows, analysis_rows, jsonl_rows
        )
    )
    return MatrixLoad(
        root=root,
        training_steps=training_steps,
        instrumentation_gate_passed=gate,
        episodes=episodes,
    )


def summary_gate(analysis: Mapping[str, object]) -> bool:
    """Return the derived instrumentation flag, which must be a real boolean."""

    value = analysis.get("instrumentation_gate_passed")
    if value is not True and value is not False:
        raise RegressionError("analysis.json instrumentation_gate_passed must be boolean")
    return bool(value)


def validate_inventory(loaded: MatrixLoad) -> InventoryResult:
    """Apply the 52-episode historical inventory contract."""

    if loaded.training_steps != 0:
        raise RegressionError(
            f"summary.json training_steps is {loaded.training_steps}, expected 0"
        )
    if loaded.instrumentation_gate_passed is not True:
        raise RegressionError("analysis.json instrumentation_gate_passed is not true")
    episodes = loaded.episodes
    if len(episodes) != EXPECTED_EPISODES:
        raise RegressionError(
            f"expected {EXPECTED_EPISODES} episodes, found {len(episodes)}"
        )
    counts = {policy: sum(episode.policy == policy for episode in episodes) for policy in POLICIES}
    for policy in POLICIES:
        if counts[policy] != EXPECTED_PER_POLICY:
            raise RegressionError(
                f"expected {EXPECTED_PER_POLICY} {policy} episodes, found {counts[policy]}"
            )
    expected_ids = [
        f"{scenario}_{policy}_{seed}"
        for scenario in EXPECTED_SCENARIOS
        for policy, seed in PILOT_ORDER
    ]
    found_ids = [episode.episode_id for episode in episodes]
    if found_ids != expected_ids:
        raise RegressionError(
            "manifest episode_order does not match the 13-scenario pilot order"
        )
    for episode, expected_id in zip(episodes, expected_ids):
        scenario, policy, seed_text = _split_expected_id(expected_id)
        if (episode.scenario, episode.policy, episode.seed) != (scenario, policy, int(seed_text)):
            raise RegressionError(
                f"{episode.episode_id}: scenario/policy/seed does not match {expected_id}"
            )
        if not episode.success and not episode.collision and not episode.timeout:
            raise RegressionError(f"{episode.episode_id}: episode has no outcome flag")
    return InventoryResult(
        episode_count=len(episodes),
        initial_episodes=counts["initial"],
        final_episodes=counts["final"],
        training_steps=loaded.training_steps,
        instrumentation_gate_passed=True,
        episode_ids=tuple(found_ids),
    )


def stream_telemetry(path: Path, expected_policy_by_episode: Mapping[str, str]) -> TelemetryResult:
    """Stream telemetry.jsonl without retaining the raw rows."""

    path = Path(path)
    if not path.is_file():
        raise RegressionError(f"missing evidence file: {path.name}")
    by_episode = {episode_id: EpisodeTelemetry() for episode_id in expected_policy_by_episode}
    rows_by_policy = {policy: 0 for policy in POLICIES}
    angular_sum_by_policy = {policy: 0.0 for policy in POLICIES}
    for line_number, row in _iter_json_objects(path):
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or episode_id not in by_episode:
            raise RegressionError(
                f"{path.name} line {line_number}: unknown episode_id {episode_id!r}"
            )
        policy = row.get("policy")
        expected_policy = expected_policy_by_episode[episode_id]
        if policy != expected_policy:
            raise RegressionError(
                f"{path.name} line {line_number}: episode {episode_id} policy {policy!r} "
                f"does not match {expected_policy}"
            )
        if policy not in rows_by_policy:
            raise RegressionError(
                f"{path.name} line {line_number}: episode {episode_id} has unknown policy {policy!r}"
            )
        angular = _angular_component(row.get("actor_mean"), f"{path.name} line {line_number}")
        step = _integer(row.get("step"), f"{path.name} line {line_number} step")
        net_progress = _finite(
            row.get("net_progress_m"),
            f"{path.name} line {line_number} net_progress_m",
        )
        aligned_path = _finite(
            row.get("cumulative_path_length_m"),
            f"{path.name} line {line_number} cumulative_path_length_m",
        )
        episode = by_episode[episode_id]
        if step in episode.steps:
            raise RegressionError(
                f"{path.name} line {line_number}: duplicate step {step} for episode {episode_id}"
            )
        episode.steps.add(step)
        episode.rows += 1
        episode.angular_sum += angular
        if angular < -SIGN_THRESHOLD:
            episode.right_sign_rows += 1
        if angular < SAMPLE_SATURATION_THRESHOLD:
            episode.below_sample_saturation_rows += 1
        episode.min_angular = min(episode.min_angular, angular)
        episode.max_angular = max(episode.max_angular, angular)
        if step >= episode.terminal_step:
            episode.terminal_step = step
            episode.terminal_net_progress_m = net_progress
            episode.terminal_aligned_path_m = aligned_path
        rows_by_policy[policy] += 1
        angular_sum_by_policy[policy] += angular
    for episode_id, episode in by_episode.items():
        if episode.rows == 0:
            raise RegressionError(f"{episode_id}: no telemetry rows")
    return TelemetryResult(
        by_episode=by_episode,
        rows_by_policy=rows_by_policy,
        angular_sum_by_policy=angular_sum_by_policy,
    )


def evaluate_saturation(episodes: Iterable[EpisodeRecord], telemetry: TelemetryResult) -> SaturationResult:
    """Compare derived episode means with telemetry recomputed from actor_mean."""

    if telemetry.rows_by_policy["initial"] != EXPECTED_INITIAL_ROWS:
        raise RegressionError(
            "initial telemetry rows are "
            f"{telemetry.rows_by_policy['initial']}, expected {EXPECTED_INITIAL_ROWS}"
        )
    if telemetry.rows_by_policy["final"] != EXPECTED_FINAL_ROWS:
        raise RegressionError(
            "final telemetry rows are "
            f"{telemetry.rows_by_policy['final']}, expected {EXPECTED_FINAL_ROWS}"
        )
    least_negative = -math.inf
    below_saturation = 0
    final_episodes = 0
    for episode in episodes:
        measured = telemetry.by_episode[episode.episode_id]
        if not _close(measured.angular_mean, episode.analysis_angular_mean):
            raise RegressionError(
                f"{episode.episode_id}: telemetry angular mean {measured.angular_mean} "
                f"does not match analysis actor_angular.mean {episode.analysis_angular_mean}"
            )
        if not _close(measured.right_action_fraction, episode.analysis_right_action_fraction):
            raise RegressionError(
                f"{episode.episode_id}: telemetry right-turn fraction {measured.right_action_fraction} "
                f"does not match analysis right_action_fraction {episode.analysis_right_action_fraction}"
            )
        if episode.policy != "final":
            continue
        final_episodes += 1
        if episode.analysis_right_action_fraction != 1.0 or measured.right_sign_rows != measured.rows:
            raise RegressionError(
                f"{episode.episode_id}: final right_action_fraction is "
                f"{episode.analysis_right_action_fraction}, expected 1.0 at sign threshold "
                f"{SIGN_THRESHOLD}"
            )
        if episode.analysis_angular_mean > EPISODE_MEAN_SATURATION_THRESHOLD:
            raise RegressionError(
                f"{episode.episode_id}: final actor_angular.mean {episode.analysis_angular_mean} "
                f"is above {EPISODE_MEAN_SATURATION_THRESHOLD}"
            )
        if measured.max_angular >= -SIGN_THRESHOLD:
            raise RegressionError(
                f"{episode.episode_id}: final sample {measured.max_angular} is not below "
                f"the sign threshold {-SIGN_THRESHOLD}"
            )
        least_negative = max(least_negative, measured.max_angular)
        below_saturation += measured.below_sample_saturation_rows
    if final_episodes != EXPECTED_PER_POLICY:
        raise RegressionError(f"expected {EXPECTED_PER_POLICY} final episodes, found {final_episodes}")
    not_below = EXPECTED_FINAL_ROWS - below_saturation
    return SaturationResult(
        telemetry_rows_initial=telemetry.rows_by_policy["initial"],
        telemetry_rows_final=telemetry.rows_by_policy["final"],
        recomputed_angular_mean_initial=telemetry.recomputed_angular_mean_initial,
        recomputed_angular_mean_final=telemetry.recomputed_angular_mean_final,
        sign_threshold=SIGN_THRESHOLD,
        sample_saturation_threshold=SAMPLE_SATURATION_THRESHOLD,
        episode_mean_saturation_threshold=EPISODE_MEAN_SATURATION_THRESHOLD,
        every_final_right_action_fraction_is_one=True,
        every_final_episode_mean_within_saturation=True,
        every_final_sample_below_sign_threshold=True,
        every_final_row_below_sample_saturation=not_below == 0,
        final_rows_below_sample_saturation=below_saturation,
        final_rows_not_below_sample_saturation=not_below,
        least_negative_final_angular=least_negative,
    )


def evaluate_progress(episodes: Iterable[EpisodeRecord]) -> ProgressResult:
    """Check path use. The ratio cap does not apply to heading_2.7."""

    subject_to_cap = 0
    heading_exceptions = 0
    initial_open = 0
    north_progress: list[tuple[int, float]] = []
    north_ratios: list[tuple[int, float]] = []
    paths_differ = False
    for episode in episodes:
        if not _close(
            episode.progress_path_ratio,
            _ratio(episode.net_progress_m, episode.aligned_snapshot_path_m),
        ):
            raise RegressionError(
                f"{episode.episode_id}: progress_path_ratio {episode.progress_path_ratio} "
                "does not match max(0, net progress) / aligned_snapshot_path_m"
            )
        if not _close(episode.aligned_snapshot_path_m, episode.evaluator_pose_path_m):
            paths_differ = True
        if episode.policy == "final" and episode.scenario == PROGRESS_EXCEPTION_SCENARIO:
            heading_exceptions += 1
        elif episode.policy == "final":
            subject_to_cap += 1
            if episode.progress_path_ratio > FINAL_PROGRESS_PATH_RATIO_MAX:
                raise RegressionError(
                    f"{episode.episode_id}: progress_path_ratio {episode.progress_path_ratio} "
                    f"is above {FINAL_PROGRESS_PATH_RATIO_MAX}"
                )
        if episode.policy == "initial" and episode.scenario in INITIAL_OPEN_SCENARIOS:
            initial_open += 1
            if episode.progress_path_ratio < INITIAL_OPEN_RATIO_MIN:
                raise RegressionError(
                    f"{episode.episode_id}: initial progress_path_ratio {episode.progress_path_ratio} "
                    f"is below {INITIAL_OPEN_RATIO_MIN}"
                )
        if episode.scenario == "translation_north" and episode.policy == "final":
            if episode.net_progress_m >= 0 or episode.progress_path_ratio != 0.0:
                raise RegressionError(
                    f"{episode.episode_id}: translation_north final net progress "
                    f"{episode.net_progress_m} and ratio {episode.progress_path_ratio} "
                    "do not show negative progress with ratio 0"
                )
            north_progress.append((episode.seed, episode.net_progress_m))
            north_ratios.append((episode.seed, episode.progress_path_ratio))
    if heading_exceptions != len(SEEDS):
        raise RegressionError(
            f"expected {len(SEEDS)} final {PROGRESS_EXCEPTION_SCENARIO} episodes, "
            f"found {heading_exceptions}"
        )
    if initial_open != len(INITIAL_OPEN_SCENARIOS) * len(SEEDS):
        raise RegressionError(
            f"expected {len(INITIAL_OPEN_SCENARIOS) * len(SEEDS)} initial open episodes, "
            f"found {initial_open}"
        )
    if len(north_progress) != len(SEEDS):
        raise RegressionError(
            f"expected {len(SEEDS)} final translation_north episodes, found {len(north_progress)}"
        )
    north_progress.sort()
    north_ratios.sort()
    return ProgressResult(
        final_episodes_subject_to_ratio_cap=subject_to_cap,
        heading_exception_episodes=heading_exceptions,
        initial_open_episodes=initial_open,
        translation_north_final_net_progress_m=tuple(value for _, value in north_progress),
        translation_north_final_ratios=tuple(value for _, value in north_ratios),
        paths_differ_on_at_least_one_episode=paths_differ,
        progress_path_ratio_is_spl=False,
    )


def evaluate_obstacles(episodes: Iterable[EpisodeRecord]) -> ObstacleResult:
    """Require horizon timeouts on every obstacle episode and the ratio gap."""

    found = [episode for episode in episodes if episode.scenario in OBSTACLE_SCENARIOS]
    expected = len(OBSTACLE_SCENARIOS) * len(POLICIES) * len(SEEDS)
    if len(found) != expected:
        raise RegressionError(f"expected {expected} obstacle episodes, found {len(found)}")
    keys = {(episode.scenario, episode.policy, episode.seed) for episode in found}
    expected_keys = {
        (scenario, policy, seed)
        for scenario in OBSTACLE_SCENARIOS
        for policy in POLICIES
        for seed in SEEDS
    }
    if keys != expected_keys:
        missing = expected_keys - keys
        raise RegressionError(f"obstacle episode set is incomplete: {sorted(missing)}")
    final_ratios = []
    initial_ratios = []
    for episode in found:
        if episode.timeout is not True or episode.termination_reason != "time_limit":
            raise RegressionError(
                f"{episode.episode_id}: obstacle outcome is {episode.termination_reason}, "
                "expected time_limit"
            )
        if episode.control_steps != HORIZON_STEPS:
            raise RegressionError(
                f"{episode.episode_id}: obstacle control_steps is {episode.control_steps}, "
                f"expected {HORIZON_STEPS}"
            )
        if episode.success or episode.collision:
            raise RegressionError(
                f"{episode.episode_id}: obstacle episode records success or collision"
            )
        if episode.policy == "final":
            final_ratios.append(episode.progress_path_ratio)
        else:
            initial_ratios.append(episode.progress_path_ratio)
    final_max = max(final_ratios)
    initial_min = min(initial_ratios)
    if final_max > FINAL_OBSTACLE_RATIO_MAX:
        raise RegressionError(
            f"final obstacle progress_path_ratio reaches {final_max}, "
            f"above {FINAL_OBSTACLE_RATIO_MAX}"
        )
    if initial_min < INITIAL_OBSTACLE_RATIO_MIN:
        raise RegressionError(
            f"initial obstacle progress_path_ratio falls to {initial_min}, "
            f"below {INITIAL_OBSTACLE_RATIO_MIN}"
        )
    return ObstacleResult(
        episode_count=len(found),
        final_ratio_max=final_max,
        initial_ratio_min=initial_min,
    )


def evaluate_comparison(episodes: Iterable[EpisodeRecord]) -> ComparisonResult:
    """Pair initial and final by scenario and seed."""

    pairs: dict[tuple[str, int], dict[str, EpisodeRecord]] = {}
    for episode in episodes:
        pairs.setdefault((episode.scenario, episode.seed), {})[episode.policy] = episode
    if len(pairs) != len(EXPECTED_SCENARIOS) * len(SEEDS):
        raise RegressionError(f"expected {len(EXPECTED_SCENARIOS) * len(SEEDS)} pairs, found {len(pairs)}")
    losses = []
    for scenario in EXPECTED_SCENARIOS:
        scenario_losses = []
        for seed in SEEDS:
            pair = pairs.get((scenario, seed))
            if pair is None or set(pair) != set(POLICIES):
                raise RegressionError(f"missing initial/final pair for {scenario} seed {seed}")
            initial = pair["initial"]
            final = pair["final"]
            if int(final.success) > int(initial.success):
                raise RegressionError(
                    f"{scenario} seed {seed}: final success exceeds initial success"
                )
            scenario_losses.append(initial.success and not final.success)
        if all(scenario_losses):
            losses.append(scenario)
    if tuple(losses) != BOTH_SEED_LOSS_SCENARIOS:
        raise RegressionError(
            "both-seed final losses are "
            f"{tuple(losses)}, expected {BOTH_SEED_LOSS_SCENARIOS}"
        )
    return ComparisonResult(
        scenario_count=len(EXPECTED_SCENARIOS),
        pairs=len(pairs),
        final_does_not_exceed_initial=True,
        both_seed_loss_scenarios=tuple(losses),
    )


def _require_telemetry_coverage(episodes: Iterable[EpisodeRecord], telemetry: TelemetryResult) -> None:
    for episode in episodes:
        measured = telemetry.by_episode.get(episode.episode_id)
        if measured is None or measured.rows == 0:
            raise RegressionError(f"{episode.episode_id}: no telemetry rows")
        if measured.rows != episode.control_steps:
            raise RegressionError(
                f"{episode.episode_id}: telemetry rows {measured.rows} "
                f"do not match control_steps {episode.control_steps}"
            )
        expected_steps = set(range(1, measured.rows + 1))
        if measured.steps != expected_steps:
            raise RegressionError(f"{episode.episode_id}: telemetry steps are not 1..{measured.rows}")
        if not _close(measured.terminal_net_progress_m, episode.net_progress_m):
            raise RegressionError(
                f"{episode.episode_id}: terminal telemetry net_progress_m "
                f"{measured.terminal_net_progress_m} does not match {episode.net_progress_m}"
            )
        if not _close(measured.terminal_aligned_path_m, episode.aligned_snapshot_path_m):
            raise RegressionError(
                f"{episode.episode_id}: terminal telemetry path {measured.terminal_aligned_path_m} "
                f"does not match aligned_snapshot_path_m {episode.aligned_snapshot_path_m}"
            )


def _merge_episode(
    identity: tuple[str, str, str, int],
    summary_row: Mapping[str, object],
    analysis_row: Mapping[str, object],
    jsonl_row: Mapping[str, object],
) -> EpisodeRecord:
    episode_id, policy, scenario, seed = identity
    for source, row in (
        ("summary.json", summary_row),
        ("analysis.json", analysis_row),
        ("episodes.jsonl", jsonl_row),
    ):
        _require_identity(row, episode_id, policy, scenario, seed, source)
    summary_eval = _object(summary_row.get("evaluator"), f"summary.json {episode_id} evaluator")
    jsonl_eval = _object(jsonl_row.get("evaluator"), f"episodes.jsonl {episode_id} evaluator")
    outcome = _outcome(jsonl_eval, f"episodes.jsonl {episode_id}")
    _require_same_outcome(summary_eval, outcome, f"summary.json {episode_id}")
    _require_same_outcome(analysis_row, outcome, f"analysis.json {episode_id}")
    net_progress = _same_number(
        (
            ("summary.json", summary_row.get("net_progress_m")),
            ("analysis.json", analysis_row.get("net_progress_m")),
            ("episodes.jsonl", jsonl_row.get("net_progress_m")),
        ),
        f"{episode_id} net_progress_m",
    )
    aligned_path = _same_number(
        (
            ("summary.json", summary_row.get("path_length_m")),
            ("analysis.json", analysis_row.get("aligned_snapshot_path_m")),
            ("episodes.jsonl", jsonl_row.get("path_length_m")),
        ),
        f"{episode_id} aligned path",
    )
    evaluator_path = _same_number(
        (
            ("summary.json evaluator", summary_eval.get("path_length_m")),
            ("analysis.json", analysis_row.get("evaluator_pose_path_m")),
            ("episodes.jsonl evaluator", jsonl_eval.get("path_length_m")),
        ),
        f"{episode_id} evaluator path",
    )
    ratio = _same_number(
        (
            ("summary.json", summary_row.get("path_efficiency")),
            ("analysis.json", analysis_row.get("progress_path_ratio")),
            ("episodes.jsonl", jsonl_row.get("path_efficiency")),
        ),
        f"{episode_id} progress_path_ratio",
    )
    steering = _object(analysis_row.get("steering"), f"analysis.json {episode_id} steering")
    actor = _object(steering.get("actor_angular"), f"analysis.json {episode_id} actor_angular")
    if analysis_row.get("valid") is not True:
        raise RegressionError(f"analysis.json {episode_id}: valid is not true")
    return EpisodeRecord(
        episode_id=episode_id,
        policy=policy,
        scenario=scenario,
        seed=seed,
        success=outcome[0],
        collision=outcome[1],
        timeout=outcome[2],
        termination_reason=outcome[3],
        control_steps=outcome[4],
        net_progress_m=net_progress,
        aligned_snapshot_path_m=aligned_path,
        evaluator_pose_path_m=evaluator_path,
        progress_path_ratio=ratio,
        analysis_angular_mean=_finite(actor.get("mean"), f"analysis.json {episode_id} actor_angular.mean"),
        analysis_right_action_fraction=_finite(
            steering.get("right_action_fraction"),
            f"analysis.json {episode_id} right_action_fraction",
        ),
    )


def _outcome(row: Mapping[str, object], context: str) -> tuple[bool, bool, bool, str, int]:
    success = _boolean(row.get("success"), f"{context} success")
    collision = _boolean(row.get("collision"), f"{context} collision")
    timeout = _boolean(row.get("timeout"), f"{context} timeout")
    if sum((success, collision, timeout)) != 1:
        raise RegressionError(f"{context}: expected exactly one of success, collision, timeout")
    reason = row.get("termination_reason")
    if not isinstance(reason, str) or not reason:
        raise RegressionError(f"{context}: termination_reason is missing")
    steps = _integer(row.get("control_steps"), f"{context} control_steps")
    if steps < 1:
        raise RegressionError(f"{context}: control_steps must be positive")
    return success, collision, timeout, reason, steps


def _require_same_outcome(
    row: Mapping[str, object],
    outcome: tuple[bool, bool, bool, str, int],
    context: str,
) -> None:
    other = _outcome(row, context)
    if other != outcome:
        raise RegressionError(f"{context}: outcome does not match episodes.jsonl")


def _identity(row: Mapping[str, object], context: str) -> tuple[str, str, str, int]:
    episode_id = row.get("episode_id")
    policy = row.get("policy")
    scenario = row.get("scenario")
    seed = row.get("seed")
    if not isinstance(episode_id, str) or not episode_id:
        raise RegressionError(f"{context}: episode_id is missing")
    if policy not in POLICIES:
        raise RegressionError(f"{context}: policy {policy!r} is not initial or final")
    if not isinstance(scenario, str) or not scenario:
        raise RegressionError(f"{context}: scenario is missing")
    return episode_id, policy, scenario, _integer(seed, f"{context} seed")


def _require_identity(
    row: Mapping[str, object],
    episode_id: str,
    policy: str,
    scenario: str,
    seed: int,
    source: str,
) -> None:
    found = _identity(row, f"{source} {episode_id}")
    if found != (episode_id, policy, scenario, seed):
        raise RegressionError(
            f"{source} {episode_id}: identity {found} does not match manifest {episode_id}"
        )


def _episode_order(manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    order = manifest.get("episode_order")
    if not isinstance(order, list) or not order:
        raise RegressionError("manifest.json episode_order must be a non-empty list")
    rows = []
    for index, item in enumerate(order):
        rows.append(_object(item, f"manifest.json episode_order[{index}]"))
    return rows


def _episode_list(document: Mapping[str, object], source: str) -> list[Mapping[str, object]]:
    rows = document.get("episodes")
    if not isinstance(rows, list):
        raise RegressionError(f"{source} episodes must be a list")
    return [_object(item, f"{source} episodes[{index}]") for index, item in enumerate(rows)]


def _reject_duplicate_ids(episode_ids: Iterable[str]) -> None:
    seen: set[str] = set()
    for episode_id in episode_ids:
        if episode_id in seen:
            raise RegressionError(f"duplicate episode ID {episode_id}")
        seen.add(episode_id)


def _require_same_order(found: list[object], expected: list[str], source: str) -> None:
    if found != expected:
        raise RegressionError(f"{source} episode order does not match manifest.json episode_order")


def _split_expected_id(episode_id: str) -> tuple[str, str, str]:
    for policy in POLICIES:
        marker = f"_{policy}_"
        if marker in episode_id:
            scenario, seed_text = episode_id.split(marker, 1)
            return scenario, policy, seed_text
    raise RegressionError(f"unrecognized episode ID {episode_id}")


def _ratio(net_progress: float, aligned_path: float) -> float:
    if aligned_path <= 0:
        raise RegressionError(f"aligned path {aligned_path} is not positive")
    return max(0.0, net_progress) / aligned_path


def _read_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RegressionError(f"{path.name}: invalid JSON") from exc
    return _object(value, path.name)


def _read_jsonl_objects(path: Path) -> list[dict[str, object]]:
    return [row for _, row in _iter_json_objects(path)]


def _iter_json_objects(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RegressionError(f"{path.name} line {line_number}: invalid JSON") from exc
            yield line_number, _object(value, f"{path.name} line {line_number}")


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RegressionError(f"{context}: expected a JSON object")
    return value


def _boolean(value: object, context: str) -> bool:
    if value is not True and value is not False:
        raise RegressionError(f"{context}: expected a boolean")
    return bool(value)


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RegressionError(f"{context}: expected an integer")
    return value


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegressionError(f"{context}: expected a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise RegressionError(f"{context}: expected a finite number")
    return number


def _angular_component(value: object, context: str) -> float:
    if not isinstance(value, list) or len(value) != 2:
        raise RegressionError(f"{context}: actor_mean must contain linear and angular values")
    _finite(value[0], f"{context} actor_mean[0]")
    return _finite(value[1], f"{context} actor_mean[1]")


def _same_number(sources: Iterable[tuple[str, object]], context: str) -> float:
    values = [(name, _finite(value, f"{name} {context}")) for name, value in sources]
    anchor_name, anchor = values[0]
    for name, value in values[1:]:
        if not _close(anchor, value):
            raise RegressionError(
                f"{context}: {name} has {value} and {anchor_name} has {anchor}"
            )
    return anchor


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=_NUMBER_TOLERANCE, abs_tol=_NUMBER_TOLERANCE)
