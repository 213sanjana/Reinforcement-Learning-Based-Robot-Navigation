"""Jackal curriculum over verified free endpoints in the existing box world.

Decisions consume development evaluation only; training outcomes never promote.
An optional path-efficiency gate can also reject a development pass. With the
gate disabled, promotion still depends only on the success rate, collision
rate, dwell, and consecutive-pass count.
"""
from dataclasses import dataclass, asdict, field
import logging
import math
import numpy as np

from navigation.jackal_bc_evaluation import EvalScenario, evaluation_scenarios

_LOG = logging.getLogger(__name__)


def validate_scenario(s):
    if len(s.start) != 3 or len(s.goal) != 2 or not all(math.isfinite(x) for x in (*s.start, *s.goal)):
        raise ValueError('Invalid finite scenario coordinates')
    for x, y in (s.start[:2], s.goal):
        if not (-1.5 <= x <= 4.5 and -2.5 <= y <= 2.5):
            raise ValueError('Endpoint outside tested world bounds')
        if .8 <= x <= 2.2 and -.7 <= y <= .7:
            raise ValueError('Endpoint in inflated obstacle')
    if math.dist(s.start[:2], s.goal) <= .30:
        raise ValueError('Scenario starts inside goal tolerance')
    # All endpoints are in the connected free region around one box.
    return s


def stages():
    def s(name, start, goal):
        return validate_scenario(EvalScenario(name, 'development', tuple(start), tuple(goal), 1, 90., 'Phase 6H development'))
    a = tuple(s(f'A_{i}', (0., -1.8, yaw), (.75, -1.8))
              for i, yaw in enumerate((-.4, 0., .4)))
    b = tuple(s(f'B_{i}', (x, y, yaw), (gx, gy)) for i, (x,y,yaw,gx,gy) in enumerate((
        (-.5,-1.5,0.,1.,-1.5), (2.5,-1.8,2.7,1.,-1.8),
        (-.5,1.5,.4,1.5,1.5), (3.5,1.8,-2.7,2.,1.8))))
    c = tuple(s(f'C_{i}', start, goal) for i,(start,goal) in enumerate((
        ((.3,-1.2,.8),(2.7,-1.2)), ((.3,1.2,-.8),(2.7,1.2)),
        ((.3,-1.2,1.57),(.3,1.2)), ((2.7,1.2,-1.57),(2.7,-1.2)))))
    d = tuple(validate_scenario(v) for v in evaluation_scenarios(5)
              if v.scenario_id in ('open_south','obstacle_around','obstacle_west_east','obstacle_south_north'))
    return (a,b,c,d)


def _finite_metric(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def path_efficiency(start_distance, final_distance, path_length_m):
    """Return ``(efficiency, rejection)`` for net goal progress over path length.

    ``rejection`` is ``None`` when the ratio is a finite measurement. Invalid,
    nonfinite, or nonpositive path length does not produce a ratio. Negative
    net progress stays a finite ratio and fails a positive threshold later.
    """

    start = _finite_metric(start_distance)
    final = _finite_metric(final_distance)
    length = _finite_metric(path_length_m)
    if start is None or start < 0:
        return None, "invalid_start_goal_distance"
    if final is None or final < 0:
        return None, "invalid_final_goal_distance"
    if length is None:
        return None, "invalid_path_length"
    if length <= 0:
        return None, "nonpositive_path_length"
    efficiency = (start - final) / length
    if not math.isfinite(efficiency):
        return None, "nonfinite_efficiency"
    return efficiency, None


def _episode_efficiency(episode):
    if episode.get("odom_coverage_complete") is not True:
        return None, "incomplete_odometry_coverage"
    return path_efficiency(
        episode.get("start_goal_distance_m"),
        episode.get("final_goal_distance_m"),
        episode.get("path_length_odom_m"),
    )


def _efficiency_report(episodes, threshold):
    """Score one development batch.

    Every episode must have valid odometry coverage and valid distance/path
    telemetry. The threshold applies only to successful, collision-free
    episodes. A failed episode with low but valid efficiency does not fail
    the gate. Invalid telemetry on any episode does.
    """

    rows = []
    values = []
    telemetry_valid = True
    successes_pass = True
    for episode in episodes:
        efficiency, reason = _episode_efficiency(episode)
        counts_as_success = bool(episode.get("success")) and not episode.get("collision")
        if reason is None and counts_as_success and efficiency < threshold:
            reason = "below_threshold"
        if reason not in (None, "below_threshold"):
            telemetry_valid = False
        if counts_as_success and reason is not None:
            successes_pass = False
        if efficiency is not None and reason in (None, "below_threshold"):
            values.append(efficiency)
        row = dict(
            scenario_id=episode.get("scenario_id"),
            path_efficiency=efficiency,
            efficiency_gate="pass" if reason is None else "fail",
            rejection=reason,
        )
        rows.append(row)
        _LOG.info(
            "curriculum episode scenario=%s path_efficiency=%s efficiency_gate=%s rejection=%s",
            row["scenario_id"],
            row["path_efficiency"],
            row["efficiency_gate"],
            row["rejection"],
        )
    mean = None if not values else float(sum(values) / len(values))
    return telemetry_valid and successes_pass, rows, mean


@dataclass
class JackalCurriculum:
    level: int = 0
    steps: int = 0
    level_steps: int = 0
    passes: int = 0
    failures: int = 0
    last_decision: int = -1
    history: list = field(default_factory=list)
    min_steps: int = 2048
    min_episodes: int = 12
    required_passes: int = 2
    efficiency_enabled: bool = False
    efficiency_threshold: float | None = 0.5

    def __post_init__(self):
        if (not 0 <= self.level < 4 or min(self.steps,self.level_steps,self.passes,self.failures)<0
                or self.min_steps < 1 or self.min_episodes < 12 or self.required_passes < 2
                or self.last_decision > self.steps):
            raise ValueError('Invalid curriculum state or decision settings')
        if type(self.efficiency_enabled) is not bool:
            raise ValueError('efficiency_enabled must be a bool')
        if self.efficiency_enabled or self.efficiency_threshold is not None:
            if isinstance(self.efficiency_threshold, bool) or not isinstance(self.efficiency_threshold, (int, float)):
                raise ValueError('efficiency threshold must be a real number')
            if not math.isfinite(float(self.efficiency_threshold)) or float(self.efficiency_threshold) <= 0:
                raise ValueError('efficiency threshold must be finite and positive')
            self.efficiency_threshold = float(self.efficiency_threshold)

    def charge(self, steps):
        if not isinstance(steps,int) or steps <= 0:
            raise ValueError('Steps must be a positive integer')
        self.steps += steps
        self.level_steps += steps

    def observe(self, episodes, *, source='development'):
        if source != 'development':
            raise ValueError('Only development evaluations may drive stage decisions')
        if self.last_decision == self.steps:
            return 'retain'
        if len(episodes)<self.min_episodes or any(e.get('status')!='completed' for e in episodes):
            return 'insufficient'
        expected={s.scenario_id for s in stages()[self.level]}
        if not expected.issubset({e.get('scenario_id') for e in episodes}):
            raise ValueError('Development evaluation omitted a current-stage scenario')
        self.last_decision=self.steps
        success=sum(bool(e.get('success')) and not e.get('collision') for e in episodes)/len(episodes)
        collision=sum(bool(e.get('collision')) for e in episodes)/len(episodes)
        efficiency_pass=True
        efficiency_rows=[]
        mean_efficiency=None
        if self.efficiency_enabled:
            efficiency_pass, efficiency_rows, mean_efficiency = _efficiency_report(
                episodes, self.efficiency_threshold)
        rates_pass=success>=.8 and collision<=.05
        passed=rates_pass and efficiency_pass
        eligible=rates_pass and self.level_steps>=self.min_steps
        self.passes=self.passes+1 if passed and self.level_steps>=self.min_steps else 0
        self.failures=self.failures+1 if success<.4 or collision>.2 else 0
        previous=self.level
        decision='retain'
        if self.passes>=self.required_passes and self.level<3:
            self.level+=1; decision='promote'
        elif self.failures>=2 and self.level>0 and self.level_steps>=self.min_steps:
            self.level-=1; decision='fallback'
        if decision=='promote':
            promotion='accepted'
        elif self.efficiency_enabled and eligible and not efficiency_pass:
            promotion='rejected'
        else:
            promotion='not_applied'
        if self.level!=previous:
            self.level_steps=0; self.passes=0; self.failures=0
        self.history.append(dict(
            steps=self.steps, previous=previous, level=self.level, decision=decision,
            success_rate=success, collision_rate=collision, episodes=len(episodes),
            efficiency_gate=self.efficiency_enabled,
            efficiency_pass=efficiency_pass if self.efficiency_enabled else None,
            efficiency_threshold=self.efficiency_threshold if self.efficiency_enabled else None,
            mean_path_efficiency=mean_efficiency, promotion=promotion,
            efficiency_episodes=efficiency_rows,
        ))
        _LOG.info(
            "curriculum decision=%s promotion=%s efficiency_gate=%s efficiency_pass=%s threshold=%s mean_path_efficiency=%s",
            decision, promotion, self.efficiency_enabled,
            efficiency_pass if self.efficiency_enabled else None,
            self.efficiency_threshold if self.efficiency_enabled else None,
            mean_efficiency,
        )
        return decision

    def sample(self, rng):
        level=self.level
        if level and rng.random()<.2:
            level=int(rng.integers(level))
        pool=stages()[level]
        return pool[int(rng.integers(len(pool)))]

    def state_dict(self):
        return asdict(self)
