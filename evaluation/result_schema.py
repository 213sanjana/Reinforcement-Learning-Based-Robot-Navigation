"""Versioned per-episode metrics; steps include collisions and failed attempts."""
from dataclasses import asdict, dataclass
import math

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EpisodeResult:
    policy: str
    scenario_id: str
    map_id: str
    map_seed: int
    density: float
    horizon: int
    success: bool
    timeout: bool
    episode_length: int
    collisions: int
    reward: float
    optimal_steps: int
    path_efficiency: float | None
    spl: float

    def to_dict(self):
        if not 1 <= self.optimal_steps <= self.episode_length <= self.horizon:
            raise ValueError('Invalid episode/reference length')
        if not 0 <= self.collisions <= self.episode_length or not math.isfinite(self.reward):
            raise ValueError('Invalid collisions/reward')
        if self.success == self.timeout or (self.timeout and self.episode_length != self.horizon):
            raise ValueError('Invalid episode ending')
        efficiency = self.optimal_steps/self.episode_length if self.success else None
        if self.path_efficiency != efficiency or self.spl != (efficiency or 0.):
            raise ValueError('Invalid efficiency/SPL')
        return asdict(self)


def wilson(successes, n):
    z = 1.959963984540054
    p = successes/n; denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0., center-radius), min(1., center+radius)]


def summarize(rows):
    n = len(rows); successes = sum(r['success'] for r in rows)
    return dict(episodes=n, successes=successes, success_rate=successes/n,
                success_rate_wilson95=wilson(successes,n),
                mean_collisions=sum(r['collisions'] for r in rows)/n,
                mean_episode_length=sum(r['episode_length'] for r in rows)/n,
                mean_reward=sum(r['reward'] for r in rows)/n,
                mean_spl=sum(r['spl'] for r in rows)/n,
                mean_successful_path_efficiency=(sum(r['path_efficiency'] or 0. for r in rows)/successes if successes else None))
