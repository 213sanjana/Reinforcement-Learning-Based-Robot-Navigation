"""Serializable curriculum decisions; no access to training-success statistics."""
from dataclasses import dataclass, asdict
import math
import numpy as np


def wilson_lower(successes, episodes, z=1.96):
    if episodes < 1 or not 0 <= successes <= episodes:
        raise ValueError('Invalid binomial counts')
    p=successes/episodes
    value=(p+z*z/(2*episodes)-z*math.sqrt(p*(1-p)/episodes+z*z/(4*episodes*episodes)))/(1+z*z/episodes)
    return max(0.,min(1.,value))


@dataclass
class CurriculumManager:
    densities: list
    level_budget: int
    threshold: float=.8
    required_passes: int=3
    min_episodes: int=200
    min_level_steps: int=10240
    forgetting_drop: float=.15
    mode: str='curriculum'
    fixed_level: int=4
    level: int=0
    level_steps: int=0
    phase_steps: int=0
    streak: int=0
    last_evaluation: int=-1
    status: str='running'
    previous_reference: dict=None

    def __post_init__(self):
        if (not self.densities or any(not 0 <= d < 1 for d in self.densities)
                or self.densities != sorted(set(self.densities))):
            raise ValueError('Densities must be strictly increasing in [0,1)')
        if self.mode not in ('curriculum','fixed') or not 0 <= self.fixed_level < len(self.densities):
            raise ValueError('Invalid mode/fixed level')
        if (self.level_budget<1 or self.required_passes<2 or self.min_episodes<1 or self.min_level_steps<0
                or not 0 < self.threshold <= 1 or not 0 <= self.forgetting_drop <= 1):
            raise ValueError('Invalid curriculum decision settings')
        if self.previous_reference is None:
            self.previous_reference={}
        if self.mode=='fixed':
            self.level=self.fixed_level

    def charge(self, steps):
        if self.status!='running' or steps<1:
            raise ValueError('Cannot charge stopped curriculum or nonpositive steps')
        self.phase_steps+=int(steps); self.level_steps+=int(steps)

    def observe(self, summaries, *, allow_promotion=True):
        """One decision per distinct phase step. Baseline does not count as a pass."""
        if self.status!='running' or self.phase_steps==self.last_evaluation:
            return 'none'
        self.last_evaluation=self.phase_steps
        if self.mode=='fixed':
            return 'none'
        if len(summaries)!=len(self.densities):
            raise ValueError('Evaluate every density, including previous levels')
        current=summaries[self.level]
        large=current['episodes']>=self.min_episodes
        passed=large and wilson_lower(current['successes'],current['episodes'])>=self.threshold
        retained=all(summaries[int(k)]['episodes']>=self.min_episodes and
                     summaries[int(k)]['success_rate']>=v-self.forgetting_drop
                     for k,v in self.previous_reference.items())
        eligible=allow_promotion and self.level_steps>=self.min_level_steps
        self.streak=self.streak+1 if eligible and passed and retained else 0
        if self.streak>=self.required_passes:
            self.previous_reference[str(self.level)]=current['success_rate']
            self.streak=0
            if self.level==len(self.densities)-1:
                self.status='mastered'; return 'mastered'
            self.level+=1; self.level_steps=0
            return 'promoted'
        return 'none'

    def enforce_budget(self):
        if self.mode=='curriculum' and self.status=='running' and self.level_steps>=self.level_budget:
            self.status='level_budget_exhausted'
        return self.status

    def weights(self, easier=.2, harder=.1):
        if easier<0 or harder<0 or easier+harder>=1:
            raise ValueError('Mixture weights must be nonnegative with sum <1')
        w=np.zeros(len(self.densities),dtype=float)
        if self.mode=='fixed':
            w[self.fixed_level]=1; return w
        w[self.level]=1
        if self.level>0:
            w[:self.level]=easier/self.level; w[self.level]-=easier
        if self.level+1<len(w):
            w[self.level+1]=harder; w[self.level]-=harder
        return w

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state(cls,state):
        result=cls(**state)
        if not 0 <= result.level < len(result.densities) or result.phase_steps<0 or result.level_steps<0:
            raise ValueError('Invalid saved curriculum state')
        return result
