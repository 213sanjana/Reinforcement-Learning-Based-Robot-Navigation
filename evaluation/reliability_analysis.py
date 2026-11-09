"""Observation-only scenario descriptors and explicit reliability strata."""
from collections import deque
import numpy as np


def scenario_features(s):
    grid=np.asarray(s['obstacles']);h,w=grid.shape
    start=tuple(s['start']);goal=tuple(s['goal'])
    distance=abs(start[0]-goal[0])+abs(start[1]-goal[1])
    seen={start};queue=deque([start]);dead=0
    while queue:
        r,c=queue.popleft();neighbors=[]
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            p=(r+dr,c+dc)
            if 0<=p[0]<h and 0<=p[1]<w and not grid[p]:neighbors.append(p)
        dead+=int(len(neighbors)==1)
        for p in neighbors:
            if p not in seen:seen.add(p);queue.append(p)
    fraction=dead/len(seen)
    return dict(scenario_id=s['scenario_id'],density=s['density'],manhattan_distance=distance,
        optimal_steps=s['optimal_steps'],detour_ratio=s['optimal_steps']/distance,
        realized_obstacle_fraction=float(grid.mean()),reachable_free_cells=len(seen),
        reachable_fraction_of_free_cells=len(seen)/int((grid==0).sum()),
        reachable_dead_end_fraction=fraction,
        distance_bin='0-5' if distance<6 else ('6-9' if distance<10 else ('10-14' if distance<15 else '15+')),
        optimal_length_bin='1-9' if s['optimal_steps']<10 else ('10-14' if s['optimal_steps']<15 else '15+'),
        complexity_bin='dead_ends_lt_0.10' if fraction<.1 else ('dead_ends_0.10_to_0.25' if fraction<.25 else 'dead_ends_ge_0.25'))


def strata(scenarios):
    features=[scenario_features(s) for s in scenarios]
    result=[]
    # Never pool densities. Empty cells are retained in the reporting grid.
    axes=dict(distance_bin=['0-5','6-9','10-14','15+'],optimal_length_bin=['1-9','10-14','15+'],
              complexity_bin=['dead_ends_lt_0.10','dead_ends_0.10_to_0.25','dead_ends_ge_0.25'])
    for density in (.1,.3,.5):
        result.append(dict(density=density,axis='density',bin='all',scenario_ids=[f['scenario_id'] for f in features if f['density']==density]))
        for axis,bins in axes.items():
            for label in bins:
                result.append(dict(density=density,axis=axis,bin=label,
                    scenario_ids=[f['scenario_id'] for f in features if f['density']==density and f[axis]==label]))
    return features,result
