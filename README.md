# RL Robot Navigation — Phase 1 GridWorld

Python 3.10 / Gymnasium custom navigation environment with solvable random maps, deterministic seeding, collision handling, reward shaping, rendering and tests.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.manual_navigation --seed 42
```

Run commands from the repository root. Arrow keys or WASD move; R resets; Esc quits.

Observations: `{'agent': int32[2], 'goal': int32[2], 'obstacles': int8[H,W]}`.
Actions: 0 up, 1 down, 2 left, 3 right. Collisions are penalized but nonterminal.
Maps are sampled until BFS confirms reachability, or an explicit error is raised.

For ML experiments, split train and test by disjoint seed lists; do not reuse held-out evaluation episodes during training.
