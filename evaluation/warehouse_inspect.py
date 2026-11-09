"""Read warehouse checkpoint metadata without loading historical PPO zips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.warehouse_checkpoint import assert_compatible, assert_load_allowed


def inspect_run(directory: Path) -> dict:
    directory = assert_load_allowed(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert_compatible(metadata)
    return {
        "directory": str(directory),
        "metadata": metadata,
        "model_present": (directory / "ppo_model.zip").is_file(),
        "normalization_present": (directory / "vecnormalize.pkl").is_file(),
        "episodes_present": (directory / "episodes.csv").is_file(),
        "loaded_weights": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect warehouse PPO metadata. Does not load weights.")
    parser.add_argument("run_directory")
    args = parser.parse_args(argv)
    print(json.dumps(inspect_run(Path(args.run_directory)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
