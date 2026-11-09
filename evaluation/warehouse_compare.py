"""Compare recorded warehouse evaluation files.

Missing files stay NOT_RUN. This script does not launch A*, classical
navigation, behavior cloning, or PPO.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHODS = ("ppo", "astar", "classical", "behavior_cloning")


def load_summary(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {
            "status": "NOT_RUN",
            "success_rate": None,
            "collision_rate": None,
            "timeout_rate": None,
            "source": None,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "summary" in payload and isinstance(payload["summary"], dict):
        payload = payload["summary"]
    required = ("success_rate", "collision_rate", "timeout_rate", "episodes")
    if any(key not in payload for key in required):
        return {
            "status": "UNUSABLE",
            "success_rate": None,
            "collision_rate": None,
            "timeout_rate": None,
            "source": str(path),
            "reason": "summary is missing success, collision, timeout, or episode count",
        }
    return {
        "status": "RECORDED",
        "success_rate": payload.get("success_rate"),
        "collision_rate": payload.get("collision_rate"),
        "timeout_rate": payload.get("timeout_rate"),
        "episodes": payload.get("episodes"),
        "source": str(path),
    }


def compare(paths: dict[str, Path | None]) -> dict:
    report = {method: load_summary(paths.get(method)) for method in METHODS}
    report["ranking"] = None
    report["note"] = (
        "No method is ranked unless every requested file has status RECORDED. "
        "NOT_RUN is not a failure and is not a success."
    )
    if all(item["status"] == "RECORDED" for item in report.values() if isinstance(item, dict) and "status" in item):
        recorded = {method: report[method] for method in METHODS}
        report["ranking"] = sorted(
            recorded,
            key=lambda method: (
                -(recorded[method]["success_rate"] or 0.0),
                recorded[method]["collision_rate"] or 0.0,
            ),
        )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read warehouse evaluation JSON files and compare them.")
    parser.add_argument("--ppo")
    parser.add_argument("--astar")
    parser.add_argument("--classical")
    parser.add_argument("--behavior-cloning")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    report = compare(
        {
            "ppo": None if args.ppo is None else Path(args.ppo),
            "astar": None if args.astar is None else Path(args.astar),
            "classical": None if args.classical is None else Path(args.classical),
            "behavior_cloning": None if args.behavior_cloning is None else Path(args.behavior_cloning),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
