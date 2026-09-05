from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import PeelEnv
from .scripted import scripted_episode


def save_replay(replay: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(replay, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a real Peel scripted replay")
    parser.add_argument(
        "--stage", choices=("exit", "banana", "door", "camera"), default="door"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay, _ = scripted_episode(PeelEnv(stage=args.stage), args.seed)
    save_replay(replay, args.output)
    print(
        json.dumps(
            {k: replay[k] for k in ("id", "success", "steps", "stage")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
