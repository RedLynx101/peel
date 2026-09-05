"""Bounded mastery curriculum. Test rooms are never used for advancement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import evaluate, load_policy
from .maps import STAGES
from .train import load_config, resolve_device, train


def mastered(successes: int, episodes: int, threshold: float) -> bool:
    if episodes < 1 or not 0 <= threshold <= 1 or not 0 <= successes <= episodes:
        raise ValueError("invalid mastery assessment")
    return successes / episodes >= threshold


def run_curriculum(
    config: dict,
    directory: Path,
    stages: list[str],
    blocks: int = 2,
    threshold: float = 0.8,
    episodes: int = 32,
) -> dict:
    if (
        blocks < 1
        or episodes < 1
        or not 0 <= threshold <= 1
        or any(s not in STAGES for s in stages)
    ):
        raise ValueError("invalid curriculum limits")
    if directory.exists():
        raise ValueError(
            "choose a fresh curriculum directory; existing evidence is preserved"
        )
    directory.mkdir(parents=True)
    history = []
    warm_start = None
    outcome = "complete"
    for stage_index, stage in enumerate(stages):
        for block in range(blocks):
            current = {
                **config,
                "stage": stage,
                "seed": int(config["seed"]) + stage_index * 10000 + block * 1000,
            }
            run_dir = directory / f"{stage}-{block + 1}"
            final = train(current, run_dir, warm_start=warm_start)
            warm_start = run_dir / "champion.pt"
            if not warm_start.exists():
                warm_start = final
            device = resolve_device(current["device"])
            model, saved_config = load_policy(warm_start, device)
            # Fresh training-domain probes; neither validation nor final test namespace.
            base = 800_000 + stage_index * 1000 + block * 100
            summary, _ = evaluate(
                model, saved_config, stage, list(range(base, base + episodes)), device
            )
            passed = mastered(summary["successes"], episodes, threshold)
            history.append(
                {
                    "stage": stage,
                    "block": block + 1,
                    "probe_seed": base,
                    "split": "training-mastery-probe",
                    "summary": summary,
                    "passed": passed,
                    "checkpoint": str(warm_start.relative_to(directory)),
                }
            )
            (directory / "curriculum.json").write_text(
                json.dumps({"threshold": threshold, "history": history}, indent=2)
            )
            if passed:
                break
        if not passed:
            outcome = "budget_exhausted_before_mastery"
            break
    report = {"outcome": outcome, "threshold": threshold, "history": history}
    (directory / "curriculum.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tiny.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/curriculum"))
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--episodes", type=int, default=32)
    args = parser.parse_args()
    print(
        json.dumps(
            run_curriculum(
                load_config(args.config),
                args.run_dir,
                args.stages,
                args.blocks,
                args.threshold,
                args.episodes,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
