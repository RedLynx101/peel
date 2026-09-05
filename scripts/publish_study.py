"""Freeze validation-selected models, evaluate test rooms once, export demo evidence.

Run only after training and recipe selection end. Existing test results are reused
only when the exact checkpoint SHA256 matches. Never select weights by test score.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch

from peel.evaluate import evaluate, load_policy, policy_episode

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "artifacts" / "data"
RUN_IDS = [
    "banana-scratch-seed11",
    "banana-study-seed11",
    "banana-refined-seed11",
    "camera-study-seed23",
]
LABELS = [
    "PPO from scratch",
    "Demonstrations + PPO",
    "Longer warm-up · gentle PPO",
    "Camera-stage exploration",
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    experiments, test_results = [], []
    for run_id, label in zip(RUN_IDS, LABELS):
        directory = ROOT / "runs" / run_id
        if not (directory / "final.pt").exists():
            print(f"Pending bounded run: {run_id}", flush=True)
            continue
        config = read(directory / "config.json")
        records = [
            json.loads(line)
            for line in (directory / "metrics.jsonl").read_text().splitlines()
        ]
        raw_dir = DATA / "runs" / run_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name in ["config.json", "metrics.jsonl", "champion.json", "runtime.json"]:
            if (directory / name).exists():
                shutil.copy2(directory / name, raw_dir / name)
        curve = [r for r in records if r["phase"] == "eval"]
        selected = (
            read(directory / "champion.json")
            if (directory / "champion.json").exists()
            else None
        )
        for phase, filename in [
            ("random", "initial-random.pt"),
            ("bc", "post-bc.pt"),
            ("champion", "champion.pt"),
            ("latest", "final.pt"),
        ]:
            checkpoint = directory / filename
            if not checkpoint.exists():
                continue
            checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            cache = raw_dir / f"test-{phase}.json"
            if cache.exists() and read(cache)["checkpoint_sha256"] == checksum:
                result = read(cache)
            else:
                model, saved = load_policy(checkpoint)
                summary, replays = evaluate(
                    model, saved, config["stage"], list(range(2_000_000, 2_000_128))
                )
                result = {
                    "run": run_id,
                    "phase": phase,
                    "split": "test",
                    "seed_base": 2_000_000,
                    "checkpoint_sha256": checksum,
                    "summary": summary,
                    "episodes": [
                        {
                            "seed": r["seed"],
                            "success": r["success"],
                            "steps": r["steps"],
                            "return": r["metrics"]["return"],
                            "status": r["metrics"]["status"],
                        }
                        for r in replays
                    ],
                }
                write(cache, result)
            test_results.append(result)
            print(
                json.dumps({"run": run_id, "phase": phase, **result["summary"]}),
                flush=True,
            )
        champion_test = next(
            (
                r["summary"]
                for r in test_results
                if r["run"] == run_id and r["phase"] == "champion"
            ),
            None,
        )
        experiments.append(
            {
                "id": run_id,
                "label": label,
                "stage": config["stage"],
                "seed": config["seed"],
                "device": config["device"],
                "parameters": 77606,
                "total_steps": config["total_steps"],
                "config": config,
                "curve": curve,
                "test": champion_test,
                "champion": selected,
                "notes": "Validation curve: 32 fixed rooms. Headline: validation-selected champion on 128 untouched test rooms. One training seed per recipe; demonstrations add extra compute. Test rooms have a held-out start row.",
            }
        )
    write(DATA / "experiments.json", experiments)
    write(DATA / "test-results.json", test_results)
    # Showcase selection is based only on a successful validation replay.
    directory = ROOT / "runs" / "banana-refined-seed11"
    chosen = read(DATA / "replays" / "champion-banana-refined-seed11.json")
    seed = chosen["seed"]
    entries = []
    for phase, filename, label in [
        ("random", "initial-random.pt", "01 · Before practice"),
        ("bc", "post-bc.pt", "02 · After watching the guide"),
        ("ppo", "champion.pt", "03 · The banana champion"),
    ]:
        model, config = load_policy(directory / filename)
        replay = policy_episode(model, config, "banana", seed)
        replay.update(id=f"showcase-{phase}", label=label, kind=phase)
        write(DATA / "replays" / f"showcase-{phase}.json", replay)
        entries.append(
            {
                k: replay[k]
                for k in ("id", "label", "kind", "seed", "success", "steps", "stage")
            }
        )
    for filename in ["champion-camera-study-seed23.json", "scripted-door-0.json"]:
        path = DATA / "replays" / filename
        if path.exists():
            replay = read(path)
            replay["label"] = (
                "04 · Camera-stage champion"
                if "camera" in filename
                else "05 · Scripted guide (not learned)"
            )
            write(path, replay)
            entries.append(
                {
                    k: replay[k]
                    for k in (
                        "id",
                        "label",
                        "kind",
                        "seed",
                        "success",
                        "steps",
                        "stage",
                    )
                }
            )
    write(DATA / "replays" / "index.json", entries)
    # Portable inference-only weights: no optimizer or RNG pickle objects.
    checkpoint = torch.load(
        directory / "champion.pt", map_location="cpu", weights_only=False
    )
    compact = {
        k: checkpoint[k]
        for k in (
            "format_version",
            "observation_version",
            "config",
            "model",
            "global_step",
            "phase",
        )
    }
    destination = ROOT / "artifacts" / "models" / "joyce-banana.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, destination)
    write(
        DATA / "model-manifest.json",
        {
            "storage": "bundled",
            "champion": destination.name,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "run": directory.name,
            "validation": read(directory / "champion.json"),
        },
    )


if __name__ == "__main__":
    main()
