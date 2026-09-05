from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .env import ACTION_NAMES, PeelEnv
from .history import HistoryBuffer
from .model import TemporalActorCritic
from .scripted import scripted_episode


def load_policy(path: Path, device: str = "cpu") -> tuple[TemporalActorCritic, dict]:
    raw = torch.load(path, map_location=device, weights_only=False)
    config = raw["config"]
    torch.set_num_threads(int(config.get("torch_num_threads", 2)))
    model = TemporalActorCritic(
        config["context"], config["width"], config["layers"], config["heads"]
    ).to(device)
    load_checkpoint(path, model, device=device)
    model.eval()
    return model, config


@torch.inference_mode()
def policy_episode(
    model: TemporalActorCritic,
    config: dict,
    stage: str,
    seed: int,
    grid: list[str] | None = None,
    deterministic: bool = True,
    device: str = "cpu",
) -> dict:
    env = PeelEnv(stage=stage, max_steps=config.get("max_steps", 96), grid=grid)
    obs, info = env.reset(seed=seed)
    history = HistoryBuffer(model.context)
    frames, actions, probabilities = [info["state"]], [], []
    total = 0.0
    while env.status == "running":
        history.append(obs)
        values, valid = history.snapshot()
        logits, _ = model(
            torch.from_numpy(values).unsqueeze(0).to(device),
            torch.from_numpy(valid).unsqueeze(0).to(device),
        )
        probs = torch.softmax(logits, -1)[0]
        action = int(
            torch.argmax(probs) if deterministic else torch.multinomial(probs, 1)[0]
        )
        obs, reward, terminated, truncated, step_info = env.step(action)
        total += reward
        actions.append(action)
        probabilities.append([float(value) for value in probs.cpu()])
        frames.append(step_info["state"])
        if terminated or truncated:
            break
    return {
        "id": f"policy-{stage}-{seed}",
        "label": f"Learned policy · {stage} · seed {seed}",
        "kind": "policy",
        "seed": seed,
        "success": env.status == "escaped",
        "steps": len(actions),
        "stage": stage,
        "frames": frames,
        "actions": actions,
        "metrics": {
            "return": total,
            "status": env.status,
            "action_names": list(ACTION_NAMES),
            "action_probabilities": probabilities,
        },
    }


def evaluate(
    model: TemporalActorCritic,
    config: dict,
    stage: str,
    seeds: list[int],
    device: str = "cpu",
) -> tuple[dict, list[dict]]:
    replays = [
        policy_episode(model, config, stage, seed, device=device) for seed in seeds
    ]
    successes = sum(replay["success"] for replay in replays)
    returns = [replay["metrics"]["return"] for replay in replays]
    summary = {
        "stage": stage,
        "episodes": len(seeds),
        "successes": successes,
        "success_rate": successes / len(seeds),
        "mean_return": float(np.mean(returns)),
        "mean_steps": float(np.mean([r["steps"] for r in replays])),
    }
    return summary, replays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scripted", action="store_true")
    parser.add_argument(
        "--stage", choices=("exit", "banana", "door", "camera"), default="door"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--seed", type=int, default=2_000_000, help="untouched test-domain seed base"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.scripted and args.checkpoint is None:
        parser.error("provide --checkpoint or --scripted")
    if args.scripted:
        replays = [
            scripted_episode(PeelEnv(args.stage), args.seed + index)[0]
            for index in range(args.episodes)
        ]
        summary = {
            "stage": args.stage,
            "episodes": args.episodes,
            "kind": "scripted",
            "success_rate": sum(r["success"] for r in replays) / args.episodes,
        }
    else:
        model, config = load_policy(args.checkpoint, args.device)
        summary, replays = evaluate(
            model,
            config,
            args.stage,
            list(range(args.seed, args.seed + args.episodes)),
            args.device,
        )
    payload = {"summary": summary, "replays": replays}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
