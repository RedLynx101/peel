from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .checkpoint import load_checkpoint, save_checkpoint
from .env import PeelEnv
from .evaluate import evaluate
from .history import HistoryBuffer
from .model import TemporalActorCritic, parameter_count
from .scripted import scripted_episode

DEFAULTS: dict[str, Any] = {
    "seed": 1,
    "stage": "door",
    "max_steps": 96,
    "context": 16,
    "width": 64,
    "layers": 2,
    "heads": 4,
    "device": "auto",
    "learning_rate": 0.0003,
    "num_envs": 8,
    "rollout_steps": 64,
    "total_steps": 98_304,
    "update_epochs": 4,
    "minibatch_size": 128,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "bc_episodes": 128,
    "bc_epochs": 4,
    "bc_batch_size": 128,
    "checkpoint_interval": 10,
    "eval_interval": 10,
    "eval_episodes": 32,
    "max_milestones": 5,
    "torch_num_threads": 2,
}


def resolve_device(name: str) -> str:
    return (
        "cuda"
        if name == "auto" and torch.cuda.is_available()
        else ("cpu" if name == "auto" else name)
    )


def load_config(path: Path) -> dict[str, Any]:
    config = DEFAULTS.copy()
    config.update(json.loads(path.read_text(encoding="utf-8")))
    return config


class JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def demonstration_data(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    contexts, masks, labels = [], [], []
    attempts = successes = 0
    target = int(config["bc_episodes"])
    while successes < target and attempts < max(target * 4, 1):
        seed = int(config["seed"]) + attempts
        replay, examples = scripted_episode(
            PeelEnv(config["stage"], config["max_steps"]), seed
        )
        attempts += 1
        if not replay["success"]:
            continue
        successes += 1
        history = HistoryBuffer(config["context"])
        for obs, action in examples:
            history.append(obs)
            values, valid = history.snapshot()
            contexts.append(values)
            masks.append(valid)
            labels.append(action)
    if target and not contexts:
        raise RuntimeError("observable teacher produced no successful demonstrations")
    return (
        np.asarray(contexts),
        np.asarray(masks),
        np.asarray(labels),
        {"attempts": attempts, "successes": successes},
    )


def behavior_clone(
    model: TemporalActorCritic,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    device: str,
    logger: JsonlLogger,
) -> None:
    if int(config["bc_episodes"]) == 0 or int(config["bc_epochs"]) == 0:
        return
    contexts, masks, labels, demo_stats = demonstration_data(config)
    batch_size = int(config["bc_batch_size"])
    rng = np.random.default_rng(int(config["seed"]))
    for epoch in range(int(config["bc_epochs"])):
        indices = rng.permutation(len(labels))
        losses, correct = [], 0
        model.train()
        for start in range(0, len(indices), batch_size):
            chosen = indices[start : start + batch_size]
            histories = torch.from_numpy(contexts[chosen]).to(device)
            valid = torch.from_numpy(masks[chosen]).to(device)
            targets = torch.from_numpy(labels[chosen]).long().to(device)
            logits, _ = model(histories, valid)
            loss = nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.argmax(-1) == targets).sum())
        logger.write(
            {
                "phase": "bc",
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "accuracy": correct / len(labels),
                "examples": len(labels),
                **demo_stats,
            }
        )


def _evaluation(
    model: TemporalActorCritic,
    config: dict[str, Any],
    device: str,
    global_step: int,
    checkpoint_phase: str,
) -> dict[str, Any]:
    base = 1_000_000 + (int(config["seed"]) % 9_000) * 100
    episodes = int(config["eval_episodes"])
    summary, replays = evaluate(
        model, config, config["stage"], list(range(base, base + episodes)), device
    )
    return {
        "phase": "eval",
        "split": "validation",
        "checkpoint_phase": checkpoint_phase,
        "global_step": global_step,
        "eval_seed_base": base,
        "eval_episodes": episodes,
        "eval_successes": summary["successes"],
        "eval_success_rate": summary["success_rate"],
        "eval_mean_return": summary["mean_return"],
        "eval_mean_steps": summary["mean_steps"],
        "representative_replay": next(
            (replay for replay in replays if replay["success"]), replays[0]
        ),
    }


def _export_experiment(
    run_dir: Path, config: dict[str, Any], logger_path: Path
) -> None:
    records = [
        json.loads(line)
        for line in logger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    curve = [
        {
            key: row[key]
            for key in (
                "global_step",
                "checkpoint_phase",
                "eval_episodes",
                "eval_success_rate",
                "eval_mean_return",
                "eval_mean_steps",
            )
        }
        for row in records
        if row.get("phase") == "eval"
    ]
    data_path = (
        Path(__file__).resolve().parents[2] / "artifacts" / "data" / "experiments.json"
    )
    current = (
        json.loads(data_path.read_text(encoding="utf-8")) if data_path.exists() else []
    )
    current = [item for item in current if item.get("id") != run_dir.name]
    current.append(
        {
            "id": run_dir.name,
            "label": f"{config['stage'].title()} {'BC + PPO' if config['bc_episodes'] and config['bc_epochs'] else 'PPO from scratch'}",
            "stage": config["stage"],
            "seed": config["seed"],
            "kind": "measured-validation",
            "device": config["device"],
            "parameters": parameter_count(
                TemporalActorCritic(
                    config["context"],
                    config["width"],
                    config["layers"],
                    config["heads"],
                )
            ),
            "total_steps": config["total_steps"],
            "config": config,
            "curve": curve,
        }
    )
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(current, indent=2), encoding="utf-8")


def _maybe_promote(
    record: dict[str, Any],
    model: TemporalActorCritic,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    run_dir: Path,
    global_step: int,
    update: int,
    best: tuple[int, float],
    logger: JsonlLogger,
) -> tuple[int, float]:
    score = (int(record["eval_successes"]), float(record["eval_mean_return"]))
    replay = record.pop("representative_replay")
    replay_id = f"{run_dir.name}-{record['checkpoint_phase']}-{global_step}"
    replay["id"] = replay_id
    replay["kind"] = record["checkpoint_phase"]
    replay["label"] = (
        f"{run_dir.name} · {record['checkpoint_phase']} · validation step {global_step}"
    )
    eval_replay_path = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "data"
        / "replays"
        / f"{replay_id}.json"
    )
    eval_replay_path.parent.mkdir(parents=True, exist_ok=True)
    eval_replay_path.write_text(json.dumps(replay, indent=2), encoding="utf-8")
    logger.write(record)
    # A zero-success wiring check is useful telemetry but is not a champion.
    if score[0] == 0 or score <= best:
        return best
    save_checkpoint(
        run_dir / "champion.pt",
        model,
        optimizer,
        config,
        global_step,
        update,
        "champion",
    )
    milestone = run_dir / "milestones" / f"step-{global_step:09d}.pt"
    save_checkpoint(
        milestone, model, optimizer, config, global_step, update, "champion"
    )
    milestones = sorted(milestone.parent.glob("step-*.pt"))
    milestone_root = milestone.parent.resolve()
    run_root = run_dir.resolve()
    if milestone_root.parent != run_root:
        raise RuntimeError("refusing milestone retention outside the run directory")
    for old in milestones[: -int(config["max_milestones"])]:
        resolved = old.resolve()
        if resolved.parent != milestone_root:
            raise RuntimeError("refusing to prune an unverified milestone path")
        resolved.unlink()
    champion = {
        "global_step": global_step,
        "update": update,
        "validation": record,
        "score": list(score),
    }
    (run_dir / "champion.json").write_text(
        json.dumps(champion, indent=2), encoding="utf-8"
    )
    replay_path = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "data"
        / "replays"
        / f"champion-{run_dir.name}.json"
    )
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    champion_replay = {
        **replay,
        "id": f"champion-{run_dir.name}",
        "label": f"Champion {run_dir.name} · validation step {global_step}",
    }
    replay_path.write_text(json.dumps(champion_replay, indent=2), encoding="utf-8")
    try:
        relative = (
            (run_dir / "champion.pt")
            .resolve()
            .relative_to((Path(__file__).resolve().parents[2] / "runs").resolve())
        )
    except ValueError:
        return score
    manifest = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "data"
        / "model-manifest.json"
    )
    manifest.write_text(
        json.dumps({"champion": relative.as_posix(), "validation": record}, indent=2),
        encoding="utf-8",
    )
    return score


@dataclass
class Rollout:
    histories: torch.Tensor
    masks: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    episode_stats: list[dict[str, Any]]


@torch.no_grad()
def collect_rollout(
    model: TemporalActorCritic,
    envs: list[PeelEnv],
    histories: list[HistoryBuffer],
    observations: list[dict[str, np.ndarray]],
    config: dict[str, Any],
    device: str,
    episode_returns: list[float],
    episode_lengths: list[int],
    next_seeds: list[int],
) -> Rollout:
    steps, count = int(config["rollout_steps"]), len(envs)
    stored_h, stored_m, actions, log_probs, rewards, dones, values = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    episode_stats: list[dict[str, Any]] = []
    for _ in range(steps):
        snapshots = [history.snapshot() for history in histories]
        h_np, m_np = (
            np.stack([item[0] for item in snapshots]),
            np.stack([item[1] for item in snapshots]),
        )
        h_tensor, m_tensor = (
            torch.from_numpy(h_np).to(device),
            torch.from_numpy(m_np).to(device),
        )
        action, log_prob, _, value = model.act(h_tensor, m_tensor)
        stored_h.append(h_np)
        stored_m.append(m_np)
        action_np = action.cpu().numpy()
        actions.append(action_np)
        log_probs.append(log_prob.cpu().numpy())
        values.append(value.cpu().numpy())
        step_rewards, step_dones = [], []
        for index, env in enumerate(envs):
            obs, reward, terminated, truncated, _ = env.step(int(action_np[index]))
            done = terminated or truncated
            episode_returns[index] += reward
            episode_lengths[index] += 1
            step_rewards.append(reward)
            step_dones.append(done)
            if done:
                episode_stats.append(
                    {
                        "return": episode_returns[index],
                        "length": episode_lengths[index],
                        "success": env.status == "escaped",
                        "status": env.status,
                    }
                )
                episode_returns[index] = 0.0
                episode_lengths[index] = 0
                next_seeds[index] += len(envs)
                obs, _ = env.reset(seed=next_seeds[index])
                histories[index].reset()
            histories[index].append(obs)
            observations[index] = obs
        rewards.append(step_rewards)
        dones.append(step_dones)
    current = [history.snapshot() for history in histories]
    current_h = torch.from_numpy(np.stack([x[0] for x in current])).to(device)
    current_m = torch.from_numpy(np.stack([x[1] for x in current])).to(device)
    _, next_value = model(current_h, current_m)
    reward_np = np.asarray(rewards, np.float32)
    done_np = np.asarray(dones, np.float32)
    value_np = np.asarray(values, np.float32)
    advantages = np.zeros_like(reward_np)
    last_gae = np.zeros(count, np.float32)
    for step in reversed(range(steps)):
        if step == steps - 1:
            following_value = next_value.cpu().numpy()
        else:
            following_value = value_np[step + 1]
        nonterminal = 1.0 - done_np[step]
        delta = (
            reward_np[step]
            + float(config["gamma"]) * following_value * nonterminal
            - value_np[step]
        )
        last_gae = (
            delta
            + float(config["gamma"])
            * float(config["gae_lambda"])
            * nonterminal
            * last_gae
        )
        advantages[step] = last_gae
    returns = advantages + value_np
    flatten = lambda value: torch.from_numpy(
        np.asarray(value).reshape((-1,) + np.asarray(value).shape[2:])
    ).to(device)
    return Rollout(
        flatten(stored_h),
        flatten(stored_m),
        flatten(actions).long(),
        flatten(log_probs),
        flatten(advantages),
        flatten(returns),
        flatten(values),
        episode_stats,
    )


def ppo_update(
    model: TemporalActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    config: dict[str, Any],
) -> dict[str, float]:
    size = rollout.actions.shape[0]
    indices = np.arange(size)
    metrics: dict[str, list[float]] = {
        key: [] for key in ("policy_loss", "value_loss", "entropy", "approx_kl")
    }
    for _ in range(int(config["update_epochs"])):
        np.random.shuffle(indices)
        for start in range(0, size, int(config["minibatch_size"])):
            batch = indices[start : start + int(config["minibatch_size"])]
            _, new_log_prob, entropy, new_value = model.act(
                rollout.histories[batch], rollout.masks[batch], rollout.actions[batch]
            )
            ratio = (new_log_prob - rollout.log_probs[batch]).exp()
            advantage = rollout.advantages[batch]
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            policy_loss = torch.max(
                -advantage * ratio,
                -advantage
                * ratio.clamp(
                    1 - float(config["clip_coef"]), 1 + float(config["clip_coef"])
                ),
            ).mean()
            value_loss = 0.5 * (new_value - rollout.returns[batch]).pow(2).mean()
            loss = (
                policy_loss
                - float(config["entropy_coef"]) * entropy.mean()
                + float(config["value_coef"]) * value_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
            optimizer.step()
            metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
            metrics["value_loss"].append(float(value_loss.detach().cpu()))
            metrics["entropy"].append(float(entropy.mean().detach().cpu()))
            metrics["approx_kl"].append(
                float(((ratio - 1) - ratio.log()).mean().detach().cpu())
            )
    return {key: float(np.mean(value)) for key, value in metrics.items()}


def train(
    config: dict[str, Any],
    run_dir: Path,
    resume: Path | None = None,
    warm_start: Path | None = None,
) -> Path:
    run_started = time.perf_counter()
    for key in (
        "num_envs",
        "rollout_steps",
        "minibatch_size",
        "update_epochs",
        "eval_interval",
        "eval_episodes",
        "checkpoint_interval",
        "max_milestones",
    ):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if int(config["minibatch_size"]) < 2:
        raise ValueError(
            "minibatch_size must be at least 2 for advantage normalization"
        )
    batch_steps = int(config["rollout_steps"]) * int(config["num_envs"])
    if int(config["total_steps"]) < 0 or int(config["total_steps"]) % batch_steps:
        raise ValueError(f"total_steps must be a nonnegative multiple of {batch_steps}")
    if (run_dir / "metrics.jsonl").exists() and not resume:
        raise ValueError(
            "run directory already contains metrics; use --resume or a new run directory"
        )
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(str(config["device"]))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(int(config["torch_num_threads"]))
    model = TemporalActorCritic(
        config["context"], config["width"], config["layers"], config["heads"]
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]), eps=1e-5
    )
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    run_dir.mkdir(parents=True, exist_ok=True)
    start_update = global_step = 0
    best: tuple[int, float] = (-1, float("-inf"))
    champion_file = run_dir / "champion.json"
    if champion_file.exists():
        raw_score = json.loads(champion_file.read_text(encoding="utf-8")).get(
            "score", [-1, float("-inf")]
        )
        best = (int(raw_score[0]), float(raw_score[1]))
    if resume:
        payload = load_checkpoint(resume, model, optimizer, device, restore_rng=True)
        allowed_changes = {"total_steps", "device"}
        incompatible = [
            key
            for key in set(config) | set(payload["config"])
            if key not in allowed_changes
            and config.get(key) != payload["config"].get(key)
        ]
        if incompatible:
            raise ValueError(
                f"resume config differs in immutable fields: {sorted(incompatible)}; use --warm-start for a new stage"
            )
        start_update, global_step = int(payload["update"]), int(payload["global_step"])
        if int(config["total_steps"]) < global_step:
            raise ValueError("resume total_steps cannot precede the saved global_step")
    else:
        if warm_start:
            load_checkpoint(warm_start, model, device=device)
        else:
            save_checkpoint(
                run_dir / "initial-random.pt", model, optimizer, config, 0, 0, "random"
            )
            record = _evaluation(model, config, device, 0, "random")
            best = _maybe_promote(
                record, model, optimizer, config, run_dir, 0, 0, best, logger
            )
        if int(config["bc_episodes"]) and int(config["bc_epochs"]):
            behavior_clone(model, optimizer, config, device, logger)
            save_checkpoint(
                run_dir / "post-bc.pt", model, optimizer, config, 0, 0, "bc"
            )
            record = _evaluation(model, config, device, 0, "bc")
            best = _maybe_promote(
                record, model, optimizer, config, run_dir, 0, 0, best, logger
            )
        elif warm_start:
            record = _evaluation(model, config, device, 0, "warm-start")
            best = _maybe_promote(
                record, model, optimizer, config, run_dir, 0, 0, best, logger
            )
    # A lower RL rate can protect a useful imitation policy from abrupt drift.
    for group in optimizer.param_groups:
        group["lr"] = float(config.get("ppo_learning_rate", config["learning_rate"]))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    envs = [
        PeelEnv(config["stage"], config["max_steps"])
        for _ in range(int(config["num_envs"]))
    ]
    histories = [HistoryBuffer(config["context"]) for _ in envs]
    if resume and payload.get("trainer_state", {}).get("next_seeds"):
        saved_seeds = payload["trainer_state"]["next_seeds"]
        next_seeds = [int(value) + len(envs) for value in saved_seeds]
    else:
        next_seeds = [seed + global_step * len(envs) + i for i in range(len(envs))]
    if max(next_seeds) >= 1_000_000:
        raise ValueError(
            "training seed stream reached the reserved validation namespace; reduce total_steps or revise the split"
        )
    observations = []
    for env, history, env_seed in zip(envs, histories, next_seeds):
        obs, _ = env.reset(seed=env_seed)
        history.append(obs)
        observations.append(obs)
    episode_returns, episode_lengths = [0.0] * len(envs), [0] * len(envs)
    batch_steps = int(config["rollout_steps"]) * len(envs)
    updates = int(config["total_steps"]) // batch_steps
    started = time.perf_counter()
    for update in range(start_update, updates):
        model.eval()
        rollout = collect_rollout(
            model,
            envs,
            histories,
            observations,
            config,
            device,
            episode_returns,
            episode_lengths,
            next_seeds,
        )
        model.train()
        losses = ppo_update(model, optimizer, rollout, config)
        global_step += batch_steps
        record: dict[str, Any] = {
            "phase": "ppo",
            "update": update + 1,
            "global_step": global_step,
            "seconds": time.perf_counter() - started,
            **losses,
        }
        if rollout.episode_stats:
            record.update(
                {
                    "episodes": len(rollout.episode_stats),
                    "mean_episode_return": float(
                        np.mean([x["return"] for x in rollout.episode_stats])
                    ),
                    "success_rate": float(
                        np.mean([x["success"] for x in rollout.episode_stats])
                    ),
                }
            )
        logger.write(record)
        if (update + 1) % int(config["checkpoint_interval"]) == 0:
            save_checkpoint(
                run_dir / "latest.pt",
                model,
                optimizer,
                config,
                global_step,
                update + 1,
                "ppo",
                {"next_seeds": next_seeds},
            )
        if (update + 1) % int(config["eval_interval"]) == 0 or update + 1 == updates:
            model.eval()
            eval_record = _evaluation(model, config, device, global_step, "ppo")
            best = _maybe_promote(
                eval_record,
                model,
                optimizer,
                config,
                run_dir,
                global_step,
                update + 1,
                best,
                logger,
            )
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in eval_record.items()
                        if key != "representative_replay"
                    }
                ),
                flush=True,
            )
        print(json.dumps(record), flush=True)
    final = run_dir / "final.pt"
    save_checkpoint(
        final,
        model,
        optimizer,
        config,
        global_step,
        updates,
        "ppo",
        {"next_seeds": next_seeds},
    )
    save_checkpoint(
        run_dir / "latest.pt",
        model,
        optimizer,
        config,
        global_step,
        updates,
        "ppo",
        {"next_seeds": next_seeds},
    )
    _export_experiment(run_dir, config, logger.path)
    (run_dir / "runtime.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": time.perf_counter() - run_started,
                "device": device,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated()
                if device == "cuda"
                else None,
                "includes": "initialization, demonstrations, PPO, evaluation, checkpoint and artifact writes",
            },
            indent=2,
        )
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="BC warm-up and PPO training for Peel")
    parser.add_argument("--config", type=Path, default=Path("configs/tiny.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/tiny"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--warm-start",
        type=Path,
        help="load weights only for a new stage/run; resets optimizer and counters",
    )
    parser.add_argument("--device")
    parser.add_argument("--total-steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(int(config["torch_num_threads"]))
    if args.device:
        config["device"] = args.device
    if args.total_steps is not None:
        config["total_steps"] = args.total_steps
    print(
        json.dumps(
            {
                "device": resolve_device(config["device"]),
                "parameters": parameter_count(
                    TemporalActorCritic(
                        config["context"],
                        config["width"],
                        config["layers"],
                        config["heads"],
                    )
                ),
                "config": config,
            },
            indent=2,
        )
    )
    if args.resume and args.warm_start:
        parser.error("--resume and --warm-start are mutually exclusive")
    train(config, args.run_dir, args.resume, args.warm_start)


if __name__ == "__main__":
    main()
