from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .history import FEATURE_SIZE
from .model import TemporalActorCritic


def make_dataset(
    samples: int, context: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cue is visible only at t=0; the final observation is identical."""
    rng = np.random.default_rng(seed)
    histories = np.zeros((samples, context, FEATURE_SIZE), np.float32)
    masks = np.ones((samples, context), np.bool_)
    labels = rng.integers(0, 2, samples, dtype=np.int64)
    histories[:, :, 54] = 4  # camera direction hidden
    histories[:, :, 55] = 0.5  # constant time prevents deadline leakage
    histories[:, :, 53] = 5  # constant previous action
    histories[:, 0, 12] = labels + 3  # two tile-category cues encode left/right
    histories[:, 0, 25 + 12] = 1
    return histories, masks, labels


def train_variant(
    contexts: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    context: int,
    width: int,
    epochs: int,
    memoryless: bool,
    seed: int,
    device: str,
) -> dict[str, float]:
    torch.manual_seed(seed)
    model = TemporalActorCritic(context=context, width=width, layers=1, heads=4).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    split = int(len(labels) * 0.8)
    train_idx, test_idx = np.arange(split), np.arange(split, len(labels))
    for _ in range(epochs):
        np.random.default_rng(seed + _).shuffle(train_idx)
        model.train()
        for start in range(0, len(train_idx), 64):
            idx = train_idx[start : start + 64]
            h = torch.from_numpy(contexts[idx]).to(device)
            m = torch.from_numpy(masks[idx]).to(device)
            if memoryless:
                h = h.clone()
                h[:, 0] = h[:, -1]
                h[:, 1:] = 0
                m = torch.zeros_like(m)
                m[:, 0] = True
            logits, _ = model(h, m)
            loss = nn.functional.cross_entropy(
                logits[:, :2], torch.from_numpy(labels[idx]).to(device)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        h = torch.from_numpy(contexts[test_idx]).to(device)
        m = torch.from_numpy(masks[test_idx]).to(device)
        if memoryless:
            h = h.clone()
            h[:, 0] = h[:, -1]
            h[:, 1:] = 0
            m = torch.zeros_like(m)
            m[:, 0] = True
        logits, _ = model(h, m)
        accuracy = float(
            (logits[:, :2].argmax(-1).cpu().numpy() == labels[test_idx]).mean()
        )
        # Same trained model, remove all past observations at evaluation time.
        h = h.clone()
        h[:, 0] = h[:, -1] if not memoryless else h[:, 0]
        h[:, 1:] = 0
        m = torch.zeros_like(m)
        m[:, 0] = True
        ablated, _ = model(h, m)
        removed = float(
            (ablated[:, :2].argmax(-1).cpu().numpy() == labels[test_idx]).mean()
        )
        return {"accuracy": accuracy, "history_removed_accuracy": removed}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled cue benchmark; supervised and separate from heist PPO"
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_num_threads)
    dataset = make_dataset(args.samples, args.context, args.seed)
    history = train_variant(
        *dataset, args.context, args.width, args.epochs, False, args.seed, args.device
    )
    memoryless = train_variant(
        *dataset, args.context, args.width, args.epochs, True, args.seed, args.device
    )
    result = {
        "kind": "controlled-cue-supervised",
        "samples": args.samples,
        "context": args.context,
        "seed": args.seed,
        "epochs": args.epochs,
        "test_samples": args.samples - int(args.samples * 0.8),
        "transformer_history_accuracy": history["accuracy"],
        "history_removed_accuracy": history["history_removed_accuracy"],
        "memoryless_last_observation_accuracy": memoryless["accuracy"],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
