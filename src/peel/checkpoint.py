from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

OBSERVATION_VERSION = "peel-symbolic-5x5-v2-camera-visible-facing"


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    global_step: int,
    update: int,
    phase: str,
    trainer_state: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "observation_version": OBSERVATION_VERSION,
        "resume_boundary": "fresh_episode",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "global_step": global_step,
        "update": update,
        "phase": phase,
        "trainer_state": trainer_state or {},
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("observation_version") != OBSERVATION_VERSION:
        raise ValueError("checkpoint observation version is incompatible")
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        random.setstate(payload["rng"]["python"])
        np.random.set_state(payload["rng"]["numpy"])
        torch.set_rng_state(payload["rng"]["torch"].cpu())
        if torch.cuda.is_available() and payload["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    return payload
