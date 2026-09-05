from pathlib import Path

import numpy as np
import torch

from peel.checkpoint import load_checkpoint, save_checkpoint
from peel.env import PeelEnv
from peel.history import HistoryBuffer, encode_observation
from peel.memory_benchmark import make_dataset
from peel.model import TemporalActorCritic


def test_history_reset_and_independent_buffers():
    obs, _ = PeelEnv("exit").reset(seed=0)
    first, second = HistoryBuffer(4), HistoryBuffer(4)
    first.append(obs)
    assert len(first) == 1 and len(second) == 0
    data, mask = first.snapshot()
    assert mask.tolist() == [True, False, False, False]
    assert np.array_equal(data[0], encode_observation(obs))
    first.reset()
    assert not first.snapshot()[1].any()


def test_transformer_is_causal():
    torch.manual_seed(0)
    model = TemporalActorCritic(context=4, width=32, layers=1, heads=4).eval()
    history = torch.zeros(1, 4, 56)
    history[:, :, 53] = 5
    history[:, :, 54] = 4
    valid = torch.ones(1, 4, dtype=torch.bool)
    with torch.inference_mode():
        before, _ = model.forward_all(history, valid)
        history[:, 3, 0] = 7
        after, _ = model.forward_all(history, valid)
    assert torch.allclose(before[:, :3], after[:, :3], atol=1e-6)
    assert not torch.allclose(before[:, 3], after[:, 3])


def test_controlled_memory_current_observation_has_no_label_leakage():
    histories, masks, labels = make_dataset(100, context=8, seed=3)
    assert masks.all()
    assert np.unique(labels).size == 2
    for index in range(1, 8):
        assert np.array_equal(histories[0, index], histories[-1, index])
    assert not np.array_equal(
        histories[labels == 0][0, 0], histories[labels == 1][0, 0]
    )


def test_checkpoint_roundtrip_reproduces_outputs(tmp_path: Path):
    torch.manual_seed(4)
    config = {"context": 4, "width": 32, "layers": 1, "heads": 4}
    model = TemporalActorCritic(**config)
    optimizer = torch.optim.Adam(model.parameters())
    history = torch.zeros(2, 4, 56)
    history[:, :, 53] = 5
    history[:, :, 54] = 4
    valid = torch.ones(2, 4, dtype=torch.bool)
    expected = model(history, valid)[0].detach()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, config, 20, 2, "ppo")
    restored = TemporalActorCritic(**config)
    payload = load_checkpoint(path, restored, device="cpu")
    assert payload["resume_boundary"] == "fresh_episode"
    assert torch.equal(expected, restored(history, valid)[0].detach())


def test_checkpoint_can_restore_rng_state(tmp_path: Path):
    import random

    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    config = {"context": 4, "width": 32, "layers": 1, "heads": 4}
    model = TemporalActorCritic(**config)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "rng.pt"
    save_checkpoint(path, model, optimizer, config, 0, 0, "test")
    expected = (random.random(), float(np.random.random()), float(torch.rand(())))
    load_checkpoint(path, model, optimizer, restore_rng=True)
    actual = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert actual == expected
