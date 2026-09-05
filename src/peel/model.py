from __future__ import annotations

import torch
from torch import nn

from .history import FEATURE_SIZE


class TemporalActorCritic(nn.Module):
    def __init__(
        self, context: int = 16, width: int = 64, layers: int = 2, heads: int = 4
    ):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.context = int(context)
        self.width = int(width)
        self.tile_embedding = nn.Embedding(9, 4)
        self.direction_embedding = nn.Embedding(4, 4)
        self.action_embedding = nn.Embedding(6, 4)
        self.camera_embedding = nn.Embedding(5, 4)
        # 25*4 tile channels + 25 visibility + 2 inventory + three 4d categories + time.
        self.input_projection = nn.Sequential(nn.Linear(140, width), nn.Tanh())
        self.position = nn.Parameter(torch.zeros(1, context, width))
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            width * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False
        )
        self.actor = nn.Linear(width, 5)
        self.critic = nn.Linear(width, 1)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, 2**0.5)
            nn.init.zeros_(module.bias)

    def _tokens(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != FEATURE_SIZE:
            raise ValueError(f"history must have shape [batch,time,{FEATURE_SIZE}]")
        tiles = self.tile_embedding(history[..., :25].long().clamp(0, 8)).flatten(-2)
        visible = history[..., 25:50]
        inventory = history[..., 50:52]
        direction = self.direction_embedding(history[..., 52].long().clamp(0, 3))
        previous = self.action_embedding(history[..., 53].long().clamp(0, 5))
        camera = self.camera_embedding(history[..., 54].long().clamp(0, 4))
        time = history[..., 55:56]
        return self.input_projection(
            torch.cat(
                (tiles, visible, inventory, direction, previous, camera, time), dim=-1
            )
        )

    def forward_all(
        self, history: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = history.shape[1]
        tokens = self._tokens(history) + self.position[:, :length]
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=history.device),
            diagonal=1,
        )
        encoded = self.transformer(
            tokens, mask=causal, src_key_padding_mask=~valid.bool()
        )
        return self.actor(encoded), self.critic(encoded).squeeze(-1)

    def forward(
        self, history: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = self.forward_all(history, valid)
        positions = torch.arange(history.shape[1], device=history.device).expand_as(
            valid
        )
        last = torch.where(valid.bool(), positions, -1).max(dim=1).values.clamp_min(0)
        rows = torch.arange(history.shape[0], device=history.device)
        return logits[rows, last], values[rows, last]

    def act(
        self,
        history: torch.Tensor,
        valid: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        logits, value = self(history, valid)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample() if action is None else action
        return action, distribution.log_prob(action), distribution.entropy(), value


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
