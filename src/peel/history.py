from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import numpy as np

FEATURE_SIZE = 56


def encode_observation(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["tiles"], dtype=np.float32).reshape(-1),
            np.asarray(obs["visible"], dtype=np.float32).reshape(-1),
            np.asarray(obs["inventory"], dtype=np.float32).reshape(-1),
            np.asarray(obs["direction"], dtype=np.float32).reshape(1),
            np.asarray(obs["previous_action"], dtype=np.float32).reshape(1),
            np.asarray(obs["camera_direction"], dtype=np.float32).reshape(1),
            np.asarray(obs["time"], dtype=np.float32).reshape(1),
        ]
    )


class HistoryBuffer:
    def __init__(self, context: int):
        self.context = int(context)
        self._items: deque[np.ndarray] = deque(maxlen=self.context)

    def reset(self) -> None:
        self._items.clear()

    def append(self, obs: dict[str, np.ndarray] | np.ndarray) -> None:
        value = (
            encode_observation(obs)
            if isinstance(obs, dict)
            else np.asarray(obs, dtype=np.float32)
        )
        if value.shape != (FEATURE_SIZE,):
            raise ValueError(
                f"expected encoded observation shape {(FEATURE_SIZE,)}, got {value.shape}"
            )
        self._items.append(value.copy())

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        data = np.zeros((self.context, FEATURE_SIZE), dtype=np.float32)
        mask = np.zeros(self.context, dtype=np.bool_)
        if self._items:
            values = np.stack(tuple(self._items))
            data[: len(values)] = values
            mask[: len(values)] = True
        return data, mask

    def extend(self, observations: Iterable[dict[str, np.ndarray]]) -> None:
        for obs in observations:
            self.append(obs)

    def __len__(self) -> int:
        return len(self._items)
