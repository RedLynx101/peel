from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .env import DIRS


@dataclass
class ObservableTeacher:
    """A map-building teacher that receives policy observations only.

    Coordinates are dead-reckoned relative to reset. It never reads the env,
    seed, public state, hidden camera direction, or generator labels.
    """

    position: tuple[int, int] = (0, 0)
    direction: int = 0
    known: dict[tuple[int, int], int] = field(default_factory=dict)
    last_action: int = 5
    pending_forward: bool = False

    def reset(self) -> None:
        self.position = (0, 0)
        self.direction = 0
        self.known.clear()
        self.last_action = 5
        self.pending_forward = False

    def _integrate_motion(self, obs: dict[str, np.ndarray]) -> None:
        observed_dir = int(obs["direction"])
        if self.pending_forward:
            # Forward was issued only for a remembered passable tile.
            dx, dy = DIRS[self.direction]
            self.position = (self.position[0] + dx, self.position[1] + dy)
        self.direction = observed_dir
        self.pending_forward = False

    def _remember(self, obs: dict[str, np.ndarray]) -> None:
        px, py = self.position
        tiles = obs["tiles"]
        vis = obs["visible"]
        for oy in range(5):
            for ox in range(5):
                if vis[oy, ox]:
                    self.known[(px + ox - 2, py + oy - 2)] = int(tiles[oy, ox])

    def _target_kind(self, obs: dict[str, np.ndarray]) -> int:
        key, banana = map(bool, obs["inventory"])
        if not banana:
            if 6 in self.known.values():
                return 6
            if not key and 5 in self.known.values() and 4 in self.known.values():
                return 4
            if key and 5 in self.known.values():
                return 5
            # Until all reachable frontiers are mapped, an observed exit does
            # not prove this is the exit-only stage; a banana may still be hidden.
            has_frontier = any(
                kind in {1, 3, 8}
                and any(
                    neighbor not in self.known
                    for neighbor in (
                        (p[0] + 1, p[1]),
                        (p[0] - 1, p[1]),
                        (p[0], p[1] + 1),
                        (p[0], p[1] - 1),
                    )
                )
                for p, kind in self.known.items()
            )
            return 6 if has_frontier or 3 not in self.known.values() else 3
        return 3

    def _path_to(
        self, kinds: set[int], adjacent: bool = False
    ) -> list[tuple[int, int]] | None:
        start = self.position
        passable = {1, 3, 8}
        targets = {p for p, kind in self.known.items() if kind in kinds}
        if adjacent:
            targets = {
                n
                for p in targets
                for n in (
                    (p[0] + 1, p[1]),
                    (p[0] - 1, p[1]),
                    (p[0], p[1] + 1),
                    (p[0], p[1] - 1),
                )
                if self.known.get(n) in passable
            }
        queue = deque([start])
        prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while queue:
            pos = queue.popleft()
            if pos in targets:
                path = []
                while pos != start:
                    path.append(pos)
                    pos = prev[pos]  # type: ignore[assignment]
                return list(reversed(path))
            for nxt in (
                (pos[0] + 1, pos[1]),
                (pos[0] - 1, pos[1]),
                (pos[0], pos[1] + 1),
                (pos[0], pos[1] - 1),
            ):
                if nxt not in prev and self.known.get(nxt) in passable:
                    prev[nxt] = pos
                    queue.append(nxt)
        return None

    def _front(self) -> tuple[int, int]:
        dx, dy = DIRS[self.direction]
        return self.position[0] + dx, self.position[1] + dy

    def act(self, obs: dict[str, np.ndarray]) -> int:
        self._integrate_motion(obs)
        self._remember(obs)
        target_kind = self._target_kind(obs)
        # Interactable goals are handled from an adjacent floor.
        if self.known.get(self._front()) == target_kind and target_kind in {4, 5, 6}:
            action = 3
        else:
            adjacent_target = next(
                (
                    p
                    for p, kind in self.known.items()
                    if kind == target_kind
                    and abs(p[0] - self.position[0]) + abs(p[1] - self.position[1]) == 1
                ),
                None,
            )
            if adjacent_target is not None and target_kind in {4, 5, 6}:
                delta = (
                    adjacent_target[0] - self.position[0],
                    adjacent_target[1] - self.position[1],
                )
                desired = DIRS.index(delta)
                diff = (desired - self.direction) % 4
                action = 1 if diff in (1, 2) else 0
                self.last_action = action
                return action
            path = self._path_to({target_kind}, adjacent=target_kind in {4, 5, 6})
            if not path:
                # Explore the closest remembered floor bordering an unseen cell.
                frontier = {
                    p
                    for p, k in self.known.items()
                    if k in {1, 3, 8}
                    and any(
                        n not in self.known
                        for n in (
                            (p[0] + 1, p[1]),
                            (p[0] - 1, p[1]),
                            (p[0], p[1] + 1),
                            (p[0], p[1] - 1),
                        )
                    )
                }
                path = self._path_to_positions(frontier)
            if not path:
                action = 1
            else:
                nx, ny = path[0]
                delta = (nx - self.position[0], ny - self.position[1])
                desired = DIRS.index(delta)
                diff = (desired - self.direction) % 4
                action = 2 if diff == 0 else (1 if diff in (1, 2) else 0)
        self.last_action = action
        self.pending_forward = action == 2
        return action

    def _path_to_positions(
        self, targets: set[tuple[int, int]]
    ) -> list[tuple[int, int]] | None:
        start = self.position
        queue = deque([start])
        prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while queue:
            pos = queue.popleft()
            if pos in targets and pos != start:
                path = []
                while pos != start:
                    path.append(pos)
                    pos = prev[pos]  # type: ignore[assignment]
                return list(reversed(path))
            for nxt in (
                (pos[0] + 1, pos[1]),
                (pos[0] - 1, pos[1]),
                (pos[0], pos[1] + 1),
                (pos[0], pos[1] - 1),
            ):
                if nxt not in prev and self.known.get(nxt) in {1, 3, 8}:
                    prev[nxt] = pos
                    queue.append(nxt)
        return None


def scripted_episode(
    env: Any, seed: int
) -> tuple[dict[str, Any], list[tuple[dict[str, np.ndarray], int]]]:
    obs, info = env.reset(seed=seed)
    teacher = ObservableTeacher(direction=int(obs["direction"]))
    frames = [info["state"]]
    actions: list[int] = []
    examples: list[tuple[dict[str, np.ndarray], int]] = []
    total = 0.0
    while env.status == "running":
        action = teacher.act(obs)
        examples.append(({key: value.copy() for key, value in obs.items()}, action))
        obs, reward, terminated, truncated, step_info = env.step(action)
        total += reward
        actions.append(action)
        frames.append(step_info["state"])
        if terminated or truncated:
            break
    replay = {
        "id": f"scripted-{env.stage}-{seed}",
        "label": f"Scripted observable teacher · {env.stage} · seed {seed}",
        "kind": "scripted",
        "seed": seed,
        "success": env.status == "escaped",
        "steps": len(actions),
        "stage": env.stage,
        "frames": frames,
        "actions": actions,
        "metrics": {
            "return": total,
            "status": env.status,
            "action_names": ["left", "right", "forward", "interact", "wait"],
        },
    }
    return replay, examples
