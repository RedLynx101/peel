from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .maps import generate_map, locate, validate_map

ACTION_NAMES = ("left", "right", "forward", "interact", "wait")
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))
TILE_ID = {
    "?": 0,
    ".": 1,
    "S": 1,
    "#": 2,
    "E": 3,
    "K": 4,
    "D": 5,
    "B": 6,
    "C": 7,
    "d": 8,
}


def _line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    """Integer Bresenham cells from origin through target."""
    points: list[tuple[int, int]] = []
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        points.append((x0, y0))
        if (x0, y0) == (x1, y1):
            return points
        twice = 2 * err
        if twice >= dy:
            err += dy
            x0 += sx
        if twice <= dx:
            err += dx
            y0 += sy


class PeelEnv(gym.Env[dict[str, np.ndarray], int]):
    """Nine-by-nine, five-action, partially observable banana heist."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self, stage: str = "door", max_steps: int = 96, grid: list[str] | None = None
    ):
        super().__init__()
        self.stage = stage
        self.max_steps = int(max_steps)
        self.fixed_grid = list(grid) if grid is not None else None
        if self.fixed_grid is not None:
            result = validate_map(self.fixed_grid)
            if not result.valid:
                raise ValueError("invalid map: " + "; ".join(result.errors))
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict(
            {
                "tiles": spaces.Box(0, max(TILE_ID.values()), (5, 5), np.int64),
                "visible": spaces.MultiBinary((5, 5)),
                "inventory": spaces.MultiBinary(2),
                "direction": spaces.Discrete(4),
                "previous_action": spaces.Discrete(6),
                "camera_direction": spaces.Discrete(5),
                "time": spaces.Box(0.0, 1.0, (1,), np.float32),
            }
        )
        self.grid: list[list[str]] = []
        self.agent = (0, 0)
        self.direction = 0
        self.camera: tuple[int, int] | None = None
        self.camera_dir = 0
        self.has_key = False
        self.has_banana = False
        self.step_count = 0
        self.previous_action = 5
        self.status = "running"
        self._stage_rewards: set[str] = set()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        actual_seed = int(
            seed if seed is not None else self.np_random.integers(0, 2**31 - 1)
        )
        rows = list(self.fixed_grid or generate_map(self.stage, actual_seed))
        self.grid = [list(row) for row in rows]
        self.agent = locate(rows, "S")[0]
        self.grid[self.agent[1]][self.agent[0]] = "."
        cameras = locate(rows, "C")
        self.camera = cameras[0] if cameras else None
        if self.camera:
            self.grid[self.camera[1]][self.camera[0]] = "C"
        self.direction = 0 if self.agent[0] < 4 else 2
        self.camera_dir = actual_seed % 4
        self.has_key = self.has_banana = False
        self.step_count = 0
        self.previous_action = 5
        self.status = "running"
        self._stage_rewards.clear()
        obs = self._observation()
        return obs, {
            "seed": actual_seed,
            "stage": self.stage,
            "state": self.public_state(),
        }

    def _opaque(self, x: int, y: int) -> bool:
        return self.grid[y][x] in "#D"

    def _can_see(self, tx: int, ty: int) -> bool:
        ax, ay = self.agent
        for x, y in _line(ax, ay, tx, ty)[1:-1]:
            if self._opaque(x, y):
                return False
        return True

    def _observation(self) -> dict[str, np.ndarray]:
        ax, ay = self.agent
        tiles = np.zeros((5, 5), dtype=np.int64)
        visible = np.zeros((5, 5), dtype=np.int8)
        for oy, dy in enumerate(range(-2, 3)):
            for ox, dx in enumerate(range(-2, 3)):
                x, y = ax + dx, ay + dy
                if 0 <= x < 9 and 0 <= y < 9 and self._can_see(x, y):
                    tiles[oy, ox] = TILE_ID[self.grid[y][x]]
                    visible[oy, ox] = 1
        camera_visible = False
        if self.camera is not None:
            cx, cy = self.camera
            if abs(cx - ax) <= 2 and abs(cy - ay) <= 2 and self._can_see(cx, cy):
                camera_visible = True
        return {
            "tiles": tiles,
            "visible": visible,
            "inventory": np.asarray([self.has_key, self.has_banana], dtype=np.int8),
            "direction": np.asarray(self.direction, dtype=np.int64),
            "previous_action": np.asarray(self.previous_action, dtype=np.int64),
            "camera_direction": np.asarray(
                self.camera_dir if camera_visible else 4, dtype=np.int64
            ),
            "time": np.asarray(
                [max(0.0, (self.max_steps - self.step_count) / self.max_steps)],
                dtype=np.float32,
            ),
        }

    def _camera_sees_agent(self) -> bool:
        if self.camera is None:
            return False
        cx, cy = self.camera
        dx, dy = DIRS[self.camera_dir]
        x, y = cx + dx, cy + dy
        while 0 <= x < 9 and 0 <= y < 9 and self.grid[y][x] not in "#D":
            if (x, y) == self.agent:
                return True
            x, y = x + dx, y + dy
        return False

    def step(self, action: int):
        if self.status != "running":
            raise RuntimeError("step called after episode ended; call reset")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action}")
        reward = -0.01
        if action == 0:
            self.direction = (self.direction - 1) % 4
        elif action == 1:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            dx, dy = DIRS[self.direction]
            nx, ny = self.agent[0] + dx, self.agent[1] + dy
            if self.grid[ny][nx] not in "#DKBC":
                self.agent = (nx, ny)
        elif action == 3:
            dx, dy = DIRS[self.direction]
            x, y = self.agent[0] + dx, self.agent[1] + dy
            cell = self.grid[y][x]
            if cell == "K":
                self.has_key = True
                self.grid[y][x] = "."
                if "key" not in self._stage_rewards:
                    reward += 0.10
                    self._stage_rewards.add("key")
            elif cell == "D" and self.has_key:
                self.has_key = False
                self.grid[y][x] = "d"
                if "door" not in self._stage_rewards:
                    reward += 0.20
                    self._stage_rewards.add("door")
            elif cell == "B":
                self.has_banana = True
                self.grid[y][x] = "."
                if "banana" not in self._stage_rewards:
                    reward += 1.0
                    self._stage_rewards.add("banana")

        self.step_count += 1
        self.previous_action = action
        terminated = truncated = False
        if self._camera_sees_agent():
            reward -= 5.0
            self.status = "caught"
            terminated = True
        elif self.grid[self.agent[1]][self.agent[0]] == "E" and (
            self.has_banana or self.stage == "exit"
        ):
            reward += 10.0
            self.status = "escaped"
            terminated = True
        elif self.step_count >= self.max_steps:
            self.status = "timeout"
            # The deadline is part of the task and observable to the policy, so
            # reaching it is a terminal MDP state rather than an external cutoff.
            terminated = True
        if self.camera is not None:
            self.camera_dir = (self.camera_dir + 1) % 4
        obs = self._observation()
        return (
            obs,
            float(reward),
            terminated,
            truncated,
            {"state": self.public_state(reward), "status": self.status},
        )

    def public_state(self, reward: float | None = None) -> dict[str, Any]:
        rows = [row.copy() for row in self.grid]
        data: dict[str, Any] = {
            "grid": ["".join(row) for row in rows],
            "agent": {"x": self.agent[0], "y": self.agent[1], "dir": self.direction},
            "inventory": {"key": self.has_key, "banana": self.has_banana},
            "visible": [
                [self.agent[0] + ox, self.agent[1] + oy]
                for oy in range(-2, 3)
                for ox in range(-2, 3)
                if 0 <= self.agent[0] + ox < 9
                and 0 <= self.agent[1] + oy < 9
                and self._can_see(self.agent[0] + ox, self.agent[1] + oy)
            ],
            "step": self.step_count,
            "max_steps": self.max_steps,
            "stage": self.stage,
            "status": self.status,
        }
        if self.camera is not None:
            data["camera"] = {
                "x": self.camera[0],
                "y": self.camera[1],
                "dir": self.camera_dir,
            }
        if reward is not None:
            data["reward"] = float(reward)
        return data
