from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

ALLOWED = frozenset("#.SEKDBC")
STAGES = ("exit", "banana", "door", "camera")


@dataclass(frozen=True)
class MapValidation:
    valid: bool
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "validation_scope": "geometry_only",
        }


def _base(stage: str) -> list[str]:
    if stage == "exit":
        return [
            "#########",
            "#.......#",
            "#.......#",
            "#..S....#",
            "#.......#",
            "#....E..#",
            "#.......#",
            "#.......#",
            "#########",
        ]
    if stage == "banana":
        return [
            "#########",
            "#.......#",
            "#.......#",
            "#..S....#",
            "#.......#",
            "#..B.E..#",
            "#.......#",
            "#.......#",
            "#########",
        ]
    camera = "#...#.C.#" if stage == "camera" else "#...#...#"
    return [
        "#########",
        "#...#...#",
        camera,
        "#S..D.BE#",
        "#...#...#",
        "#K..#...#",
        "#...#...#",
        "#...#...#",
        "#########",
    ]


def generate_map(stage: str, seed: int) -> list[str]:
    """Generate deterministic layouts with disjoint train/validation/test rows.

    Seed namespaces [0,1m), [1m,2m), and [2m,+) reserve disjoint start
    and door rows. Coordinates and reflection vary inside each namespace.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    domain = min(max(int(seed) // 1_000_000, 0), 2)
    rng = random.Random(int(seed) % 1_000_000)
    row_sets = ((2, 4, 6), (1, 3, 5), (7,))
    start_y = rng.choice(row_sets[domain])
    cells = [
        ["#" if x in (0, 8) or y in (0, 8) else "." for x in range(9)] for y in range(9)
    ]
    if stage in {"exit", "banana"}:
        sx = rng.randint(1, 3)
        cells[start_y][sx] = "S"
        available = [
            (x, y) for y in range(1, 8) for x in range(1, 8) if cells[y][x] == "."
        ]
        if stage == "banana":
            bx, by = rng.choice(available)
            cells[by][bx] = "B"
            available.remove((bx, by))
        ex, ey = rng.choice(available)
        cells[ey][ex] = "E"
    else:
        for y in range(1, 8):
            cells[y][4] = "#"
        door_y = rng.choice(row_sets[domain])
        cells[door_y][4] = "D"
        sx = rng.randint(1, 3)
        cells[start_y][sx] = "S"
        left = [(x, y) for y in range(1, 8) for x in range(1, 4) if cells[y][x] == "."]
        kx, ky = rng.choice(left)
        cells[ky][kx] = "K"
        right = [(x, y) for y in range(1, 8) for x in range(5, 8)]
        bx, by = rng.choice(right)
        cells[by][bx] = "B"
        right.remove((bx, by))
        ex, ey = rng.choice(right)
        cells[ey][ex] = "E"
        right.remove((ex, ey))
        if stage == "camera":
            safe_camera_cells = [(x, y) for x, y in right if x != sx and y != start_y]
            cx, cy = rng.choice(safe_camera_cells or right)
            cells[cy][cx] = "C"
    grid = ["".join(row) for row in cells]
    if rng.randrange(2):
        grid = [row[::-1] for row in grid]
    if stage == "camera" and not camera_timed_route_exists(grid, int(seed) % 4):
        mutable = [list(row) for row in grid]
        old_camera = locate(grid, "C")[0]
        mutable[old_camera[1]][old_camera[0]] = "."
        candidates = [
            (x, y) for y in range(1, 8) for x in range(1, 8) if mutable[y][x] == "."
        ]
        rng.shuffle(candidates)
        for cx, cy in candidates:
            mutable[cy][cx] = "C"
            candidate = ["".join(row) for row in mutable]
            if camera_timed_route_exists(candidate, int(seed) % 4):
                return candidate
            mutable[cy][cx] = "."
        raise RuntimeError(
            f"camera generator failed to find a timed-solvable placement for seed {seed}"
        )
    return grid


def locate(grid: list[str], cell: str) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y, row in enumerate(grid)
        for x, value in enumerate(row)
        if value == cell
    ]


def camera_timed_route_exists(
    grid: list[str], initial_camera_dir: int, max_steps: int = 96
) -> bool:
    """Privileged QA search used to reject impossible generated camera maps."""
    start = locate(grid, "S")[0]
    camera = locate(grid, "C")[0]
    initial_direction = 0 if start[0] < 4 else 2
    # x,y,dir,key,key_collected,banana,opened,camera_dir,step
    initial = (
        *start,
        initial_direction,
        False,
        False,
        False,
        False,
        initial_camera_dir % 4,
        0,
    )
    queue = deque([initial])
    seen = {initial[:-1]}
    directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
    while queue:
        x, y, direction, key, key_collected, banana, opened, camera_dir, step = (
            queue.popleft()
        )
        for action in range(5):
            nx, ny, ndir = x, y, direction
            nkey, nkey_collected, nbanana, nopened = key, key_collected, banana, opened
            if action == 0:
                ndir = (direction - 1) % 4
            elif action == 1:
                ndir = (direction + 1) % 4
            elif action == 2:
                dx, dy = directions[direction]
                target = grid[y + dy][x + dx]
                blocked = (
                    target in "#C"
                    or (target == "K" and not key_collected)
                    or (target == "B" and not banana)
                    or (target == "D" and not opened)
                )
                if not blocked:
                    nx, ny = x + dx, y + dy
            elif action == 3:
                dx, dy = directions[direction]
                target = grid[y + dy][x + dx]
                if target == "K" and not key_collected:
                    nkey = nkey_collected = True
                elif target == "D" and key and not opened:
                    nkey, nopened = False, True
                elif target == "B" and not banana:
                    nbanana = True
            # Capture uses the current camera ray, then the camera rotates.
            cdx, cdy = directions[camera_dir]
            cx, cy = camera[0] + cdx, camera[1] + cdy
            caught = False
            while grid[cy][cx] != "#" and not (grid[cy][cx] == "D" and not nopened):
                if (cx, cy) == (nx, ny):
                    caught = True
                    break
                cx, cy = cx + cdx, cy + cdy
            if caught:
                continue
            if grid[ny][nx] == "E" and nbanana:
                return True
            if step + 1 >= max_steps:
                continue
            state = (
                nx,
                ny,
                ndir,
                nkey,
                nkey_collected,
                nbanana,
                nopened,
                (camera_dir + 1) % 4,
            )
            if state not in seen:
                seen.add(state)
                queue.append((*state, step + 1))
    return False


def validate_map(grid: object) -> MapValidation:
    errors: list[str] = []
    if (
        not isinstance(grid, list)
        or len(grid) != 9
        or any(not isinstance(r, str) or len(r) != 9 for r in grid)
    ):
        return MapValidation(
            False, ("grid must be an array of exactly nine 9-character strings",)
        )
    bad = sorted({cell for row in grid for cell in row if cell not in ALLOWED})
    if bad:
        errors.append(f"unsupported cells: {''.join(bad)}")
    for cell, label, allowed_counts in (
        ("S", "start", {1}),
        ("E", "exit", {1}),
        ("K", "key", {0, 1}),
        ("D", "door", {0, 1}),
        ("B", "banana", {0, 1}),
        ("C", "camera", {0, 1}),
    ):
        count = sum(row.count(cell) for row in grid)
        if count not in allowed_counts:
            errors.append(
                f"expected {label} count in {sorted(allowed_counts)}, found {count}"
            )
    if any(grid[0][x] != "#" or grid[8][x] != "#" for x in range(9)) or any(
        grid[y][0] != "#" or grid[y][8] != "#" for y in range(9)
    ):
        errors.append("outer boundary must be walls")
    if "D" in "".join(grid) and "K" not in "".join(grid):
        errors.append("a locked door requires a key")
    if errors:
        return MapValidation(False, tuple(errors))

    start = locate(grid, "S")[0]
    goals = {"E"}
    if locate(grid, "B"):
        goals.add("B")
    initial = (start[0], start[1], False, False, False)
    seen = {initial}
    queue = deque([initial])
    solved = False
    while queue:
        x, y, key, banana, opened = queue.popleft()
        cell = grid[y][x]
        key = key or cell == "K"
        banana = banana or cell == "B"
        if cell == "E" and (banana or "B" not in goals):
            solved = True
            break
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            target = grid[ny][nx]
            next_opened = opened
            next_key = key
            if target in "#C":
                continue
            if target == "D":
                if opened:
                    pass
                elif not key:
                    continue
                else:
                    next_opened, next_key = True, False
            state = (nx, ny, next_key, banana, next_opened)
            if state not in seen:
                seen.add(state)
                queue.append(state)
    if not solved:
        errors.append("no valid key/door/banana/exit route exists")
    return MapValidation(not errors, tuple(errors))
