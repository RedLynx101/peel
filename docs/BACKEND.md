# Peel backend

Peel is a bounded Gymnasium environment and a small causal transformer actor-critic. It has a five-action museum heist, behavioral-cloning warm-up from an observation-only teacher, PPO, checkpoint/resume, evaluation replay export, a controlled cue-memory benchmark, and a FastAPI interface.

## Why a custom Gymnasium environment

The proposal recommended a MiniGrid subclass as the likely fastest route. The implementation instead uses the Gymnasium API directly while retaining the same 9x9 symbolic grid conventions. MiniGrid's carried-object and view encodings work against this project's fixed two-bit inventory and exact 5x5 contract. The custom environment is about 200 lines, makes camera timing and ASCII replay state explicit, and avoids translating between two rule sets. It still uses standard `reset` and `step` signatures, action/observation spaces, deterministic seeding, and terminal semantics.

The procedural distribution is intentionally bounded: seeded start, key, banana, exit, camera, door-row coordinates, and horizontal reflection vary within open-room templates. Seed namespaces below one million, one-to-two million, and two million onward reserve disjoint start and door rows for training, validation, and final test, respectively. This prevents content-identical held-out maps while keeping initial learning tractable. `validate_map` checks dimensions, vocabulary, counts, a wall boundary, and a geometric state-space route that respects key-before-door and banana-before-exit dependencies. Its API response labels this `validation_scope: geometry_only`; a camera map still needs timed-policy evaluation.

## Rules and observations

Actions are `left`, `right`, `forward`, `interact`, and `wait` (IDs 0 through 4). Interact collects the key or banana directly ahead or consumes a held key to open the door. Stage rewards are one-time; repeated interaction cannot farm them. A successful escape dominates shaping. The observable deadline is part of the finite task, so timeout is a terminal state. The camera checks its current cardinal ray after the action and then rotates clockwise.

The policy receives a wall-occluded, north-up 5x5 symbolic window; a 5x5 visibility mask; key and banana bits; facing; previous action; normalized remaining time; and camera facing only while the camera tile is visible. Hidden camera facing uses a separate `unknown` category, so the observation cannot leak global phase. It does not receive global coordinates, full map state, status, reward, seed, or privileged distances. This is observation version `peel-symbolic-5x5-v2-camera-visible-facing`. The scripted teacher consumes the exact same observation dictionaries. It reconstructs a relative map using visible history and dead reckoning; demonstration labels never use environment internals or hidden generator state.

## Commands

From the repository root after the provided environment is installed:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m peel.replay --stage door --seed 0 --output artifacts/data/replays/scripted-door-0.json
.\.venv\Scripts\python.exe -m peel.train --config configs/smoke.json --run-dir runs/smoke
.\.venv\Scripts\python.exe -m peel.train --config configs/tiny.json --run-dir runs/tiny
.\.venv\Scripts\python.exe -m peel.train --config configs/tiny.json --run-dir runs/tiny --resume runs/tiny/latest.pt
.\.venv\Scripts\python.exe -m peel.train --config configs/tiny.json --run-dir runs/next-stage --warm-start runs/tiny/champion.pt
.\.venv\Scripts\python.exe -m peel.evaluate --checkpoint runs/tiny/final.pt --stage door --episodes 100 --output artifacts/data/tiny-eval.json
.\.venv\Scripts\python.exe -m peel.memory_benchmark --output artifacts/data/memory-benchmark.json
.\.venv\Scripts\python.exe -m peel.server
```

Training prints and writes only measured records. BC metrics and PPO training-batch metrics are labeled by phase and are never treated as held-out results. Random, post-BC, periodic, and final candidates run on fixed validation seeds within the one-to-two-million namespace. This row holdout is a deliberate geometry distribution shift rather than an IID random split. Champion promotion requires at least one validation success and then compares matched success count, followed by return. `artifacts/data/experiments.json` contains only those validation curves. The evaluation CLI defaults to the separate untouched test domain beginning at seed `2_000_000`. Resume restores RNG state, weights, optimizer, counters, and a fresh-episode seed cursor; immutable config differences are rejected. Exact mid-episode simulator/history continuation is intentionally unsupported and recorded in checkpoint metadata. `--warm-start` is the explicit staged-curriculum path: it loads weights into a new run while resetting optimizer and counters.

## Data format

`artifacts/data/replays/*.json` follows `BACKEND_CONTRACT.md`. Frames contain the initial state and every state after an action. `actions` contains integer action IDs, while `metrics.action_names` provides the stable mapping. Policy replays can also contain `metrics.action_probabilities`. `artifacts/data/experiments.json` is an array of published real run summaries; an empty array means no experiment has been run or published. Local `/api/experiments` falls back to actual JSONL records found beneath `runs/`.

The controlled cue benchmark is a separately labeled supervised diagnostic, not heist PPO evidence. At its final decision, every current observation is identical; only a cue at the first timestep identifies the target action. It reports measured held-out accuracy for the temporal transformer and an independently initialized model trained with only the last observation visible.

## Limitations

No learning or throughput result is assumed by the code. The narrow mirrored templates are an initial curriculum, not broad procedural generalization. Camera geometry is cardinal and deterministic. The teacher uses dead reckoning and intentionally skips failed demonstrations; its attempt and success counts are logged. Checkpoint selection/champion promotion and multi-seed statistical aggregation remain evaluation workflow decisions rather than automatic claims.
