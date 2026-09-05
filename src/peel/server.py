from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .env import PeelEnv
from .evaluate import load_policy, policy_episode
from .maps import STAGES, validate_map
from .scripted import scripted_episode

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "artifacts" / "data"
RUNS = ROOT / "runs"


class GridRequest(BaseModel):
    grid: list[str]


class PlayRequest(BaseModel):
    seed: int = 0
    stage: str = "door"
    checkpoint: str | None = None
    grid: list[str] | None = None
    max_steps: int = Field(96, ge=1, le=256)


app = FastAPI(title="Peel API", version=__version__)


def _safe_checkpoint(name: str) -> Path:
    if name == "champion":
        manifest = DATA / "model-manifest.json"
        if not manifest.exists():
            raise HTTPException(
                409,
                "no champion has been promoted; run evaluation-backed training first",
            )
        name = str(json.loads(manifest.read_text(encoding="utf-8")).get("champion", ""))
    candidate = (RUNS / name).resolve()
    if RUNS.resolve() not in candidate.parents or not candidate.is_file():
        raise HTTPException(404, "checkpoint not found beneath runs/")
    return candidate


def _replay_files() -> list[Path]:
    directory = DATA / "replays"
    return sorted(
        (path for path in directory.glob("*.json") if path.name != "index.json"),
        key=lambda p: p.name,
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "inference_device": "cpu",
    }


@app.get("/api/experiments")
def experiments() -> list[dict[str, Any]]:
    published = DATA / "experiments.json"
    if published.exists():
        value = json.loads(published.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else value.get("experiments", [])
    summaries = []
    for metrics_path in RUNS.glob("*/metrics.jsonl"):
        records = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if records:
            summaries.append(
                {
                    "id": metrics_path.parent.name,
                    "source": "local-run",
                    "curve": records,
                }
            )
    return summaries


@app.get("/api/replays")
def replays() -> list[dict[str, Any]]:
    result = []
    for path in _replay_files():
        replay = json.loads(path.read_text(encoding="utf-8"))
        result.append(
            {
                key: replay[key]
                for key in ("id", "label", "kind", "seed", "success", "steps", "stage")
            }
        )
    return result


@app.get("/api/replays/{replay_id}")
def replay(replay_id: str) -> dict[str, Any]:
    for path in _replay_files():
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("id") == replay_id:
            return value
    raise HTTPException(404, "replay not found")


@app.post("/api/validate")
def validate(request: GridRequest) -> dict[str, Any]:
    return validate_map(request.grid).as_dict()


@app.post("/api/play")
def play(request: PlayRequest) -> dict[str, Any]:
    if request.stage not in STAGES:
        raise HTTPException(422, f"stage must be one of {STAGES}")
    if request.grid is not None:
        result = validate_map(request.grid)
        if not result.valid:
            raise HTTPException(422, {"valid": False, "errors": list(result.errors)})
    if request.checkpoint == "scripted":
        replay_value, _ = scripted_episode(
            PeelEnv(request.stage, request.max_steps, request.grid), request.seed
        )
        return replay_value
    if not request.checkpoint:
        raise HTTPException(
            409,
            "no model selected; set checkpoint to a path beneath runs/ or explicitly use 'scripted'",
        )
    path = _safe_checkpoint(request.checkpoint)
    try:
        model, config = load_policy(path, "cpu")
        config = {**config, "max_steps": request.max_steps}
        return policy_episode(
            model, config, request.stage, request.seed, request.grid, device="cpu"
        )
    except (ValueError, KeyError, RuntimeError) as error:
        raise HTTPException(422, f"checkpoint could not be loaded: {error}") from error


WEB = ROOT / "web"
ARTIFACTS = ROOT / "artifacts"
if ARTIFACTS.exists():
    app.mount("/artifacts", StaticFiles(directory=ARTIFACTS), name="artifacts")
if WEB.exists():
    # Registered last so /api routes retain precedence.
    app.mount("/", StaticFiles(directory=WEB, html=True), name="frontend")


def main() -> None:
    import uvicorn

    uvicorn.run("peel.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
