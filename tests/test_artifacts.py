"""Evidence and portable demo integration checks, without requiring CUDA."""

import hashlib
import json

from fastapi.testclient import TestClient

from peel import server


def test_curated_replays_have_consistent_frames_and_real_probabilities():
    index = server.DATA / "replays" / "index.json"
    if not index.exists():
        return
    for item in json.loads(index.read_text()):
        replay = json.loads((index.parent / (item["id"] + ".json")).read_text())
        assert len(replay["frames"]) == len(replay["actions"]) + 1
        assert replay["success"] == (replay["frames"][-1]["status"] == "escaped")
        probabilities = replay["metrics"].get("action_probabilities")
        if probabilities is not None:
            assert len(probabilities) == len(replay["actions"])
            assert all(abs(sum(p) - 1) < 1e-5 for p in probabilities)


def test_bundled_champion_hash_and_cpu_inference():
    manifest = server.DATA / "model-manifest.json"
    if not manifest.exists():
        return
    metadata = json.loads(manifest.read_text())
    if metadata.get("storage") != "bundled":
        return
    path = server._safe_checkpoint("champion")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["sha256"]
    result = TestClient(server.app).post(
        "/api/play",
        json={
            "checkpoint": "champion",
            "stage": "banana",
            "seed": 1001100,
            "max_steps": 32,
        },
    )
    assert result.status_code == 200
    replay = result.json()
    assert replay["kind"] == "policy" and len(replay["actions"]) <= 32


def test_api_rejects_path_escape_and_missing_heist_banana():
    client = TestClient(server.app)
    assert (
        client.post("/api/play", json={"checkpoint": "../LICENSE"}).status_code == 404
    )
    grid = ["#########", "#S.....E#"] + ["#.......#"] * 6 + ["#########"]
    response = client.post("/api/validate", json={"grid": grid, "stage": "banana"})
    assert response.status_code == 200 and not response.json()["valid"]
    assert "banana" in " ".join(response.json()["errors"])
