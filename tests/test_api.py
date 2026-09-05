from fastapi.testclient import TestClient

from peel.server import app

client = TestClient(app)


def test_health_and_empty_or_real_experiments():
    health = client.get("/api/health")
    assert health.status_code == 200 and health.json()["status"] == "ok"
    experiments = client.get("/api/experiments")
    assert experiments.status_code == 200 and isinstance(experiments.json(), list)


def test_validate_and_scripted_play_contract():
    grid = [
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
    assert client.post("/api/validate", json={"grid": grid}).json() == {
        "valid": True,
        "errors": [],
        "validation_scope": "geometry_only",
    }
    response = client.post(
        "/api/play", json={"seed": 0, "stage": "exit", "checkpoint": "scripted"}
    )
    assert response.status_code == 200
    value = response.json()
    assert value["kind"] == "scripted" and value["success"]
    assert len(value["frames"]) == len(value["actions"]) + 1


def test_play_requires_explicit_model_or_scripted_label():
    response = client.post("/api/play", json={"seed": 0, "stage": "exit"})
    assert response.status_code == 409


def test_invalid_grid_and_episode_bound_are_rejected():
    response = client.post(
        "/api/play", json={"stage": "door", "checkpoint": "scripted", "grid": ["bad"]}
    )
    assert response.status_code == 422
    assert (
        client.post(
            "/api/play", json={"checkpoint": "scripted", "max_steps": 1000}
        ).status_code
        == 422
    )
