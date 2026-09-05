import pytest

from peel import curriculum


def test_mastery_boundary():
    assert curriculum.mastered(8, 10, 0.8)
    assert not curriculum.mastered(7, 10, 0.8)
    with pytest.raises(ValueError):
        curriculum.mastered(0, 0, 0.8)


def test_curriculum_stops_when_budget_exhausted(tmp_path, monkeypatch):
    calls = []

    def train(config, directory, warm_start=None):
        calls.append(config["stage"])
        directory.mkdir()
        return directory / "final.pt"

    monkeypatch.setattr(curriculum, "train", train)
    monkeypatch.setattr(curriculum, "load_policy", lambda *a: (None, {}))
    monkeypatch.setattr(curriculum, "evaluate", lambda *a: ({"successes": 0}, []))
    report = curriculum.run_curriculum(
        {"seed": 1, "device": "cpu"}, tmp_path / "run", ["exit", "banana"], blocks=2
    )
    assert calls == ["exit", "exit"]
    assert report["outcome"] == "budget_exhausted_before_mastery"
    assert report["history"][0]["probe_seed"] < 1_000_000


def test_curriculum_advances_on_mastery(tmp_path, monkeypatch):
    calls = []

    def train(config, directory, warm_start=None):
        calls.append((config["stage"], warm_start))
        directory.mkdir()
        return directory / "final.pt"

    monkeypatch.setattr(curriculum, "train", train)
    monkeypatch.setattr(curriculum, "load_policy", lambda *a: (None, {}))
    monkeypatch.setattr(curriculum, "evaluate", lambda *a: ({"successes": 32}, []))
    report = curriculum.run_curriculum(
        {"seed": 1, "device": "cpu"}, tmp_path / "run", ["exit", "banana"]
    )
    assert [c[0] for c in calls] == ["exit", "banana"]
    assert calls[1][1] is not None
    assert report["outcome"] == "complete"
