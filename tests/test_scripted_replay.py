from peel.env import PeelEnv
from peel.scripted import scripted_episode


def test_teacher_api_accepts_only_observation_and_succeeds_on_core_stages():
    for stage in ("exit", "banana", "door"):
        for seed in (0, 1):
            replay, examples = scripted_episode(PeelEnv(stage), seed)
            assert replay["success"], (stage, seed, replay["metrics"])
            assert examples and all(
                set(obs)
                == {
                    "tiles",
                    "visible",
                    "inventory",
                    "direction",
                    "previous_action",
                    "camera_direction",
                    "time",
                }
                for obs, _ in examples
            )


def test_replay_has_initial_plus_one_frame_per_action_and_matches_terminal_state():
    replay, _ = scripted_episode(PeelEnv("door"), 0)
    assert len(replay["frames"]) == len(replay["actions"]) + 1
    assert replay["steps"] == len(replay["actions"])
    assert replay["frames"][-1]["status"] == "escaped"
    assert replay["kind"] == "scripted"
    assert replay["metrics"]["action_names"] == [
        "left",
        "right",
        "forward",
        "interact",
        "wait",
    ]
