import pytest

from peel.env import PeelEnv
from peel.maps import camera_timed_route_exists, generate_map, validate_map

DOOR_MAP = [
    "#########",
    "#...#...#",
    "#...#...#",
    "#S..D.BE#",
    "#...#...#",
    "#K..#...#",
    "#...#...#",
    "#...#...#",
    "#########",
]


def test_generated_maps_are_reproducible_and_valid():
    for stage in ("exit", "banana", "door", "camera"):
        assert generate_map(stage, 91) == generate_map(stage, 91)
        assert validate_map(generate_map(stage, 91)).valid


def test_procedural_variation_and_domains_are_geometry_disjoint():
    train = {tuple(generate_map("door", seed)) for seed in range(30)}
    validation = {tuple(generate_map("door", 1_000_000 + seed)) for seed in range(30)}
    test = {tuple(generate_map("door", 2_000_000 + seed)) for seed in range(30)}
    assert len(train) > 10 and len(validation) > 10 and len(test) > 10
    assert (
        train.isdisjoint(validation)
        and train.isdisjoint(test)
        and validation.isdisjoint(test)
    )


def test_generated_camera_maps_have_a_timed_solution():
    for seed in range(30):
        grid = generate_map("camera", seed)
        assert camera_timed_route_exists(grid, seed % 4), seed


def test_validator_rejects_bad_shape_vocabulary_boundary_and_dependency():
    assert not validate_map(["short"]).valid
    bad = DOOR_MAP.copy()
    bad[1] = "#...X...#"
    assert any("unsupported" in error for error in validate_map(bad).errors)
    bad = DOOR_MAP.copy()
    bad[0] = "#.......#"
    assert any("boundary" in error for error in validate_map(bad).errors)
    bad = [row.replace("K", ".") for row in DOOR_MAP]
    assert any("requires a key" in error for error in validate_map(bad).errors)


def test_key_door_banana_escape_transition_chain_and_no_reward_farming():
    env = PeelEnv("door", grid=DOOR_MAP)
    obs, _ = env.reset(seed=0)
    assert tuple(obs["inventory"]) == (0, 0)
    env.step(1)  # face south
    env.step(2)  # y=4
    _, first_key, *_ = env.step(3)
    _, repeat_key, *_ = env.step(3)
    assert env.has_key and first_key > repeat_key
    env.step(0)  # east
    env.step(2)
    env.step(2)  # x=3
    env.step(0)
    env.step(2)
    env.step(1)  # north to y=3, then east
    _, door_reward, *_ = env.step(3)
    _, repeat_door, *_ = env.step(3)
    assert env.grid[3][4] == "d" and not env.has_key and door_reward > repeat_door
    env.step(2)
    env.step(2)  # x=5
    _, banana_reward, *_ = env.step(3)
    _, repeat_banana, *_ = env.step(3)
    assert env.has_banana and banana_reward > repeat_banana
    env.step(2)
    _, reward, terminated, truncated, info = env.step(2)
    assert terminated and not truncated and reward > 9 and info["status"] == "escaped"


def test_camera_capture_and_finite_deadline_semantics():
    camera_map = [
        "#########",
        "#S.C...E#",
        "#.......#",
        "#.......#",
        "#.......#",
        "#.......#",
        "#.......#",
        "#.......#",
        "#########",
    ]
    env = PeelEnv("camera", max_steps=5, grid=camera_map)
    env.reset(seed=2)  # west-facing camera sees the agent
    _, reward, terminated, truncated, info = env.step(4)
    assert terminated and not truncated and reward < -4 and info["status"] == "caught"
    env = PeelEnv("exit", max_steps=1)
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(4)
    assert terminated and not truncated and info["status"] == "timeout"


def test_observation_hides_global_state_and_occludes_behind_walls():
    env = PeelEnv("door", grid=DOOR_MAP)
    obs, _ = env.reset(seed=0)
    assert set(obs) == {
        "tiles",
        "visible",
        "inventory",
        "direction",
        "previous_action",
        "camera_direction",
        "time",
    }
    assert obs["tiles"].shape == (5, 5)
    # Insert a wall one cell east and banana two cells east; the banana is in
    # the local square but invisible through the wall.
    env.grid[3][2] = "#"
    env.grid[3][3] = "B"
    obs = env._observation()
    assert obs["visible"][2, 4] == 0 and obs["tiles"][2, 4] == 0
    assert "agent" not in obs and "camera" not in obs and "reward" not in obs


def test_step_after_terminal_requires_reset():
    env = PeelEnv("exit", max_steps=1)
    env.reset(seed=0)
    env.step(4)
    with pytest.raises(RuntimeError):
        env.step(4)
