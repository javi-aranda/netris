import gymnasium
import numpy as np
import pytest

from netris import NetrisEnvConfig, TetrisEnv


@pytest.mark.rom
def test_observation_space_keys(rom_path):
    env = TetrisEnv(rom_path, config=NetrisEnvConfig(include_next_piece=False))
    obs, _ = env.reset()
    assert set(obs.keys()) == {
        "board", "current_piece", "row_completion", "holes", "bumpiness",
        "aggregate_height",
    }
    assert obs["board"].shape == (1, 18, 10)
    env.close()

    env = TetrisEnv(rom_path, config=NetrisEnvConfig(include_next_piece=True))
    obs, _ = env.reset()
    assert "next_piece" in obs
    env.close()


@pytest.mark.rom
def test_step_returns_gymnasium_tuple(rom_path):
    env = TetrisEnv(rom_path)
    _, info = env.reset(seed=0)
    _, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "lines_cleared" in info
    env.close()


@pytest.mark.rom
def test_render_rgb_array(rom_path):
    env = TetrisEnv(rom_path, render_mode="rgb_array")
    env.reset()
    frame = env.render()
    assert frame.shape == (144, 160, 3)
    assert frame.dtype == np.uint8
    env.close()


@pytest.mark.rom
def test_seed_is_reproducible(rom_path):
    env_a = TetrisEnv(rom_path)
    env_b = TetrisEnv(rom_path)
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)
    assert np.array_equal(obs_a["board"], obs_b["board"])
    assert np.array_equal(obs_a["current_piece"], obs_b["current_piece"])
    env_a.close()
    env_b.close()


def test_registration():
    import netris  # noqa: F401

    assert "Netris/GameBoyTetris-v0" in gymnasium.registry
