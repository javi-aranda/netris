"""Gymnasium registration.

The environment requires a GameBoy Tetris ROM (not bundled). Provide it via the
`rom_path` kwarg or the `NETRIS_ROM` environment variable:

    import gymnasium as gym
    import netris  # registers the environment

    env = gym.make("Netris/GameBoyTetris-v0", rom_path="path/to/tetris.gb")
"""

import gymnasium

ENV_ID = "Netris/GameBoyTetris-v0"


def register_env():
    if ENV_ID not in gymnasium.registry:
        gymnasium.register(
            id=ENV_ID,
            entry_point="netris.envs.tetris_env:TetrisEnv",
        )
