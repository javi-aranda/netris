"""Netris: a Gymnasium environment for GameBoy Tetris."""

from .envs import NetrisEnvConfig, TetrisEnv
from .registration import ENV_ID, register_env

__version__ = "0.1.0"

__all__ = ["ENV_ID", "NetrisEnvConfig", "TetrisEnv", "__version__", "register_env"]

register_env()
