# netris

A [Gymnasium](https://gymnasium.farama.org/) environment for **Game Boy Tetris**,
emulated with [PyBoy](https://github.com/Baekalfen/PyBoy).

- Placement-based action space: `rotation (0-3) * 10 + column (0-9)` (40 actions).
- Dict observation: board, current/next piece, row completion, holes,
  bumpiness and aggregate height.
- Line clears are confirmed against the game's own on-screen counter, so
  phantom clears (e.g. at game over) cannot happen.
- Optional 1-piece lookahead information via the on-screen NEXT panel
  (`NetrisEnvConfig(include_next_piece=True)`).

## Install

```bash
pip install netris
```

## ROM

The Game Boy Tetris ROM is **not** distributed with this package. Provide your
own dump:

```bash
export NETRIS_ROM=/path/to/tetris.gb
```

## Quickstart

```python
import gymnasium as gym
import netris  # registers "Netris/GameBoyTetris-v0"

env = gym.make("Netris/GameBoyTetris-v0")  # uses NETRIS_ROM
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

Or directly:

```python
from netris import NetrisEnvConfig, TetrisEnv

env = TetrisEnv("/path/to/tetris.gb", config=NetrisEnvConfig())
obs, info = env.reset()
```

## License

MIT. PyBoy is an LGPL-3.0 dependency; the Tetris ROM is copyrighted and must be
supplied by the user.
