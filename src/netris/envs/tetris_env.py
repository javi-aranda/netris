import logging
import os
from typing import ClassVar

import numpy as np
from gymnasium import Env, spaces
from pyboy import PyBoy

from ..gameboy.pieces import (
    PIECE_TILES,
    TETROMINO_NUMBERS,
    TETROMINO_TYPES,
    TILE_TO_PIECE,
)
from .config import NetrisEnvConfig
from .metrics import MetricsCollector, NullMetricsCollector

logger = logging.getLogger(__name__)

ROTATE_TICKS = 2
MOVE_TICKS = 2
DROP_TICKS = 2

MAX_MOVE_ATTEMPTS = 20
MAX_DROP_ATTEMPTS = 40

# After a piece locks the game may blink completed rows on/off for ~40 frames
# before removing them, so the board must be considered settled only after a
# window longer than the blink gap (SETTLE_CHUNK * SETTLE_STABLE_CHUNKS ticks).
SETTLE_CHUNK = 4
SETTLE_STABLE_CHUNKS = 3
SETTLE_MAX_FRAMES = 240

# The game's own line counter (screen digits) updates slightly after the clear
# animation; poll it a bit longer when completed rows were seen.
LINES_CONFIRM_CHUNK = 4
LINES_CONFIRM_MAX_FRAMES = 64


class TetrisEnv(Env):
    """GameBoy Tetris as a placement-based Gymnasium environment.

    Action: ``rotation (0-3) * 10 + column (0-9)``; one step places one piece.
    The ROM is never bundled: pass ``rom_path`` or set ``NETRIS_ROM``.
    """

    metadata: ClassVar[dict] = {"render_modes": ["rgb_array", "human"], "render_fps": 60}

    def __init__(self, rom_path=None, render_mode=None, log_level="error",
                 config=None, window=None) -> None:
        self.config = config if config is not None else NetrisEnvConfig()

        self.rom_path = rom_path or os.environ.get("NETRIS_ROM", "")
        if not self.rom_path:
            raise ValueError(
                "No ROM path provided. Pass rom_path or set the NETRIS_ROM "
                "environment variable. The Tetris ROM is not distributed with "
                "this package."
            )
        if not os.path.exists(self.rom_path):
            raise FileNotFoundError(f"ROM file not found: {self.rom_path}")

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode {render_mode!r}; "
                f"choose one of {self.metadata['render_modes']} or None."
            )
        self.render_mode = render_mode
        self.window = window or ("SDL2" if render_mode == "human" else "null")
        self.include_next_piece = self.config.include_next_piece

        logging.basicConfig(level=log_level.upper())

        self.board_shape = (18, 10)

        # Action: placement = rotation (0-3) * 10 + column (0-9)
        self.action_space = spaces.Discrete(40)

        observation_spaces = {
            "board": spaces.Box(
                low=0, high=1, shape=(1, 18, 10), dtype=np.float32
            ),
            "current_piece": spaces.Box(
                low=0, high=1, shape=(7,), dtype=np.float32
            ),
            "row_completion": spaces.Box(
                low=0, high=10, shape=(18,), dtype=np.float32
            ),
            "holes": spaces.Box(low=0, high=180, shape=(1,), dtype=np.float32),
            "bumpiness": spaces.Box(low=0, high=180, shape=(1,), dtype=np.float32),
            "aggregate_height": spaces.Box(low=0, high=180, shape=(1,), dtype=np.float32),
        }
        if self.include_next_piece:
            observation_spaces["next_piece"] = spaces.Box(
                low=0, high=1, shape=(7,), dtype=np.float32
            )
        self.observation_space = spaces.Dict(observation_spaces)

        self.current_score = 0
        self.lines_cleared = 0
        self.n_deaths = 0
        self.episode_steps = 0
        self.pieces_placed = 0
        self.steps_since_last_line_clear = 0
        self.piece_rotation = 0

        self.previous_locked_board = None
        self.previous_holes = 0
        self.previous_height = 0
        self.previous_aggregate_height = 0
        self.previous_bumpiness = 0

        # Optional capture of frames when lines are cleared. Disabled by default
        # so training workers do not pay the copy cost; enabled for eval only.
        self.capture_line_clear = False
        self.line_clear_frames = []

        # Optional per-tick frame recording for visualization. Disabled by
        # default; when enabled, gameplay ticks render and frames are stored in
        # `recorded_frames` (drained by the caller).
        self.record_frames = False
        self.record_every = 1
        self.max_recorded_frames = 10000
        self.recorded_frames = []
        self._record_counter = 0

        self.collector = (
            MetricsCollector() if self.config.collect_metrics
            else NullMetricsCollector()
        )

        # Address of the `ld a,(0xC213)` instruction that loads the next
        # tetromino when spawning. It differs between ROM revisions, so it is
        # looked up in the ROM instead of hardcoded.
        self._next_piece_load_addr = self._find_next_piece_load_addr()

        self.emulator = PyBoy(
            self.rom_path,
            window=self.window,
        )
        assert self.emulator.cartridge_title == "TETRIS"
        self.emulator.set_emulation_speed(0)
        self.tetris = self.emulator.game_wrapper
        self.tetris.start_game()

        self.reset()

    def _find_next_piece_load_addr(self):
        with open(self.rom_path, "rb") as rom_file:
            rom = rom_file.read()
        return rom.find(bytes([0xFA, 0x13, 0xC2]))

    def force_next_tetromino(self, shape):
        """Test helper: force every subsequently spawned piece to `shape`."""
        addr = self._next_piece_load_addr
        if addr < 0:
            raise RuntimeError("Could not locate next-tetromino load in ROM")
        self.emulator.memory[0, addr] = 0x3E  # LD A, n
        self.emulator.memory[0, addr + 1] = TETROMINO_NUMBERS[shape]
        self.emulator.memory[0, addr + 2] = 0x00  # NOP

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        timer_div = int(seed) & 0xFF if seed is not None else None
        self.tetris.reset_game(timer_div=timer_div)

        for _ in range(50):
            self.emulator.tick(render=False)

        locked_board = self.get_locked_board()
        self.previous_locked_board = locked_board.copy()
        self.previous_holes = self.count_holes(locked_board)
        self.previous_height = self.get_max_height(locked_board)
        self.previous_aggregate_height = self.get_aggregate_height(locked_board)
        self.previous_bumpiness = self.get_bumpiness(locked_board)

        self.current_score = 0
        self.lines_cleared = 0
        self.episode_steps = 0
        self.pieces_placed = 0
        self.steps_since_last_line_clear = 0
        self.piece_rotation = 0
        self.line_clear_frames = []

        self.collector.reset_episode()

        observation = self.get_observation()

        logger.info(
            f"[EPISODE_START] height={self.previous_height}, holes={self.previous_holes}, "
            f"bumpiness={self.previous_bumpiness}"
        )

        return observation, {}

    def _read_active_cells(self):
        """Read the falling piece cells from OAM: list of (col, row, tile_id)."""
        oam = self.emulator.memory[0xFE00:0xFEA0]
        cells = []
        for i in range(0, 160, 4):
            tile = int(oam[i + 2])
            if tile not in PIECE_TILES:
                continue
            x = int(oam[i + 1]) - 8
            y = int(oam[i]) - 16
            col = x // 8 - 2
            row = y // 8
            if 0 <= col < self.board_shape[1] and 0 <= row < self.board_shape[0]:
                cells.append((col, row, tile))
        return cells

    def get_current_piece_name(self):
        cells = self._read_active_cells()
        if not cells:
            return None
        return TILE_TO_PIECE[cells[0][2]]

    def get_current_piece_onehot(self):
        onehot = np.zeros(7, dtype=np.float32)
        name = self.get_current_piece_name()
        if name is not None:
            onehot[TETROMINO_TYPES.index(name)] = 1.0
        return onehot

    def get_next_piece_name(self):
        """Next piece as shown in the on-screen NEXT panel.

        Read from the preview sprites (screen state), never from ROM/RNG, so it
        only uses information a human player can see.
        """
        oam = self.emulator.memory[0xFE00:0xFEA0]
        for i in range(0, 160, 4):
            tile = int(oam[i + 2])
            if tile not in PIECE_TILES:
                continue
            x = int(oam[i + 1]) - 8
            col = x // 8 - 2
            if col >= self.board_shape[1]:
                return TILE_TO_PIECE[tile]
        # Fallback: the same value the game uses to draw the preview
        try:
            return self.tetris.next_tetromino()
        except Exception:  # noqa: BLE001
            return None

    def get_next_piece_onehot(self):
        onehot = np.zeros(7, dtype=np.float32)
        name = self.get_next_piece_name()
        if name is not None:
            onehot[TETROMINO_TYPES.index(name)] = 1.0
        return onehot

    def get_locked_board(self):
        background = np.asarray(self.emulator.tilemap_background[2:12, 0:18])
        # Only actual tetromino tiles count as filled. From pyboy's
        # game_wrapper_tetris.py: BLANK=47, BLACK(losing)=135, and pieces are
        # 129,130,131,132,133,134 (J,Z,O,L,T,S) and 128,136,137,138,139,143 (I).
        # Everything else (UI text, preview, numbers) is treated as empty.
        piece_tiles = {128, 129, 130, 131, 132, 133, 134, 136, 137, 138, 139, 143}
        board = np.isin(background, list(piece_tiles)).astype(np.uint8)
        return board

    def get_board(self):
        """Board with the active piece rendered on top (18x10, uint8)."""
        board = self.get_locked_board().copy()
        for col, row, _tile in self._read_active_cells():
            board[row, col] = 1

        self.observation = board
        return self.observation

    def render(self):
        """Gymnasium render: "rgb_array" returns the GameBoy screen, "human"
        updates the PyBoy window (requires a display)."""
        if self.render_mode is None:
            return None
        self.emulator.tick(1, render=True)
        if self.render_mode == "rgb_array":
            return np.asarray(
                self.emulator.screen.image.convert("RGB"), dtype=np.uint8
            )
        return None

    def close(self):
        try:
            self.emulator.stop(save=False)
        except Exception as error:  # noqa: BLE001
            logger.debug("Error while closing PyBoy: %s", error)

    def get_observation(self):
        board = self.get_board()
        locked_board = self.get_locked_board()

        holes = self.count_holes(locked_board)
        bumpiness = self.get_bumpiness(locked_board)
        aggregate_height = self.get_aggregate_height(locked_board)
        row_completion = self.get_row_completion(board)

        observation = {
            "board": board.reshape(1, 18, 10).astype(np.float32),
            "current_piece": self.get_current_piece_onehot(),
            "row_completion": row_completion.astype(np.float32),
            "holes": np.array([holes], dtype=np.float32),
            "bumpiness": np.array([bumpiness], dtype=np.float32),
            "aggregate_height": np.array([aggregate_height], dtype=np.float32),
        }
        if self.include_next_piece:
            observation["next_piece"] = self.get_next_piece_onehot()

        return observation

    def _get_piece_column(self):
        cells = self._read_active_cells()
        if not cells:
            return None
        return int(min(col for col, _row, _tile in cells))

    def _press(self, button, ticks):
        self.emulator.button(button, delay=1)
        self.emulator.tick(ticks, render=self.record_frames)
        self._capture_frame()

    def _capture_frame(self):
        if not self.record_frames:
            return
        self._record_counter += 1
        if self.record_every > 1 and self._record_counter % self.record_every != 0:
            return
        frame = self.emulator.screen.image.copy()
        if self.max_recorded_frames <= 0 or len(self.recorded_frames) < self.max_recorded_frames:
            self.recorded_frames.append(frame)

    def capture_screen(self):
        """Render one frame and return a copy of the current screen image.

        Gameplay ticks use render=False for speed, which leaves the renderer
        buffer stale, so a rendered frame is forced before grabbing it.
        """
        self.emulator.tick(1, render=True)
        return self.emulator.screen.image.copy()

    def _settle(self):
        """Tick until the board stops changing (past line-clear blinking).

        Returns the settled board, a screenshot taken while completed rows were
        still visible (only when capture is enabled) and whether completed rows
        were seen at all.
        """
        prev = self.get_locked_board()
        stable = 0
        frames = 0
        captured = None
        saw_complete_rows = False
        while frames < SETTLE_MAX_FRAMES:
            self.emulator.tick(SETTLE_CHUNK, render=self.record_frames)
            self._capture_frame()
            frames += SETTLE_CHUNK
            current = self.get_locked_board()
            if (current.sum(axis=1) == 10).any():
                saw_complete_rows = True
                if self.capture_line_clear and captured is None:
                    captured = self.capture_screen()
            if np.array_equal(current, prev):
                stable += 1
                if stable >= SETTLE_STABLE_CHUNKS:
                    break
            else:
                stable = 0
                prev = current
        return self.get_locked_board(), captured, saw_complete_rows

    def _await_line_counter(self, lines_before, max_frames=LINES_CONFIRM_MAX_FRAMES):
        """Wait (bounded) for the game's on-screen line counter to advance."""
        frames = 0
        while frames < max_frames and self.tetris.lines <= lines_before:
            self.emulator.tick(LINES_CONFIRM_CHUNK, render=self.record_frames)
            self._capture_frame()
            frames += LINES_CONFIRM_CHUNK
        return max(0, self.tetris.lines - lines_before)

    def _await_active_piece(self, max_frames=48):
        """Wait (bounded) for the next piece to spawn after a clear."""
        frames = 0
        while frames < max_frames and not self._read_active_cells():
            self.emulator.tick(SETTLE_CHUNK, render=self.record_frames)
            self._capture_frame()
            frames += SETTLE_CHUNK

    def step(self, action):
        """
        Action is a placement: rotation (0-3) * 10 + column (0-9).
        The env rotates the piece, moves it to the target column, and hard drops.
        One step = one piece placed.
        """
        target_rotation = action // 10
        target_column = action % 10

        wrapper_lines_before = self.tetris.lines

        self.collector.register_action(action)

        # Apply rotations to reach target rotation
        rotations_needed = (target_rotation - self.piece_rotation) % 4
        for _ in range(rotations_needed):
            self._press("a", ROTATE_TICKS)
            self.collector.register_rotation()
        self.piece_rotation = target_rotation % 4

        # Move to target column
        for _ in range(MAX_MOVE_ATTEMPTS):
            current_col = self._get_piece_column()
            if current_col is None or current_col == target_column:
                break
            self._press("right" if current_col < target_column else "left", MOVE_TICKS)

        # Hard drop: press down until the locked board changes
        prev_board = self.get_locked_board()
        for _ in range(MAX_DROP_ATTEMPTS):
            self._press("down", DROP_TICKS)
            current_board = self.get_locked_board()
            if not np.array_equal(current_board, prev_board):
                break
            prev_board = current_board

        locked_board, captured_frame, saw_complete_rows = self._settle()

        self.episode_steps += 1
        self.steps_since_last_line_clear += 1

        board_before = self.previous_locked_board
        piece_locked = not np.array_equal(locked_board, board_before)

        done = self.game_over(locked_board)
        if not done:
            self._await_active_piece()

        # Line clears are only counted when the game's own line counter
        # advances. The board cell count is used just as evidence that a clear
        # is in progress (it must not be trusted: on game over the kill screen
        # replaces the board with tiles that are not counted as filled).
        wrapper_delta = max(0, self.tetris.lines - wrapper_lines_before)
        if not done and saw_complete_rows and wrapper_delta == 0:
            wrapper_delta = self._await_line_counter(wrapper_lines_before)

        catch_up = max(0, self.tetris.lines - self.lines_cleared)
        lines_delta = max(wrapper_delta, catch_up)

        # Build the observation *after* all ticking so it always reflects the
        # exact state the next action will be applied to.
        observation = self.get_observation()

        if saw_complete_rows and lines_delta == 0 and not done:
            logger.warning(
                "[LINE_DETECT] Completed rows seen but the game line counter did not advance"
            )

        if piece_locked:
            self.pieces_placed += 1
            self.piece_rotation = 0
            self.lines_cleared += lines_delta
            if lines_delta > 0:
                self.steps_since_last_line_clear = 0
                self.collector.register_line_clear(lines_delta)

            current_holes = self.count_holes(locked_board)
            holes_delta = current_holes - self.previous_holes
            current_height = self.get_max_height(locked_board)
            current_aggregate_height = self.get_aggregate_height(locked_board)
            current_bumpiness = self.get_bumpiness(locked_board)

            if lines_delta > 0:
                new_cells = (locked_board == 1) & (board_before == 0)
                new_cols = np.where(new_cells.any(axis=0))[0]
                placement_col = int(new_cols.min()) if len(new_cols) else target_column
                self.collector.register_piece_placed(placement_col, current_height)

                logger.info(
                    f"[LINE_CLEAR] lines={lines_delta}, total_lines={self.lines_cleared}, "
                    f"piece#{self.pieces_placed}"
                )
                if self.capture_line_clear and captured_frame is not None:
                    self.line_clear_frames.append(
                        (lines_delta, self.lines_cleared, captured_frame)
                    )
            elif self.pieces_placed % 10 == 0:
                logger.info(
                    f"[PIECE #{self.pieces_placed}] height={current_height}, "
                    f"holes={current_holes}, bumpy={current_bumpiness}"
                )

            reward = self.calculate_lock_reward(
                lines_delta, holes_delta, current_holes,
                current_height, current_aggregate_height, current_bumpiness
            )

            self.previous_locked_board = locked_board.copy()
            self.previous_holes = current_holes
            self.previous_height = current_height
            self.previous_aggregate_height = current_aggregate_height
            self.previous_bumpiness = current_bumpiness

            self.collector.update_max_height(current_height)
            self.collector.record_bumpiness(current_bumpiness)
            self.collector.record_holes(current_holes)
        else:
            reward = 0.0

        truncated = self.steps_since_last_line_clear > self.config.watchdog_steps
        if truncated:
            logger.info(
                "[WATCHDOG] Truncating episode: no line clear in %d steps",
                self.config.watchdog_steps,
            )

        if done:
            reward = self.config.game_over_penalty
            self.n_deaths += 1
            logger.info(
                f"[EPISODE_END] steps={self.episode_steps}, pieces={self.pieces_placed}, "
                f"lines_cleared={self.lines_cleared}, cum_reward={self.current_score:.2f}"
            )

        self.current_score += reward

        info = self.get_info()

        if done or truncated:
            summary = self.collector.episode_summary(
                episode=self.n_deaths,
                steps=self.episode_steps,
                pieces=self.pieces_placed,
                total_lines=self.lines_cleared,
                total_reward=self.current_score,
            )
            info["episode_summary"] = summary

        return observation, reward, done, truncated, info

    def calculate_lock_reward(self, lines_delta, holes_delta, current_holes,
                               current_height, aggregate_height, bumpiness):
        config = self.config
        if lines_delta >= 4:
            reward_lines = config.tetris_reward
        else:
            reward_lines = config.line_clear_rewards.get(lines_delta, 0.0)

        reward_holes = -config.hole_penalty * max(0, holes_delta)
        reward_height = -config.aggregate_height_penalty * aggregate_height
        reward_bumpiness = -config.bumpiness_penalty * bumpiness

        reward = (
            reward_lines + reward_holes + reward_height + reward_bumpiness
            + config.survival_bonus
        )

        self.collector.register_reward_breakdown(
            reward_lines=reward_lines,
            reward_holes=reward_holes,
            reward_height=reward_height,
            reward_bumpiness=reward_bumpiness,
            reward_clean=config.survival_bonus,
            reward_total=reward,
        )

        return reward

    def game_over(self, board=None):
        return self.tetris.game_over()

    def get_info(self):
        return {
            "score": self.current_score,
            "lines_cleared": self.lines_cleared,
            "pieces_placed": self.pieces_placed,
            "n_deaths": self.n_deaths,
        }

    def get_column_height(self, board, col):
        for i in range(len(board)):
            if board[i][col] == 1:
                return len(board) - i
        return 0

    def get_bumpiness(self, board):
        heights = [self.get_column_height(board, col) for col in range(board.shape[1])]
        return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))

    def get_aggregate_height(self, board):
        heights = [self.get_column_height(board, col) for col in range(board.shape[1])]
        return sum(heights)

    def get_max_height(self, board):
        heights = [self.get_column_height(board, col) for col in range(board.shape[1])]
        return max(heights) if heights else 0

    def count_holes(self, board):
        holes = 0
        for col in range(board.shape[1]):
            column = board[:, col]
            filled = False
            for cell in column:
                if cell == 1:
                    filled = True
                elif filled and cell == 0:
                    holes += 1
        return holes

    def get_row_completion(self, board):
        return board.sum(axis=1).astype(np.float32)
