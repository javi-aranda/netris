"""Pure board-metric helpers shared by the environment and the teacher."""

import numpy as np

BOARD_HEIGHT = 18
BOARD_WIDTH = 10


def first_filled_row(board, col):
    column = board[:, col]
    index = int(np.argmax(column))
    if column[index] == 0:
        return BOARD_HEIGHT
    return index


def clear_lines(board):
    """Remove complete rows and return (new_board, lines_cleared)."""
    kept_rows = [row for row in board if row.sum() < BOARD_WIDTH]
    cleared = BOARD_HEIGHT - len(kept_rows)
    if cleared == 0:
        return board, 0
    new_board = np.zeros_like(board)
    if kept_rows:
        new_board[cleared:, :] = np.array(kept_rows, dtype=board.dtype)
    return new_board, cleared


def column_heights(board):
    has_filled = board.any(axis=0)
    first = np.argmax(board, axis=0)
    return np.where(has_filled, BOARD_HEIGHT - first, 0)


def aggregate_height(board):
    return int(column_heights(board).sum())


def count_holes(board):
    above = np.maximum.accumulate(board, axis=0)
    return int(((above == 1) & (board == 0)).sum())


def bumpiness(board):
    heights = column_heights(board)
    return int(np.abs(np.diff(heights)).sum())


def row_transitions(board):
    horizontal = int(np.abs(np.diff(board.astype(np.int64), axis=1)).sum())
    return horizontal + int(board[:, 0].sum()) + int(board[:, -1].sum())


def column_transitions(board):
    vertical = int(np.abs(np.diff(board.astype(np.int64), axis=0)).sum())
    return vertical + int((board[-1, :] == 0).sum())


def wells(board):
    left = np.concatenate(
        [np.ones((BOARD_HEIGHT, 1), dtype=board.dtype), board[:, :-1]], axis=1
    )
    right = np.concatenate(
        [board[:, 1:], np.ones((BOARD_HEIGHT, 1), dtype=board.dtype)], axis=1
    )
    well_cell = (board == 0) & (left == 1) & (right == 1)
    total = 0
    depth = np.zeros(BOARD_WIDTH, dtype=np.int64)
    for row in range(BOARD_HEIGHT):
        depth = np.where(well_cell[row], depth + 1, 0)
        total += int(depth.sum())
    return total
