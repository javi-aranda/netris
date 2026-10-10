import numpy as np

from netris.gameboy import board as B


def test_column_heights_and_aggregate():
    board = np.zeros((18, 10), dtype=np.uint8)
    board[17, :] = 1
    assert list(B.column_heights(board)) == [1] * 10
    assert B.aggregate_height(board) == 10

    board[15:, 3] = 1
    heights = list(B.column_heights(board))
    assert heights[3] == 3
    assert B.aggregate_height(board) == 12


def test_count_holes():
    board = np.zeros((18, 10), dtype=np.uint8)
    board[10:, 0] = 1
    board[11, 0] = 0  # single hole
    assert B.count_holes(board) == 1

    full = np.ones((18, 10), dtype=np.uint8)
    assert B.count_holes(full) == 0


def test_bumpiness():
    board = np.zeros((18, 10), dtype=np.uint8)
    board[17, :5] = 1
    board[15:, 5:] = 1
    assert B.bumpiness(board) == 2


def test_clear_lines():
    board = np.zeros((18, 10), dtype=np.uint8)
    board[17, :] = 1
    board[16, 0] = 1
    new_board, cleared = B.clear_lines(board)
    assert cleared == 1
    assert new_board[17, 0] == 1
    assert new_board[17].sum() == 1


def test_clear_multiple_lines():
    board = np.zeros((18, 10), dtype=np.uint8)
    board[16:, :] = 1
    board[15, 2] = 1
    new_board, cleared = B.clear_lines(board)
    assert cleared == 2
    assert new_board[17, 2] == 1


def test_first_filled_row():
    board = np.zeros((18, 10), dtype=np.uint8)
    assert B.first_filled_row(board, 0) == 18
    board[3, 4] = 1
    assert B.first_filled_row(board, 4) == 3


def test_wells():
    board = np.zeros((18, 10), dtype=np.uint8)
    board[17, 0] = 1
    board[17, 2] = 1
    assert B.wells(board) >= 1


def test_transitions_non_negative():
    board = np.zeros((18, 10), dtype=np.uint8)
    board[17, :] = 1
    assert B.row_transitions(board) == 2
    assert B.column_transitions(board) >= 0
