from netris.gameboy.pieces import (
    PIECE_SHAPES,
    PIECE_TILES,
    TETROMINO_TYPES,
    TILE_TO_PIECE,
)


def test_shapes_have_four_cells():
    for shape, rotations in PIECE_SHAPES.items():
        assert len(rotations) == 4, shape
        for offsets in rotations:
            assert len(offsets) == 4, (shape, offsets)
            assert len(set(offsets)) == 4


def test_offsets_are_normalized():
    for rotations in PIECE_SHAPES.values():
        for offsets in rotations:
            assert min(row for row, _ in offsets) == 0
            assert min(col for _, col in offsets) == 0


def test_tile_mapping_covers_all_pieces():
    assert set(TILE_TO_PIECE.values()) == set(TETROMINO_TYPES)
    assert set(TILE_TO_PIECE.keys()).issubset(PIECE_TILES)
