"""Tests for the image grid and its drawing. No dataset needed."""

import numpy as np
import pytest

from offroad.viz.grid_overlay import draw_grid
from offroad.vlm.grid import ImageGrid

GRID = ImageGrid(height=1200, width=1920)  # RELLIS-3D image size, default 8 rows x 12 cols


def test_cell_size_for_rellis_images():
    assert GRID.cell_width == 160.0
    assert GRID.cell_height == 150.0


def test_labels_run_letters_down_and_numbers_across():
    assert GRID.label(0, 0) == "A1"
    assert GRID.label(5, 5) == "F6"
    assert GRID.label(7, 11) == "H12"
    assert GRID.all_labels()[:3] == ["A1", "A2", "A3"]
    assert len(GRID.all_labels()) == 96


def test_parse_is_inverse_of_label():
    for r in range(GRID.rows):
        for c in range(GRID.cols):
            assert GRID.parse(GRID.label(r, c)) == (r, c)


def test_parse_ignores_case_and_spaces():
    assert GRID.parse(" f6 ") == (5, 5)


@pytest.mark.parametrize("bad", ["", "F", "6", "6F", "F0", "F13", "I1", "AA1", "F-1", "F 6"])
def test_parse_rejects_bad_names(bad):
    with pytest.raises(ValueError):
        GRID.parse(bad)


def test_label_rejects_cells_outside_grid():
    with pytest.raises(ValueError):
        GRID.label(8, 0)


def test_bounds_and_center():
    assert GRID.bounds(0, 0) == (0.0, 0.0, 160.0, 150.0)
    assert GRID.bounds(7, 11) == (1760.0, 1050.0, 1920.0, 1200.0)
    assert GRID.center(2, 3) == (560.0, 375.0)


def test_cell_at_pixel():
    assert GRID.cell_at(0, 0) == (0, 0)
    assert GRID.cell_at(159.9, 149.9) == (0, 0)
    assert GRID.cell_at(160, 150) == (1, 1)  # an edge belongs to the next cell
    assert GRID.cell_at(1919.9, 1199.9) == (7, 11)
    assert GRID.cell_at(1920, 0) is None
    assert GRID.cell_at(0, -1) is None


def test_cell_at_center_round_trip():
    for r in range(GRID.rows):
        for c in range(GRID.cols):
            assert GRID.cell_at(*GRID.center(r, c)) == (r, c)


def test_cell_masks_tile_the_image():
    masks = [GRID.cell_mask(r, c) for r in range(GRID.rows) for c in range(GRID.cols)]
    coverage = np.sum(masks, axis=0)
    assert (coverage == 1).all()  # every pixel in exactly one cell
    assert masks[0][:150, :160].all()


def test_grid_with_uneven_cells():
    grid = ImageGrid(height=100, width=100, rows=3, cols=3)
    masks = [grid.cell_mask(r, c) for r in range(3) for c in range(3)]
    assert (np.sum(masks, axis=0) == 1).all()


def test_invalid_grid_size():
    with pytest.raises(ValueError):
        ImageGrid(height=100, width=100, rows=0)
    with pytest.raises(ValueError):
        ImageGrid(height=100, width=100, rows=27)


def test_draw_grid_returns_new_image_of_same_size():
    image = np.full((1200, 1920, 3), 100, dtype=np.uint8)
    drawn = draw_grid(image, GRID)
    assert drawn.shape == image.shape
    assert drawn.dtype == np.uint8
    assert (image == 100).all()  # input untouched
    assert not np.array_equal(drawn, image)
    assert (drawn[75, 80] == 100).all()  # middle of A1, away from lines and label, unchanged


def test_draw_grid_rejects_wrong_image_size():
    with pytest.raises(ValueError):
        draw_grid(np.zeros((100, 100, 3), dtype=np.uint8), GRID)
