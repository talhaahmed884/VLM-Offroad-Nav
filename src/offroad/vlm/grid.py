"""A labelled grid over the camera image, so a VLM can answer with cell names like "F6".

Rows are letters from the top (A, B, ...), columns are numbers from the left (1, 2, ...),
so "A1" is the top-left cell. Pure functions, no file I/O.

Pixel convention (same as projection.py): u is the column, v is the row, in pixels.
"""

import re
from dataclasses import dataclass

import numpy as np

ROW_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LABEL = re.compile(r"^([A-Z])(\d{1,2})$")


@dataclass(frozen=True)
class ImageGrid:
    """A rows x cols grid covering an image of size (height, width) pixels."""

    height: int
    width: int
    rows: int = 8
    cols: int = 12

    def __post_init__(self):
        if not 1 <= self.rows <= len(ROW_LETTERS):
            raise ValueError(f"rows must be 1..{len(ROW_LETTERS)}, got {self.rows}")
        if not 1 <= self.cols <= 99:
            raise ValueError(f"cols must be 1..99, got {self.cols}")

    @property
    def cell_width(self) -> float:
        return self.width / self.cols

    @property
    def cell_height(self) -> float:
        return self.height / self.rows

    def label(self, row: int, col: int) -> str:
        """Cell name for 0-based (row, col), e.g. (0, 0) -> "A1", (5, 5) -> "F6"."""
        self._check(row, col)
        return f"{ROW_LETTERS[row]}{col + 1}"

    def parse(self, label: str) -> tuple[int, int]:
        """0-based (row, col) for a cell name like "F6" (case and surrounding spaces ignored).

        Raises ValueError if the name is malformed or outside this grid.
        """
        match = _LABEL.match(label.strip().upper())
        if not match:
            raise ValueError(f"not a cell name: {label!r} (expected e.g. 'F6')")
        row, col = ROW_LETTERS.index(match.group(1)), int(match.group(2)) - 1
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"cell {label!r} is outside the {self.rows} x {self.cols} grid")
        return row, col

    def all_labels(self) -> list[str]:
        """Every cell name, row by row: A1, A2, ..., then B1, ..."""
        return [self.label(r, c) for r in range(self.rows) for c in range(self.cols)]

    def bounds(self, row: int, col: int) -> tuple[float, float, float, float]:
        """Pixel edges (u0, v0, u1, v1) of a cell; the cell covers u0 <= u < u1, v0 <= v < v1."""
        self._check(row, col)
        return (
            col * self.cell_width,
            row * self.cell_height,
            (col + 1) * self.cell_width,
            (row + 1) * self.cell_height,
        )

    def center(self, row: int, col: int) -> tuple[float, float]:
        """Pixel (u, v) of a cell's center."""
        u0, v0, u1, v1 = self.bounds(row, col)
        return (u0 + u1) / 2, (v0 + v1) / 2

    def cell_at(self, u: float, v: float) -> tuple[int, int] | None:
        """0-based (row, col) of the cell containing pixel (u, v), or None if outside the image."""
        if not (0 <= u < self.width and 0 <= v < self.height):
            return None
        return int(v // self.cell_height), int(u // self.cell_width)

    def cell_mask(self, row: int, col: int) -> np.ndarray:
        """(height, width) bool mask, True inside the cell. Useful to read label images per cell."""
        u0, v0, u1, v1 = self.bounds(row, col)
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask[round(v0):round(v1), round(u0):round(u1)] = True
        return mask

    def _check(self, row: int, col: int):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"cell ({row}, {col}) is outside the {self.rows} x {self.cols} grid")
