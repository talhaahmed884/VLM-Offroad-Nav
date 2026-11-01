"""Draw a labelled `ImageGrid` on a camera image (pure function, no file I/O)."""

import cv2
import numpy as np

from offroad.vlm.grid import ImageGrid

LINE_RGB = (255, 255, 255)
TEXT_RGB = (255, 255, 255)
TEXT_BACKGROUND_RGB = (0, 0, 0)


def draw_grid(image: np.ndarray, grid: ImageGrid, line_alpha: float = 0.6) -> np.ndarray:
    """Copy of the image with grid lines and a label ("A1", ...) in each cell's top-left corner.

    image: (H, W, 3) uint8 RGB, same size as the grid.
    line_alpha: opacity of the grid lines (0 = invisible, 1 = solid), so terrain stays visible.
    Line width and text size scale with the cell size, so the grid stays readable when the
    image is shrunk (VLMs usually resize their input).
    """
    if image.shape[:2] != (grid.height, grid.width):
        raise ValueError(f"image is {image.shape[:2]}, grid expects {(grid.height, grid.width)}")

    scale = min(grid.cell_width, grid.cell_height) / 150  # 1.0 for 160 x 150 px cells
    thickness = max(1, round(2 * scale))

    lines = image.copy()
    for c in range(1, grid.cols):
        u = round(c * grid.cell_width)
        cv2.line(lines, (u, 0), (u, grid.height - 1), LINE_RGB, thickness)
    for r in range(1, grid.rows):
        v = round(r * grid.cell_height)
        cv2.line(lines, (0, v), (grid.width - 1, v), LINE_RGB, thickness)
    out = cv2.addWeighted(lines, line_alpha, image, 1 - line_alpha, 0)

    font, font_scale, text_thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, max(1, round(2 * scale))
    pad = max(2, round(4 * scale))
    for r in range(grid.rows):
        for c in range(grid.cols):
            text = grid.label(r, c)
            u0, v0, _, _ = grid.bounds(r, c)
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
            x, y = round(u0) + pad, round(v0) + pad
            cv2.rectangle(out, (x - pad, y - pad), (x + text_w + pad, y + text_h + baseline + pad),
                          TEXT_BACKGROUND_RGB, thickness=-1)
            cv2.putText(out, text, (x, y + text_h), font, font_scale, TEXT_RGB, text_thickness, cv2.LINE_AA)
    return out
