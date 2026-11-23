"""Small pure helpers used when running a VLM on frames: image size and reply cleanup.

The model itself lives in `backend.py`; this module needs no GPU and is unit-tested.
"""

import cv2
import numpy as np


def downscale(image: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink an (H, W, 3) image so its longer side is at most `max_side` px (keeps the aspect ratio).

    Fewer pixels mean fewer image tokens, so less GPU memory and a faster answer. Images that
    are already small enough are returned unchanged.
    """
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1.0:
        return image
    size = (max(round(width * scale), 1), max(round(height * scale), 1))  # cv2 wants (width, height)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def strip_thinking(text: str) -> str:
    """Drop a "<think> ... </think>" block that reasoning models put before their answer."""
    marker = "</think>"
    return text.split(marker, 1)[1].lstrip() if marker in text else text
