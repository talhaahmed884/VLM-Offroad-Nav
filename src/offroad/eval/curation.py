"""Helpers for choosing evaluation frames from RELLIS-3D label images (pure functions, no file I/O)."""

from dataclasses import dataclass

import numpy as np

# The "ahead" region: the part of the image the vehicle is about to drive into,
# given as fractions of image height (rows) and width (columns).
AHEAD_ROWS = (0.55, 1.0)
AHEAD_COLS = (0.25, 0.75)


@dataclass(frozen=True)
class Candidate:
    """One frame scored for one category."""

    sequence: str
    frame: int
    score: float


def ahead_region_mask(
    image_size: tuple[int, int],
    rows: tuple[float, float] = AHEAD_ROWS,
    cols: tuple[float, float] = AHEAD_COLS,
) -> np.ndarray:
    """Boolean mask of the "ahead" region.

    image_size: (height, width) in pixels.
    rows, cols: (start, end) as fractions of height and width, e.g. (0.55, 1.0).
    Returns (H, W) bool, True inside the region.
    """
    height, width = image_size
    mask = np.zeros((height, width), dtype=bool)
    r0, r1 = round(rows[0] * height), round(rows[1] * height)
    c0, c1 = round(cols[0] * width), round(cols[1] * width)
    mask[r0:r1, c0:c1] = True
    return mask


def class_pixel_fractions(label_image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Fraction of pixels belonging to each class ID.

    label_image: (H, W) uint8 class IDs.
    mask: optional (H, W) bool; only pixels where it is True are counted.
    Returns (256,) float; entry i is the fraction of counted pixels with class ID i.
    """
    pixels = label_image[mask] if mask is not None else label_image.ravel()
    counts = np.bincount(pixels, minlength=256)
    return counts / max(pixels.size, 1)


def pick_spaced(
    candidates: list[Candidate],
    n: int,
    min_gap: int,
    already_picked: list[Candidate] = (),
    max_per_sequence: int | None = None,
) -> list[Candidate]:
    """Pick up to n candidates, best score first, keeping picks apart in time.

    A candidate is skipped if it is within `min_gap` frames of an earlier pick in
    the same sequence, including picks in `already_picked` (e.g. other categories).
    This avoids near-duplicate frames, since consecutive frames are ~0.1 s apart.
    If `max_per_sequence` is set, at most that many of the new picks come from any
    one sequence, so a single location cannot dominate.
    """
    picked: list[Candidate] = []
    taken = list(already_picked)
    for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
        if len(picked) == n:
            break
        too_close = any(c.sequence == cand.sequence and abs(c.frame - cand.frame) < min_gap for c in taken)
        sequence_full = (
            max_per_sequence is not None
            and sum(c.sequence == cand.sequence for c in picked) >= max_per_sequence
        )
        if not too_close and not sequence_full:
            picked.append(cand)
            taken.append(cand)
    return picked
