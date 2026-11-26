"""Automatic labels for the candidates of one decision frame (pure, no file I/O).

Settings from docs/approach-2-plan.md (section 6):
- a candidate **matches the driver** if its mean distance to the driven path is at most
  MATCH_MAX_MEAN_M (several candidates may match). The distance is measured from the first tick
  (MATCH_FROM_M) on: every candidate starts at the vehicle and they often share the first
  meters, which the camera can't see anyway, so including them makes everything "match";
- its **safe distance** is how far along it the first unsafe stretch starts: water or a hard
  obstacle, from the two-source (LiDAR + camera) ground truth (which points are unsafe is
  decided by the caller, see `nav.frame_candidates`). Mud and grass count as safe, and points
  with no ground truth count as safe (unknown). Puddles are "caution": they don't count as
  unsafe, but where the first puddle stretch starts is recorded (first_caution_m). A single unsafe point is treated as label noise:
  the stretch must be at least MIN_UNSAFE_RUN points (0.5 m) long. A candidate that ends early
  (it couldn't go further) is only safe up to its end. The VLM answers in ticks, so the safe
  distance is rounded down to a tick at TICKS_M (5 / 10 / 15 m): "none", "tick 1", "tick 2", or "all";
- **stop** is right when every candidate is unsafe before the first tick (or there are none);
- a frame has a **real choice** if at least two candidates are safe to at least the first tick.
"""

from dataclasses import dataclass

import numpy as np

from offroad.mapping.elevation import GridSpec
from offroad.nav.candidates import FULL_LENGTH_TOLERANCE_M, Candidate, mean_distance_to

MATCH_MAX_MEAN_M = 1.0
TICKS_M = (5.0, 10.0, 15.0)
SAFE_ANSWERS = ("none", "tick 1", "tick 2", "all")
MIN_UNSAFE_RUN = 2  # consecutive unsafe points (0.25 m apart) needed to count
MATCH_FROM_M = TICKS_M[0]


@dataclass
class CandidateLabels:
    """Labels of one candidate.

    driven_mean_m: mean distance from the candidate (from MATCH_FROM_M on) to the driven path, meters.
    first_unsafe_m: distance along the candidate to its first unsafe stretch (None if none).
    first_unsafe_class: the class ID there (None if none).
    first_caution_m: distance along the candidate to its first caution stretch (a puddle; None if none).
    safe_m: how far it is safe: the first unsafe stretch, or its end if it ends early.
    safe_until: safe_m rounded down to a tick ("none", "tick 1", "tick 2", "all").
    unknown_share: share of its points with no ground truth.
    """

    driven_mean_m: float
    matches_driver: bool
    first_unsafe_m: float | None
    first_unsafe_class: int | None
    first_caution_m: float | None
    safe_m: float
    safe_until: str
    unknown_share: float


def cells_along(xy: np.ndarray, spec: GridSpec) -> tuple[list[tuple[int, int]], np.ndarray]:
    """The map cells a path passes through (consecutive repeats merged), and each point's index into that list.

    Points outside the map get index -1.
    """
    nx, ny = spec.shape
    cells: list[tuple[int, int]] = []
    index = np.full(len(xy), -1, dtype=int)
    for k, (x, y) in enumerate(xy):
        i, j = spec.cell_index(x, y)
        if not (0 <= i < nx and 0 <= j < ny):
            continue
        if not cells or cells[-1] != (i, j):
            cells.append((i, j))
        index[k] = len(cells) - 1
    return cells, index


def distances_along(xy: np.ndarray) -> np.ndarray:
    """(K,) distance from the first point along the path to each point, meters."""
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def first_unsafe_index(unsafe: np.ndarray, min_run: int = MIN_UNSAFE_RUN) -> int | None:
    """Index of the first point of the first run of at least min_run consecutive unsafe points; None if none.

    unsafe: (K,) bool per path point.
    """
    run = 0
    for k, bad in enumerate(unsafe):
        run = run + 1 if bad else 0
        if run == min_run:
            return k - min_run + 1
    return None


def safe_until_tick(safe_m: float, full_length: bool, ticks: tuple[float, ...] = TICKS_M) -> str:
    """The tick answer for a safe distance: "none" below the first tick, ..., "all" if safe to the end of a full path."""
    if full_length and safe_m >= ticks[-1] - FULL_LENGTH_TOLERANCE_M:
        return "all"
    passed = sum(safe_m >= t for t in ticks[:-1])
    return SAFE_ANSWERS[passed]


def label_candidate(candidate: Candidate, point_classes: np.ndarray, point_unsafe: np.ndarray,
                    driven_xy: np.ndarray, point_caution: np.ndarray | None = None) -> CandidateLabels:
    """Labels for one candidate (see the module docstring).

    point_classes: (K,) ground-truth class ID per candidate point (0 = unknown).
    point_unsafe: (K,) True where that point is unsafe.
    point_caution: (K,) True where that point needs caution (a puddle); None = nowhere.
    driven_xy: (M, 2) the driven path from the same frame, FLU meters.
    """
    along = distances_along(candidate.xy)
    far = candidate.xy[along >= MATCH_FROM_M]
    driven_mean = mean_distance_to(far if len(far) else candidate.xy, driven_xy)
    k = first_unsafe_index(point_unsafe)
    unsafe_m = None if k is None else float(along[k])
    safe = candidate.length_m if unsafe_m is None else min(unsafe_m, candidate.length_m)
    c = first_unsafe_index(point_caution) if point_caution is not None else None
    return CandidateLabels(
        driven_mean_m=driven_mean,
        matches_driver=driven_mean <= MATCH_MAX_MEAN_M,
        first_unsafe_m=unsafe_m,
        first_unsafe_class=None if k is None else int(point_classes[k]),
        first_caution_m=None if c is None else float(along[c]),
        safe_m=safe,
        safe_until=safe_until_tick(safe, candidate.full_length and unsafe_m is None),
        unknown_share=float((point_classes == 0).mean()) if len(point_classes) else 1.0,
    )


def should_stop(labels: list[CandidateLabels]) -> bool:
    """True if no candidate is safe to at least the first tick (or there are no candidates)."""
    return all(lab.safe_until == "none" for lab in labels)


def has_real_choice(labels: list[CandidateLabels]) -> bool:
    """True if at least two candidates are safe to at least the first tick."""
    return sum(lab.safe_until != "none" for lab in labels) >= 2


def unsafe_cells(classes: np.ndarray, cells: list[tuple[int, int]], standing: np.ndarray,
                 water_ids: list[int], hard_ids: list[int]) -> np.ndarray:
    """(C,) True for cells that are water, or a hard obstacle where something stands up.

    classes: (C,) ground-truth class per cell; standing: (nx, ny) bool, something taller than
    max_step in or next to the cell.
    """
    is_standing = np.array([standing[c] for c in cells], dtype=bool)
    return np.isin(classes, water_ids) | (np.isin(classes, hard_ids) & is_standing)
