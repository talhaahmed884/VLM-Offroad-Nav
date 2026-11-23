"""Choose the frames where the VLM is asked to decide, and label them (pure, no file I/O).

Settings from docs/approach-2-plan.md (section 6): about 2 decision frames per second (as
DriveVLN recorded), split by whole sequence so the test drive is never seen in training. Only
about every other frame has a camera label image, so each sampled frame is moved to a labelled
neighbour (0.1 s away) when it has none; the safe-distance labels need that camera ground truth.
Frames where the vehicle is standing still, or with less than MIN_PATH_AHEAD_M of driving left
in the recording, have no usable "where did the driver go" reference and are skipped.
"""

from dataclasses import dataclass

import numpy as np

from offroad.nav.driven_path import (
    DIRECTION_DISTANCE_M,
    direction_bearing_deg,
    direction_word,
    driven_path_flu,
    path_length,
    sample_by_time,
)

SPLITS = {"00000": "train", "00001": "train", "00002": "train", "00003": "val", "00004": "test"}
DECISION_RATE_HZ = 2.0
MIN_PATH_AHEAD_M = 20.0  # the candidates are ~15 m long; the driven path must be longer to compare
MIN_SPEED_MPS = 0.3  # slower than this over the next second counts as standing still
SPEED_WINDOW_S = 1.0


@dataclass
class DecisionFrame:
    """One frame where the VLM decides.

    path_ahead_m: how far the recorded drive continues from here (capped at DIRECTION_DISTANCE_M).
    speed_mps: average speed over the next SPEED_WINDOW_S seconds.
    bearing_deg / direction: Mode B's direction (+ = left), from the driven path ~50 m ahead.
    """

    sequence: str
    frame: int
    split: str
    timestamp: float
    path_ahead_m: float
    speed_mps: float
    bearing_deg: float | None
    direction: str | None
    labelled: bool = False  # the frame has a camera label image


def speed_ahead(positions: np.ndarray, times: np.ndarray, i: int, window: float = SPEED_WINDOW_S) -> float:
    """Average speed (m/s) from frame i to the first frame at least `window` seconds later; 0 at the end.

    positions: (N, 3) world positions; times: (N,) seconds, increasing.
    """
    j = int(np.searchsorted(times, times[i] + window))
    if j >= len(times):
        return 0.0
    distance = np.linalg.norm(np.diff(positions[i:j + 1, :2], axis=0), axis=1).sum()
    return float(distance / (times[j] - times[i]))


def prefer_labelled(frames: list[int], labelled: np.ndarray) -> list[int]:
    """Swap each frame without a camera label image for a labelled neighbour (next, else previous), if one exists.

    labelled: (N,) bool, True where the frame has a label image (about every other frame does).
    The neighbours are 0.1 s apart, so the timing barely changes, and the safe-distance labels get
    camera ground truth. Duplicates are dropped.
    """
    chosen = []
    for i in frames:
        if not labelled[i]:
            i = next((k for k in (i + 1, i - 1) if 0 <= k < len(labelled) and labelled[k]), i)
        if not chosen or chosen[-1] != i:
            chosen.append(i)
    return chosen


def build_decision_frames(
    sequence: str,
    poses: np.ndarray,
    timestamps: np.ndarray,
    rate_hz: float = DECISION_RATE_HZ,
    min_path_ahead: float = MIN_PATH_AHEAD_M,
    min_speed: float = MIN_SPEED_MPS,
    labelled: np.ndarray | None = None,
) -> tuple[list[DecisionFrame], dict[str, int]]:
    """Decision frames of one sequence, plus how many sampled frames were skipped and why.

    poses: (N, 4, 4) T_world_from_lidar; timestamps: (N,) seconds; both indexed by frame index.
    labelled: (N,) bool, frames with a camera label image; if given, sampled frames are moved to a
        labelled neighbour where possible (see `prefer_labelled`).
    Returns (frames, {"sampled": n, "stopped": n, "near_end": n}).
    """
    positions = poses[:, :3, 3]
    sampled = sample_by_time(dict(enumerate(timestamps)), rate_hz)
    if labelled is not None:
        sampled = prefer_labelled(sampled, labelled)
    counts = {"sampled": len(sampled), "stopped": 0, "near_end": 0}
    frames = []
    for i in sampled:
        path = driven_path_flu(poses, i, DIRECTION_DISTANCE_M)
        ahead = path_length(path)
        if ahead < min_path_ahead:
            counts["near_end"] += 1
            continue
        speed = speed_ahead(positions, timestamps, i)
        if speed < min_speed:
            counts["stopped"] += 1
            continue
        bearing = direction_bearing_deg(path)
        frames.append(DecisionFrame(sequence, i, SPLITS.get(sequence, "unused"), float(timestamps[i]),
                                    ahead, speed, bearing, direction_word(bearing),
                                    labelled=bool(labelled[i]) if labelled is not None else False))
    return frames, counts
