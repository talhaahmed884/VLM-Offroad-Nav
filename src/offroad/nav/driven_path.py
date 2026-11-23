"""The path the vehicle actually drove, seen from one frame, and the labels made from it (pure).

The poses give the Ouster LiDAR's pose in a fixed world frame for every scan. Seen from frame
i, the LiDAR positions of the following frames trace the path the human driver took next. That
path, in frame i's forward-left-up (FLU) frame, is the reference for approach 2: which candidate
path matches the driver, and Mode B's coarse direction word ("bear left", ...).
"""

import numpy as np

from offroad.mapping.elevation import T_FLU_FROM_LIDAR

# Mode B direction words, from the bearing of the driven path about DIRECTION_DISTANCE_M ahead.
# Positive bearings are to the left (FLU +y).
DIRECTION_DISTANCE_M = 50.0
DIRECTION_MIN_DISTANCE_M = 20.0  # if the drive ends sooner, use its end point; shorter -> no word
STRAIGHT_MAX_DEG = 10.0
SLIGHT_MAX_DEG = 25.0
DIRECTION_WORDS = ("bear left", "slight left", "straight", "slight right", "bear right")


def poses_relative_to(poses: np.ndarray, i: int) -> np.ndarray:
    """(N, 4, 4) T_lidar_i_from_lidar_j: every scan's pose seen from scan i's LiDAR frame.

    poses: (N, 4, 4) T_world_from_lidar, one per scan.
    """
    return np.linalg.inv(poses[i]) @ poses


def driven_path_flu(poses: np.ndarray, i: int, max_distance: float) -> np.ndarray:
    """(M, 3) FLU positions (x forward, y left, z up; meters) of the LiDAR from frame i onward.

    The first row is frame i itself (the origin). The path stops at the first frame whose
    distance along the path reaches max_distance, or at the end of the sequence.
    """
    relative = poses_relative_to(poses[i:], 0)
    points_flu = (T_FLU_FROM_LIDAR[:3, :3] @ relative[:, :3, 3].T).T
    steps = np.linalg.norm(np.diff(points_flu[:, :2], axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(steps)])
    end = int(np.searchsorted(along, max_distance)) + 1  # include the first point at or past max_distance
    return points_flu[:end]


def path_length(path_xy: np.ndarray) -> float:
    """Length of a polyline (M, 2+) in the x-y plane, meters."""
    if len(path_xy) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path_xy[:, :2], axis=0), axis=1).sum())


def point_at_distance(path_xy: np.ndarray, distance: float) -> np.ndarray | None:
    """The (x, y) point `distance` meters along the path, or None if the path is shorter."""
    xy = np.asarray(path_xy, dtype=float)[:, :2]
    if len(xy) == 0:
        return None
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    if distance > along[-1] + 1e-6:  # tolerance: the path's length summed another way can differ by rounding
        return None
    return np.array([np.interp(distance, along, xy[:, 0]), np.interp(distance, along, xy[:, 1])])


def resample_path(path_xy: np.ndarray, spacing: float) -> np.ndarray:
    """(K, 2) points every `spacing` meters along the path, starting at its first point."""
    length = path_length(path_xy)
    distances = np.arange(0.0, length + 1e-9, spacing)
    return np.array([point_at_distance(path_xy, d) for d in distances]).reshape(-1, 2)


def direction_bearing_deg(
    path_xy: np.ndarray, distance: float = DIRECTION_DISTANCE_M, min_distance: float = DIRECTION_MIN_DISTANCE_M
) -> float | None:
    """Bearing (degrees, + = left) of the point `distance` m along the driven path.

    If the path is shorter, its end point is used as long as the path is at least
    `min_distance` long; otherwise None.
    """
    length = path_length(path_xy)
    if length < min_distance:
        return None
    x, y = point_at_distance(path_xy, min(distance, length))
    return float(np.degrees(np.arctan2(y, x)))


def direction_word(bearing_deg: float | None) -> str | None:
    """Mode B's coarse direction for a bearing: straight within ±10°, slight up to ±25°, else bear."""
    if bearing_deg is None:
        return None
    side = "left" if bearing_deg > 0 else "right"
    if abs(bearing_deg) < STRAIGHT_MAX_DEG:
        return "straight"
    return f"{'slight' if abs(bearing_deg) < SLIGHT_MAX_DEG else 'bear'} {side}"


def sample_by_time(timestamps: dict[int, float], rate_hz: float) -> list[int]:
    """Frame indices spaced at least 1 / rate_hz seconds apart, in time order (the first frame is kept)."""
    kept, last = [], -np.inf
    for index, t in sorted(timestamps.items(), key=lambda item: item[1]):
        if t - last >= 1.0 / rate_hz - 1e-6:
            kept.append(index)
            last = t
    return kept
