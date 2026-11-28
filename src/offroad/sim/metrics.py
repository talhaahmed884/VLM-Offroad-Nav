"""Where the vehicle is relative to the trail: progress along it and distance from its centre (pure).

The trail (the scene's waypoints, in world xy and driving order) is the answer key: it scores a
run and gives Mode B its word, but no chooser ever sees it.
"""

import numpy as np


def cumulative_length(polyline: np.ndarray) -> np.ndarray:
    """(N,) distance along the polyline to each vertex, meters (0 at the first)."""
    p = np.asarray(polyline, dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])


def project_on_trail(trail: np.ndarray, xy: np.ndarray) -> tuple[float, float]:
    """(progress, offset) of a point: distance along the trail to its closest point, and the signed
    distance from the trail centre there (+ = left of the driving direction). Meters."""
    p = np.asarray(trail, dtype=float)
    q = np.asarray(xy, dtype=float)[:2]
    a, b = p[:-1], p[1:]
    ab = b - a
    seg_len2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    t = np.clip(((q - a) * ab).sum(axis=1) / seg_len2, 0.0, 1.0)
    closest = a + t[:, None] * ab
    d = np.linalg.norm(closest - q, axis=1)
    k = int(np.argmin(d))
    progress = float(cumulative_length(p)[k] + t[k] * np.sqrt(seg_len2[k]))
    cross = ab[k, 0] * (q[1] - a[k, 1]) - ab[k, 1] * (q[0] - a[k, 0])
    return progress, float(np.sign(cross) * d[k])


def footprint_gaps(points: np.ndarray, radii: np.ndarray, position_xy: np.ndarray, yaw: float,
                   length_m: float, width_m: float) -> np.ndarray:
    """(N,) gap in meters between each obstacle (a circle at a world-xy point) and the vehicle's
    rectangle (length_m x width_m, centred on position_xy, turned by yaw); < 0 = they touch."""
    if len(points) == 0:
        return np.zeros(0)
    c, s = np.cos(yaw), np.sin(yaw)
    rel = (np.asarray(points, dtype=float) - np.asarray(position_xy, dtype=float)[:2]) @ np.array([[c, -s], [s, c]])
    dx = np.maximum(np.abs(rel[:, 0]) - length_m / 2, 0.0)
    dy = np.maximum(np.abs(rel[:, 1]) - width_m / 2, 0.0)
    return np.hypot(dx, dy) - np.asarray(radii, dtype=float)


def is_stuck(times: list[float], progress: list[float], window_s: float, min_progress_m: float) -> bool:
    """True if the trail progress grew by less than min_progress_m over the last window_s seconds."""
    if not times or times[-1] - times[0] < window_s:
        return False
    t = np.asarray(times)
    start = int(np.searchsorted(t, t[-1] - window_s))
    return progress[-1] - progress[start] < min_progress_m
