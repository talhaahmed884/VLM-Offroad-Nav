"""The picture the VLM sees: candidate paths drawn on the front camera image (pure, no file I/O).

Following DriveVLN, each candidate is drawn as a colored line on the ground, with white dots
at the ticks (5 / 10 / 15 m) and its color name at its end. Settings from docs/approach-2-plan.md:
- the image is shrunk to IMAGE_WIDTH px wide, keeping the camera's shape (800 x 500);
- colors are assigned to the candidates at random (a seeded generator, so runs repeat), so the
  model can't learn "red is usually right" or "the left one is usually right";
- all candidates are drawn the same way (same width, same dots), and nothing from the planner
  (costs, ranks) is shown;
- where candidates share a stretch (they branch off one shortest-path tree), the lines are drawn
  side by side, so every color can be followed from the bottom of the picture to its end.

Only information that exists at run time is used: the image, the calibration, the candidates,
and the LiDAR ground height from the cost map (never the recorded poses or labels), so the same
drawing works in a closed-loop simulator.

The lines sit on a smoothed ground height: each path's ground height from the cost map is
noisy from cell to cell (and unknown in the blind zone), so a low-order curve is fitted along
the path. This removes the hooks and wiggles seen in the phase 2 check pictures.

The camera starts seeing the ground about 4.4 m ahead, so the first tick (5 m) sits at or just
below the bottom edge; ticks out of view are not drawn.
"""

import cv2
import numpy as np

from offroad.data.calibration import CameraCalibration
from offroad.geometry.projection import project_points_to_image, transform_points
from offroad.geometry.transforms import invert_transform
from offroad.mapping.elevation import T_FLU_FROM_LIDAR, GridSpec
from offroad.nav.candidates import CANDIDATE_COLORS, FULL_LENGTH_TOLERANCE_M, smooth_path
from offroad.nav.driven_path import point_at_distance
from offroad.nav.labels import TICKS_M, distances_along

IMAGE_WIDTH = 800
DEFAULT_GROUND_Z = -1.2  # the LiDAR rides ~1.2 m above the ground (phase 1 check)
GROUND_FIT_DEGREE = 2
GROUND_OUTLIER_M = 0.3  # cells this far from the first fit are left out of the second
MIN_GROUND_POINTS = 6
# Extra smoothing for drawing only (13 points = 3 m): the planner's 0.25 m grid zigzag shows up as
# ~20 px wiggles at 15 m. It moves the drawn line by a few cm, far below the 1 m match tolerance.
DISPLAY_SMOOTH_WINDOW = 13
OUTLINE_RGB = (0, 0, 0)
TICK_RGB = (255, 255, 255)
# Candidates branch off one shortest-path tree, so about half share their first 10 m with
# another one; drawn on top of each other, the lower line was hidden (phase 3 run, 2026-11-23).
# Where lines would overlap they are drawn side by side, like lines on a subway map (user's
# choice of the mock-ups, 2026-11-23). Smoothing the sideways shift over 9 points (2.25 m)
# avoids kinks where lines join or leave a shared stretch.
SPREAD_SMOOTH_POINTS = 9
# Version of the drawing, stored with each selector answer so answers to different pictures
# are never mixed: 1 = lines on top of each other (first phase 3 run), 2 = side by side.
PICTURE_VERSION = 2


def assign_colors(n: int, seed: int) -> list[tuple[str, tuple[int, int, int]]]:
    """n distinct (name, RGB) colors from CANDIDATE_COLORS in random order (repeatable for a seed)."""
    if n > len(CANDIDATE_COLORS):
        raise ValueError(f"only {len(CANDIDATE_COLORS)} colors for {n} candidates")
    order = np.random.default_rng(seed).permutation(len(CANDIDATE_COLORS))[:n]
    return [CANDIDATE_COLORS[k] for k in order]


def frame_seed(sequence: str, frame: int, salt: int = 0) -> int:
    """A fixed seed per frame (Python's hash() changes between runs, so it isn't used)."""
    return int(sequence) * 100_000 + frame + 1_000_003 * salt


def smooth_ground_heights(xy: np.ndarray, ground_z: np.ndarray, spec: GridSpec,
                          fallback_z: float = DEFAULT_GROUND_Z, degree: int = GROUND_FIT_DEGREE) -> np.ndarray:
    """(K,) ground height under each path point: a smooth curve in distance along the path, FLU meters.

    xy: (K, 2) path points; ground_z: (nx, ny) cost-map ground height (NaN = unknown).
    A polynomial of `degree` is fitted to the known heights (one round of outlier removal) and
    held constant beyond the first / last known point (e.g. in the blind zone). With too few
    known points, the median of the known ones (or fallback_z) is used everywhere.
    """
    nx, ny = spec.shape
    z = np.full(len(xy), np.nan)
    for k, (x, y) in enumerate(xy):
        i, j = spec.cell_index(x, y)
        if 0 <= i < nx and 0 <= j < ny:
            z[k] = ground_z[i, j]
    s = distances_along(xy)
    known = np.isfinite(z)
    if known.sum() < MIN_GROUND_POINTS or np.ptp(s[known]) < 1.0:
        level = float(np.median(z[known])) if known.any() else fallback_z
        return np.full(len(xy), level)
    coeffs = np.polyfit(s[known], z[known], degree)
    inliers = known & (np.abs(z - np.polyval(coeffs, s)) <= GROUND_OUTLIER_M)
    if inliers.sum() >= MIN_GROUND_POINTS:
        coeffs = np.polyfit(s[inliers], z[inliers], degree)
        known = inliers
    return np.polyval(coeffs, np.clip(s, s[known].min(), s[known].max()))


def project_flu(points_flu: np.ndarray, calib: CameraCalibration, image_size: tuple[int, int]) -> np.ndarray:
    """(K, 2) pixel positions (u, v) of FLU points in the full-size image; NaN where the camera can't see them."""
    points_lidar = transform_points(invert_transform(T_FLU_FROM_LIDAR), np.asarray(points_flu, dtype=float))
    uv_kept, _, kept = project_points_to_image(points_lidar, calib.T_cam_from_lidar, calib.K, image_size)
    uv = np.full((len(points_flu), 2), np.nan)
    uv[kept] = uv_kept
    return uv


def _visible_runs(uv: np.ndarray) -> list[np.ndarray]:
    """Consecutive stretches of visible (non-NaN) pixels, as int32 arrays for cv2.polylines."""
    visible = np.isfinite(uv).all(axis=1)
    runs, start = [], None
    for k, ok in enumerate(np.append(visible, False)):
        if ok and start is None:
            start = k
        elif not ok and start is not None:
            if k - start >= 2:
                runs.append(np.round(uv[start:k]).astype(np.int32))
            start = None
    return runs


def _tangents(uv: np.ndarray) -> np.ndarray:
    """(K, 2) direction of a pixel polyline at each point, per visible stretch; NaN where not visible."""
    t = np.full(uv.shape, np.nan)
    visible = np.isfinite(uv).all(axis=1)
    k = 0
    while k < len(uv):
        if not visible[k]:
            k += 1
            continue
        end = k
        while end < len(uv) and visible[end]:
            end += 1
        if end - k >= 2:
            t[k:end] = np.gradient(uv[k:end], axis=0)
        k = end
    return t


def spread_overlapping(uvs: list[np.ndarray], spacing_px: float, overlap_px: float,
                       smooth_points: int = SPREAD_SMOOTH_POINTS) -> list[np.ndarray]:
    """Shift lines sideways where they would be drawn on top of each other (pure).

    uvs: one (K_i, 2) pixel polyline per path (NaN = not visible). The paths start at the same
    point and are resampled at the same spacing, so index k is the same distance along each.
    Where m lines are within overlap_px of each other at index k, they are spread spacing_px apart,
    perpendicular to the line, in left-to-right order (by where each line ends in the image).
    Lines that overlap nothing keep their place. Returns the shifted polylines (same shapes).
    """
    ends = []
    for uv in uvs:
        visible = np.nonzero(np.isfinite(uv).all(axis=1))[0]
        ends.append(uv[visible[-1], 0] if len(visible) else np.inf)
    rank = {i: r for r, i in enumerate(np.argsort(ends, kind="stable"))}  # 0 = ends leftmost

    shifted = []
    for i, uv in enumerate(uvs):
        offset = np.zeros(len(uv))
        for k in range(len(uv)):
            if not np.isfinite(uv[k]).all():
                continue
            group = sorted((j for j, other in enumerate(uvs)
                            if k < len(other) and np.linalg.norm(other[k] - uv[k]) < overlap_px), key=rank.get)
            offset[k] = ((len(group) - 1) / 2 - group.index(i)) * spacing_px  # > 0: to the image's left
        if smooth_points > 1 and len(offset) > 1:
            pad = smooth_points // 2
            offset = np.convolve(np.pad(offset, pad, mode="edge"), np.ones(smooth_points) / smooth_points, "valid")
        t = _tangents(uv)
        # (dv, -du) points to the image's left for a line going up the image (away from the vehicle).
        normal = np.column_stack([t[:, 1], -t[:, 0]]) / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        shifted.append(uv + normal * offset[:, None])
    return shifted


def _put_label(image: np.ndarray, text: str, anchor: tuple[int, int], color: tuple[int, int, int],
               scale: float, taken: list[tuple[int, int, int, int]]) -> None:
    """Write `text` centered above `anchor`, moved up until it doesn't overlap an earlier label; in place."""
    font, thickness = cv2.FONT_HERSHEY_SIMPLEX, max(1, round(2 * scale))
    (w, h), base = cv2.getTextSize(text, font, scale, thickness)
    height, width = image.shape[:2]
    x = int(np.clip(anchor[0] - w // 2, 2, width - w - 2))
    y = int(np.clip(anchor[1] - 8, h + 2, height - base - 2))
    for _ in range(10):
        box = (x - 3, y - h - 3, x + w + 3, y + base + 3)
        if not any(box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3] for t in taken):
            break
        y = max(h + 2, y - h - 6)
    taken.append((x - 3, y - h - 3, x + w + 3, y + base + 3))
    cv2.putText(image, text, (x, y), font, scale, OUTLINE_RGB, thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_visual_prompt(
    image: np.ndarray,
    calib: CameraCalibration,
    paths_xy: list[np.ndarray],
    colors: list[tuple[str, tuple[int, int, int]]],
    ground_z: np.ndarray,
    spec: GridSpec,
    width: int = IMAGE_WIDTH,
) -> tuple[np.ndarray, list[bool]]:
    """The VLM's picture: the (H, W, 3) RGB camera image shrunk to `width` px, with the paths drawn on it.

    paths_xy: one (K, 2) FLU path per candidate; colors: (name, RGB) for each, same order.
    ground_z: the cost map's ground height (for placing the lines on the ground).
    Returns (picture, visible): visible[k] says whether path k shows up in the picture at all.
    """
    full_size = image.shape[:2]
    scale = width / full_size[1]
    picture = cv2.resize(image, (width, round(full_size[0] * scale)), interpolation=cv2.INTER_AREA)
    line = max(2, round(4 * width / IMAGE_WIDTH))
    dot = line + 2
    font_scale = 0.5 * width / IMAGE_WIDTH

    uvs, all_ticks = [], []
    for xy in paths_xy:
        xy = smooth_path(xy, DISPLAY_SMOOTH_WINDOW)
        z = smooth_ground_heights(xy, ground_z, spec)
        uv = project_flu(np.column_stack([xy, z]), calib, full_size) * scale
        s = distances_along(xy)
        ticks = []
        for d in TICKS_M:
            if d == TICKS_M[-1] and s[-1] >= d - FULL_LENGTH_TOLERANCE_M:
                d = s[-1]  # a full-length path often ends a few cm short of the last tick: mark its end
            p = point_at_distance(xy, d)
            if p is not None:
                tick_uv = project_flu(np.array([[p[0], p[1], np.interp(d, s, z)]]), calib, full_size)[0] * scale
                if np.isfinite(tick_uv).all():
                    ticks.append(tick_uv)
        uvs.append(uv)
        all_ticks.append(ticks)

    # Shared stretches side by side; the tick dots stay on the stretch's middle (one dot for all).
    uvs = spread_overlapping(uvs, spacing_px=line + 3, overlap_px=line + 4)
    drawn = []  # (runs, tick pixels, end pixel, name, rgb) per path
    for uv, ticks, (name, rgb) in zip(uvs, all_ticks, colors):
        visible = np.isfinite(uv).all(axis=1)
        end = uv[np.nonzero(visible)[0][-1]] if visible.any() else None
        drawn.append((_visible_runs(uv), ticks, end, name, rgb))

    for runs, *_ in drawn:  # dark outlines first, so every line stands out on grass, mud and sky
        cv2.polylines(picture, runs, False, OUTLINE_RGB, line + 3, cv2.LINE_AA)
    for runs, _, _, _, rgb in drawn:
        cv2.polylines(picture, runs, False, rgb, line, cv2.LINE_AA)
    for _, ticks, *_ in drawn:
        for u, v in ticks:
            cv2.circle(picture, (round(u), round(v)), dot + 1, OUTLINE_RGB, -1, cv2.LINE_AA)
            cv2.circle(picture, (round(u), round(v)), dot - 1, TICK_RGB, -1, cv2.LINE_AA)
    taken: list[tuple[int, int, int, int]] = []
    for _, _, end, name, rgb in drawn:
        if end is not None:
            _put_label(picture, name, (round(end[0]), round(end[1]) - dot), rgb, font_scale, taken)
    return picture, [end is not None for _, _, end, _, _ in drawn]
