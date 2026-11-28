"""Phase 6, step 6.4: the test scenes, designed as trails and blockades in world xy (pure, no MAVS).

Each design is drawn into a generated MAVS forest (sim/mavs_world.build_designed_scene): its
trails become the dirt texture (MAVS's scene "Trail" path files), the plants on them are
removed, and its blockades are added as objects. Scenes (user decision 2026-11-26):
- curves: a left and a right bend (radius 40 m), ~150 m: does the chooser follow the bends?
- fork: a Y-junction 40 m in, branches curving to 25 degrees left and right; Mode B is run with
  the word for each branch (routes "left" and "right"): does the chooser follow the word?
- blocked: a 1.8 m wall across the trail and the grass beside it (11.5 m) at 40 m: the right
  behaviour is to stop before it;
- halfblocked: one wall over the left third of the trail: go round it and carry on.

A route is the polyline that scores a run and gives Mode B its word (the answer key); for the
blocked trail it runs through the wall, and stopping before the wall counts as right in 6.5.
Units are meters and radians; headings are counter-clockwise from +x (east).
"""

import math
from dataclasses import dataclass, field

import numpy as np

STEP_M = 0.4  # spacing of the trail points
BLOCKADE_MESH = "urban/roadblock/RoadBlockade_02.obj"  # 2.88 m long, 0.8 m tall, 0.49 m thick
BLOCKADE_LENGTH_M = 2.88
BLOCKADE_THICKNESS_M = 0.49
BLOCKADE_HEIGHT_SCALE = 2.25  # 0.8 m -> 1.8 m: the relaxed grid only blocks what is taller than 1.5 m


def straight(start, heading: float, length: float, step: float = STEP_M) -> np.ndarray:
    """(N, 2) points from start (excluded) along heading."""
    s = np.arange(1, int(round(length / step)) + 1) * step
    return np.asarray(start, dtype=float)[:2] + np.column_stack([np.cos(heading) * s, np.sin(heading) * s])


def arc(start, heading: float, radius: float, turn: float, step: float = STEP_M) -> np.ndarray:
    """(N, 2) points along a circular arc from start (excluded); turn > 0 = left, radians."""
    side = 1.0 if turn > 0 else -1.0
    centre = np.asarray(start, dtype=float)[:2] + radius * side * np.array([-np.sin(heading), np.cos(heading)])
    n = max(1, int(round(abs(turn) * radius / step)))
    angles = heading - side * np.pi / 2 + side * np.arange(1, n + 1) * abs(turn) / n
    return centre + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def end_heading(points: np.ndarray) -> float:
    d = points[-1] - points[-2]
    return float(math.atan2(d[1], d[0]))


def chain(start, heading: float, pieces: list[tuple]) -> np.ndarray:
    """A trail from start: pieces are ("straight", length) or ("arc", radius, turn)."""
    pts = [np.asarray(start, dtype=float)[None, :2]]
    h = heading
    for piece in pieces:
        if piece[0] == "straight":
            pts.append(straight(pts[-1][-1], h, piece[1]))
        else:
            pts.append(arc(pts[-1][-1], h, piece[1], piece[2]))
            h += piece[2]
    return np.vstack(pts)


@dataclass(frozen=True)
class Blockade:
    """A wall of BLOCKADE_LENGTH_M along `yaw` centred at (x, y)."""

    x: float
    y: float
    yaw: float

    def footprint_points(self, spacing: float = 0.2) -> np.ndarray:
        """(K, 2) points along its centre line (for the collision check, radius = half its thickness)."""
        s = np.arange(-BLOCKADE_LENGTH_M / 2, BLOCKADE_LENGTH_M / 2 + 1e-9, spacing)
        return np.column_stack([self.x + np.cos(self.yaw) * s, self.y + np.sin(self.yaw) * s])


@dataclass(frozen=True)
class SceneDesign:
    name: str
    trails: dict[str, np.ndarray]  # drawn as dirt: {name: (N, 2) points}
    routes: dict[str, np.ndarray]  # answer keys: {route name: (N, 2) points}
    blockades: tuple[Blockade, ...] = field(default_factory=tuple)
    note: str = ""

    @property
    def default_route(self) -> str:
        return next(iter(self.routes))


def across(trail: np.ndarray, distance: float, offset: float = 0.0) -> Blockade:
    """A blockade across the trail `distance` m along it, centred `offset` m to the left of the centre."""
    k = int(round(distance / STEP_M))
    d = trail[k + 1] - trail[k - 1]
    h = math.atan2(d[1], d[0])
    left = np.array([-np.sin(h), np.cos(h)])
    c = trail[k] + offset * left
    return Blockade(float(c[0]), float(c[1]), h + math.pi / 2)


# Within the RELLIS camera's limits (user decision 2026-11-26): its view is only 38 degrees wide
# (usable +-17 degrees), so a 15 m path must end within ~17 degrees of straight ahead. A 22 m
# bend radius (first design) ends the path along the trail at ~19.5 degrees and the vehicle ran
# out of visible candidates; 40 m ends it at ~11 degrees.
BEND_RADIUS_M = 40.0


def swerve(line: np.ndarray, start_m: float, length_m: float, offset_m: float, ramp_m: float = 12.0) -> np.ndarray:
    """The line shifted sideways (offset_m, + = left) over [start_m, start_m + length_m], with
    smooth (cosine) ramps of ramp_m before and after; elsewhere unchanged."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))])
    shift = np.zeros_like(s)
    up = (s > start_m - ramp_m) & (s < start_m)
    shift[up] = 0.5 * (1 - np.cos(np.pi * (s[up] - (start_m - ramp_m)) / ramp_m))
    shift[(s >= start_m) & (s <= start_m + length_m)] = 1.0
    down = (s > start_m + length_m) & (s < start_m + length_m + ramp_m)
    shift[down] = 0.5 * (1 + np.cos(np.pi * (s[down] - (start_m + length_m)) / ramp_m))
    tangent = np.gradient(line, axis=0)
    left = np.column_stack([-tangent[:, 1], tangent[:, 0]]) / np.linalg.norm(tangent, axis=1)[:, None]
    return line + (offset_m * shift)[:, None] * left


def designs() -> dict[str, SceneDesign]:
    r = BEND_RADIUS_M
    curves = chain((-68.0, -40.0), 0.0, [("straight", 15), ("arc", r, math.radians(50)), ("straight", 10),
                                         ("arc", r, math.radians(-50)), ("straight", 8), ("arc", r, math.radians(-30)),
                                         ("straight", 10)])  # 134 m
    # The branches leave the stem through a gentle arc before running straight at 25 degrees
    # (a sharp 35 degree kink put the path into either branch out of view at the junction).
    stem = chain((-60.0, 0.0), 0.0, [("straight", 40)])
    left = np.vstack([stem, chain(stem[-1], 0.0, [("arc", r, math.radians(25)), ("straight", 40)])[1:]])
    right = np.vstack([stem, chain(stem[-1], 0.0, [("arc", r, math.radians(-25)), ("straight", 40)])[1:]])
    line = chain((-60.0, 0.0), 0.0, [("straight", 120)])
    # Four walls end to end (11.5 m) close the 4 m trail and the grass beside it, so no 15 m path
    # can go round (grass is passable in the relaxed grid). The half-block covers the left third
    # of the trail (y 0.8 ... 3.6 m): passing needs a gentle 1 m swerve to the right.
    full = tuple(across(line, 40.0, offset=o) for o in (-4.32, -1.44, 1.44, 4.32))
    half = (across(line, 40.0, offset=2.2),)
    return {
        "curves": SceneDesign("curves", {"main": curves}, {"main": curves}, (),
                              f"left and right bends, radius {r:.0f} m"),
        "fork": SceneDesign("fork", {"left": left, "right": right}, {"left": left, "right": right}, (),
                            "Y-junction at 40 m, branches curving to 25 degrees left / right"),
        "blocked": SceneDesign("blocked", {"main": line}, {"main": line}, full,
                               "1.8 m wall across the trail and the grass beside it at 40 m"),
        # The route (answer key) swerves 1.2 m right around the wall, as a careful driver would;
        # along the trail centre it passed only 0.76 m from the wall's end and the oracle,
        # following it, grazed the wall (2026-11-26). The drawn trail stays straight.
        "halfblocked": SceneDesign("halfblocked", {"main": line}, {"main": swerve(line, 34.0, 12.0, -1.2)}, half,
                                   "1.8 m wall over the left third of the trail at 40 m"),
    }


VPRP_HEADER = ["QS VANE REPLAY FILE", "Version : 1", "Replay Type : PositionReplay",
               "Vehicle File : Generic4x4.vehicle.xml", "Vehicle Objects : 0", "#Start Date : 01 / 01 / 2000",
               "#Start Time : 08 : 00 : 00", "@Data"]


def vprp_text(points: np.ndarray, z: float = 0.0) -> str:
    """An ANVEL replay file (MAVS's trail format) for (N, 2) points."""
    lines = [f"{0.01 * (k + 1):.2f}:{x:.4f} {y:.4f} {z:.1f}:1 0 0 0" for k, (x, y) in enumerate(np.asarray(points))]
    return "\n".join(VPRP_HEADER + lines) + "\n"


def near_any(points: np.ndarray, polylines: list[np.ndarray], radius: float) -> np.ndarray:
    """(N,) True for points within radius of any polyline vertex (vertices are STEP_M apart)."""
    p = np.asarray(points, dtype=float)[:, :2]
    out = np.zeros(len(p), dtype=bool)
    for line in polylines:
        for chunk in range(0, len(line), 200):
            seg = line[chunk:chunk + 200]
            out |= (np.linalg.norm(p[:, None, :] - seg[None, :, :], axis=2) < radius).any(axis=1)
    return out
