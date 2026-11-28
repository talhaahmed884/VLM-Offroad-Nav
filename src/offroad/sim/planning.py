"""One decision's geometry in the simulator: LiDAR scan -> cost map -> candidate paths (pure, no MAVS).

The same phase 2 pipeline as on RELLIS (`mapping` + `nav.candidates`, relaxed grid), fed with a
MAVS scan already turned into the RELLIS Ouster's axes (`sim.rig.to_lidar_axes`). Nothing here
uses labels: candidates come from LiDAR geometry only, as they will at run time.
"""

from dataclasses import dataclass, replace

import numpy as np

from offroad.data.calibration import CameraCalibration
from offroad.mapping.costmap import CostMap, CostParams, geometric_cost_map, obstacle_height_map
from offroad.mapping.elevation import GridSpec, build_elevation_map, lidar_to_flu
from offroad.nav.candidates import (CANDIDATE_LENGTH_M, FOOTPRINT_RADIUS_M, IN_VIEW_MARGIN_DEG, RESAMPLE_SPACING_M,
                                    Candidate,
                                    camera_half_fov_deg, generate_candidates, relaxed_cost, resample_and_cut)
from offroad.nav.driven_path import direction_bearing_deg, direction_word


@dataclass
class FramePlan:
    """costmap: the geometric cost map; cost: the relaxed cost grid the candidates were planned on;
    candidates: up to 4 paths in the FLU frame at the LiDAR, left to right;
    hidden_only: no candidate, but there were paths that leave the camera's view (the VLM
    couldn't judge them, so they are dropped as on RELLIS)."""

    costmap: CostMap
    cost: np.ndarray
    candidates: list[Candidate]
    hidden_only: bool = False


OVERHANG_LOW_M = 1.0  # something standing on the ground reaches into the band max_step ... this height


def without_overhangs(costmap: CostMap, points_flu: np.ndarray, params: CostParams,
                      low_m: float = OVERHANG_LOW_M) -> CostMap:
    """The cost map with "floating" obstacles removed: cells whose only tall points hang in the air.

    An obstacle cell counts only if something stands up from the ground in it (a point between
    max_step and low_m above the ground: a trunk, a bush, a rock, a wall). A cell whose points are
    all near the ground or above low_m (tree foliage overhanging the trail) is drivable, with the
    height of its near-ground points. In MAVS the canopy edge hangs 1.5-1.8 m over the trails,
    just under the 1.8 m vehicle height, and blocked most of them (2026-11-26); on RELLIS the
    driver also drove under branches 1.65-1.8 m up (phase 4).
    """
    low = obstacle_height_map(points_flu, costmap.ground_z, costmap.spec, low_m)
    overhang = costmap.obstacle & ~(np.nan_to_num(low, nan=0.0) > params.max_step)
    return replace(costmap, obstacle_height=np.where(overhang, low, costmap.obstacle_height),
                   obstacle=costmap.obstacle & ~overhang)


# Obstacles are grown by half the vehicle's width plus 0.2 m. The RELLIS value (0.75 m) is less
# than half the MRZR4's width (1.6 m), so the oracle's path along the half-wall scraped it
# (gap -0.02 m, 2026-11-26).
SIM_COST_PARAMS = CostParams(inflation_radius=1.0)


def plan_frame(points_lidar: np.ndarray, calib: CameraCalibration, image_width: int,
               spec: GridSpec = GridSpec(), params: CostParams = SIM_COST_PARAMS,
               ignore_overhangs: bool = True, straighten: bool = True) -> FramePlan:
    """Candidates for one scan. points_lidar: (N, 3+) in the RELLIS Ouster axes; image_width in pixels."""
    points_flu = lidar_to_flu(np.asarray(points_lidar, dtype=float))
    costmap = geometric_cost_map(points_flu, build_elevation_map(points_flu, spec), params)
    if ignore_overhangs:
        costmap = without_overhangs(costmap, points_flu, params)
    cost = relaxed_cost(costmap, params)
    half_fov = camera_half_fov_deg(calib.K, image_width)
    # The phase 2 view rule checks a path's points from 5 m on, so a short path (3-5 m) to the
    # side passed it with no point checked (a 3 m path at 70 degrees, 2026-11-26). In the
    # simulator a candidate's end must be in view too, so the VLM can see where it leads.
    candidates = [c for c in generate_candidates(cost, spec, half_fov_deg=half_fov)
                  if abs(c.end_bearing_deg) <= min(half_fov) - IN_VIEW_MARGIN_DEG]
    hidden_only = not candidates and bool(generate_candidates(cost, spec))
    shaped = []
    for c in candidates:
        if straighten:
            # A straightened path starts at the vehicle (0, 0) and heads for its goal, so the
            # half-cell offset of the grid path no longer pulls the follower aside (no shift needed).
            shaped.append(Candidate(xy=pull_string(c.xy, cost, spec), target_bearing_deg=c.target_bearing_deg,
                                    cost_per_m=c.cost_per_m))
        else:
            shaped.append(centred_on_vehicle(c, spec))
    return FramePlan(costmap=costmap, cost=cost, candidates=shaped, hidden_only=hidden_only)


def segment_is_clear(a: np.ndarray, b: np.ndarray, cost: np.ndarray, spec: GridSpec, limit: float,
                     free_radius: float = FOOTPRINT_RADIUS_M) -> bool:
    """True if every cell on the straight segment a-b (FLU, m) is finite and costs at most `limit`.

    Cells within free_radius of the vehicle count as free (its own footprint, cleared as in
    generate_candidates).
    """
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / (spec.resolution / 2))) + 1)
    for p in np.linspace(a, b, n):
        if np.hypot(p[0], p[1]) <= free_radius:
            continue
        i, j = spec.cell_index(float(p[0]), float(p[1]))
        if not (0 <= i < spec.shape[0] and 0 <= j < spec.shape[1]) or not cost[i, j] <= limit:
            return False
    return True


def pull_string(xy: np.ndarray, cost: np.ndarray, spec: GridSpec) -> np.ndarray:
    """The path made as straight as the ground allows ("string pulling"), resampled every 0.25 m.

    From each point it jumps to the farthest later point that a straight segment reaches without
    crossing a cell costlier than the most expensive cell the original path crosses on that
    stretch. On a grid, shortest paths run straight as long as they can and turn late; in closed
    loop the vehicle then never turned (each new plan started straight again, 2026-11-26). The
    pulled path turns toward its goal from the start, and it is also what the picture shows.
    """
    pts = np.asarray(xy, dtype=float)
    if len(pts) < 3:
        return pts
    cells = [spec.cell_index(float(x), float(y)) for x, y in pts]
    point_cost = np.array([cost[c] if 0 <= c[0] < spec.shape[0] and 0 <= c[1] < spec.shape[1] else np.inf
                           for c in cells])
    point_cost[np.hypot(pts[:, 0], pts[:, 1]) <= FOOTPRINT_RADIUS_M] = 0.0
    keep, i = [0], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not segment_is_clear(pts[i], pts[j], cost, spec, float(np.max(point_cost[i:j + 1]))):
            j -= 1
        keep.append(j)
        i = j
    return resample_and_cut(pts[keep], RESAMPLE_SPACING_M, CANDIDATE_LENGTH_M)


def centred_on_vehicle(candidate: Candidate, spec: GridSpec) -> Candidate:
    """The candidate shifted so that its grid part starts at the vehicle, not at its cell's centre.

    Candidates run through cell centres; the vehicle's own cell has its centre (0.125, 0.125) m
    away, so every path was 12.5 cm to the left of the vehicle. Offline that didn't matter (labels
    compare paths at meter scale), but in closed loop the follower steered toward it at every
    decision and the vehicle slowly turned left (2026-11-26). The first point stays at (0, 0).
    """
    origin = np.array(spec.cell_center(*spec.cell_index(0.0, 0.0)))
    xy = np.asarray(candidate.xy, dtype=float).copy()
    xy[1:] -= origin
    return Candidate(xy=xy, target_bearing_deg=candidate.target_bearing_deg, cost_per_m=candidate.cost_per_m)


def trail_ahead_flu(trail_xy: np.ndarray, position_xy: np.ndarray, yaw: float) -> np.ndarray:
    """The trail from the vehicle onward, in the vehicle's FLU frame, starting at the vehicle (0, 0).

    trail_xy: (N, 2) world waypoints in driving order. The trail continues from the waypoint
    nearest to the vehicle. Used for Mode B's direction word (as the driven path on RELLIS) and
    for scoring; never shown to the choosers.
    """
    trail = np.asarray(trail_xy, dtype=float)
    if len(trail) == 0:
        return np.zeros((1, 2))
    start = int(np.argmin(np.hypot(*(trail - np.asarray(position_xy, dtype=float)[:2]).T)))
    c, s = np.cos(yaw), np.sin(yaw)
    ahead = (trail[start:] - np.asarray(position_xy, dtype=float)[:2]) @ np.array([[c, -s], [s, c]])
    return np.vstack([[0.0, 0.0], ahead])


def trail_direction_word(trail_xy: np.ndarray, position_xy: np.ndarray, yaw: float) -> str | None:
    """Mode B's word from the trail ~50 m ahead (as from the driven path on RELLIS); None near the end."""
    return direction_word(direction_bearing_deg(trail_ahead_flu(trail_xy, position_xy, yaw)))


def straightest_index(candidates: list[Candidate]) -> int | None:
    """The candidate whose end bearing is closest to straight ahead (the phase 4 baseline)."""
    if not candidates:
        return None
    return int(np.argmin([abs(c.end_bearing_deg) for c in candidates]))


def cheapest_index(candidates: list[Candidate]) -> int | None:
    """The candidate with the lowest mean cost per meter (the LiDAR planner's own pick)."""
    if not candidates:
        return None
    return int(np.argmin([c.cost_per_m for c in candidates]))
