"""Re-weight the geometric cost map with the VLM's cost hints (pure functions, no file I/O).

Each hint names an image grid cell; `GridLookup` says which map cells it covers. Then:
- Every hinted map cell's cost is multiplied by its level's multiplier (preferred makes it
  cheaper, avoid makes it dearer), and "impassable" cells are blocked.
- The VLM may unblock a LiDAR obstacle only if it says the cell is grass or bush at a
  drivable level and the obstacle is low (<= max_unblock_height). That is the tall-grass
  case the project is about; such cells become rough-but-drivable, never free. Hints about
  trees, rocks, people, etc. can never unblock anything.
- Hints only apply to map cells within max_hint_distance of the vehicle: near the horizon one
  grid cell covers a strip of ground 20+ m deep, so far hints are too coarse to trust.
- The VLM's waypoints are traced onto the ground, and cells within a narrow corridor along
  that route get a small discount (see `waypoints.py`), as soft guidance only.
Obstacles are re-inflated after unblocking, so the safety margin follows the obstacles that remain.

User decisions (2026-11-01): these defaults, as proposed.
"""

from dataclasses import dataclass, field

import numpy as np

from offroad.fusion.grid_lookup import GridLookup
from offroad.fusion.waypoints import route_corridor, waypoint_ground_points
from offroad.mapping.costmap import LETHAL, CostMap, CostParams, drivable_cost, inflate
from offroad.vlm.answer import TERRAIN_TYPES, TRAVERSABILITY_LEVELS

NO_HINT = -1


@dataclass(frozen=True)
class FusionParams:
    """How VLM hints change the cost map.

    multipliers: cost factor per traversability level ("impassable" blocks instead).
    unblock_terrains / unblock_levels: which hints may turn a LiDAR obstacle into drivable ground.
    max_unblock_height: only obstacles at most this tall (meters above the ground) can be unblocked.
    max_hint_distance: hints only apply to map cells at most this far from the vehicle, meters.
    corridor_half_width: map cells this close to the VLM's route (meters) count as "on the route".
    corridor_factor: cost factor on the route; < 1 nudges the planner toward it (never unblocks).
    """

    multipliers: dict[str, float] = field(default_factory=lambda: {
        "preferred": 0.7, "ok": 1.0, "caution": 2.0, "avoid": 5.0,
    })
    unblock_terrains: tuple[str, ...] = ("short_grass", "tall_grass_or_bush")
    unblock_levels: tuple[str, ...] = ("preferred", "ok", "caution")
    max_unblock_height: float = 1.5
    max_hint_distance: float = 25.0
    corridor_half_width: float = 1.0
    corridor_factor: float = 0.8


@dataclass
class HintMap:
    """The VLM hints painted onto the map. All arrays are (nx, ny).

    level: index into TRAVERSABILITY_LEVELS, NO_HINT where there is no hint.
    terrain: index into TERRAIN_TYPES, NO_HINT where there is no hint.
    """

    level: np.ndarray
    terrain: np.ndarray

    @property
    def hinted(self) -> np.ndarray:
        return self.level != NO_HINT

    def has_level(self, levels: tuple[str, ...]) -> np.ndarray:
        return np.isin(self.level, [TRAVERSABILITY_LEVELS.index(lv) for lv in levels])

    def has_terrain(self, terrains: tuple[str, ...]) -> np.ndarray:
        return np.isin(self.terrain, [TERRAIN_TYPES.index(t) for t in terrains])


@dataclass
class FusedCostMap:
    """Result of `reweight_cost_map`. All arrays are (nx, ny).

    cost: the re-weighted cost; LETHAL where blocked.
    hints: the hints that were applied (after the distance limit).
    unblocked: LiDAR obstacle cells the VLM turned into drivable ground.
    impassable: cells the VLM blocked.
    route_xy: the VLM's waypoints on the ground, FLU (x, y) meters, near to far.
    corridor: cells on the VLM's route (their cost was multiplied by corridor_factor).
    """

    cost: np.ndarray
    hints: HintMap
    unblocked: np.ndarray
    impassable: np.ndarray
    route_xy: list[tuple[float, float]]
    corridor: np.ndarray


def paint_hints(lookup: GridLookup, hints: list[dict], max_distance: float) -> HintMap:
    """Put each hint ({"cell", "terrain", "traversability"}, as saved by the runner) on the map cells it covers.

    Map cells farther than `max_distance` meters from the vehicle are left without a hint.
    """
    spec = lookup.spec
    nx, ny = spec.shape
    xs = spec.x_min + (np.arange(nx) + 0.5) * spec.resolution
    ys = spec.y_min + (np.arange(ny) + 0.5) * spec.resolution
    near = np.hypot(xs[:, None], ys[None, :]) <= max_distance

    level = np.full((nx, ny), NO_HINT, dtype=int)
    terrain = np.full((nx, ny), NO_HINT, dtype=int)
    for hint in hints:
        cells = lookup.mask(hint["cell"]) & near
        level[cells] = TRAVERSABILITY_LEVELS.index(hint["traversability"])
        terrain[cells] = TERRAIN_TYPES.index(hint["terrain"])
    return HintMap(level=level, terrain=terrain)


def reweight_cost_map(
    costmap: CostMap,
    lookup: GridLookup,
    hints: list[dict],
    params: FusionParams = FusionParams(),
    cost_params: CostParams = CostParams(),
    waypoints: list[str] = (),
) -> FusedCostMap:
    """The geometric cost map changed by the VLM's hints and route (see the module docstring).

    hints: as saved by the runner ({"cell", "terrain", "traversability"}); waypoints: cell names, near to far.
    With no hints and no waypoints, the result equals costmap.cost exactly.
    """
    hint_map = paint_hints(lookup, hints, params.max_hint_distance)
    vegetation = hint_map.has_terrain(params.unblock_terrains) & hint_map.has_level(params.unblock_levels)
    low = np.nan_to_num(costmap.obstacle_height, nan=np.inf) <= params.max_unblock_height
    unblocked = costmap.obstacle & vegetation & low
    impassable = hint_map.has_level(("impassable",))

    cost = drivable_cost(costmap.obstacle_height, cost_params)
    for level, factor in params.multipliers.items():
        cost[hint_map.has_level((level,))] *= factor

    route_xy = waypoint_ground_points(lookup, list(waypoints))
    corridor = route_corridor(costmap.spec, route_xy, params.corridor_half_width, params.max_hint_distance)
    cost[corridor] *= params.corridor_factor

    obstacle = (costmap.obstacle & ~unblocked) | impassable
    blocked = inflate(obstacle, round(cost_params.inflation_radius / costmap.spec.resolution))
    cost[blocked] = LETHAL
    return FusedCostMap(cost=cost, hints=hint_map, unblocked=unblocked, impassable=impassable,
                        route_xy=route_xy, corridor=corridor)
