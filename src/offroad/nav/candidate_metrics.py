"""Terrain and clearance numbers along one path, for the phase 4 evaluation (pure, no file I/O).

For every candidate (and the driven path), phase 2 already knows the ground-truth class at each
path point (LiDAR labels, else the camera label image). From that, and the LiDAR, this module
computes what the evaluation reports for a chosen path:
- share of the path's points on water, puddle, mud, grass / bush (drivable vegetation), tree /
  log, and with no ground truth; points are 0.25 m apart, so shares are shares of length;
- clearance: the closest distance from the path to a LiDAR point labelled as a hard obstacle
  (tree, pole, fence, ...; overhead branches and the vehicle itself left out, as in approach 1);
  closer than NEAR_MISS_M counts as a near miss (the vehicle body would touch it);
- roughness: the mean ground slope (cm per m) and mean bump height along the path.

Like the labels, these come from the ground truth and are only used to score a choice, never
shown to the model.
"""

from dataclasses import dataclass

import numpy as np

from offroad.eval.metrics import COMPLIANT_VEGETATION, MUD_CLASSES, NON_COMPLIANT_VEGETATION, distance_to_obstacles, path_roughness
from offroad.eval.path_truth import UNSEEN
from offroad.mapping.costmap import CostParams
from offroad.mapping.elevation import GridSpec

# Half the vehicle's width, as in the cost map's inflation: a hard obstacle closer than this to
# the path's center line would touch the vehicle.
NEAR_MISS_M = CostParams.inflation_radius

CLASS_GROUPS = {
    "water": ("water",),
    "puddle": ("puddle",),
    "mud": MUD_CLASSES,
    "grass_bush": COMPLIANT_VEGETATION,
    "tree_log": NON_COMPLIANT_VEGETATION,
}


@dataclass
class CandidateMetrics:
    """Numbers for one path. Shares are of the path's points (0..1); None = could not be measured.

    clearance_m: closest hard-obstacle point to the path (None if the scan has none nearby).
    """

    share_water: float
    share_puddle: float
    share_mud: float
    share_grass_bush: float
    share_tree_log: float
    share_unknown: float
    clearance_m: float | None
    slope_cm_per_m: float | None
    bump_height_m: float | None

    @property
    def near_miss(self) -> bool:
        return self.clearance_m is not None and self.clearance_m < NEAR_MISS_M


def group_ids(id_to_name: dict[int, str]) -> dict[str, list[int]]:
    """{group name: ontology IDs} for CLASS_GROUPS."""
    return {group: [i for i, name in id_to_name.items() if name in names] for group, names in CLASS_GROUPS.items()}


def candidate_metrics(
    point_classes: np.ndarray,
    xy: np.ndarray,
    cells: list[tuple[int, int]],
    obstacle_xy: np.ndarray,
    ground_z: np.ndarray,
    obstacle_height: np.ndarray,
    spec: GridSpec,
    ids: dict[str, list[int]],
) -> CandidateMetrics:
    """The numbers for one path.

    point_classes: (K,) ground-truth class per path point (UNSEEN = none); xy: (K, 2) the path, FLU m;
    cells: the map cells it passes through (from `labels.cells_along`); obstacle_xy: (P, 2) hard
    obstacle points (from `eval.metrics.hard_obstacle_xy`); ground_z / obstacle_height: from the
    cost map; ids: from `group_ids`.
    """
    classes = np.asarray(point_classes)

    def share(group_ids: list[int]) -> float:
        return float(np.isin(classes, group_ids).mean()) if len(classes) else 0.0

    clearance = distance_to_obstacles(xy, obstacle_xy)
    slope, bumps = path_roughness(cells, ground_z, obstacle_height, spec)
    return CandidateMetrics(
        share_water=share(ids["water"]),
        share_puddle=share(ids["puddle"]),
        share_mud=share(ids["mud"]),
        share_grass_bush=share(ids["grass_bush"]),
        share_tree_log=share(ids["tree_log"]),
        share_unknown=share([UNSEEN]),
        clearance_m=None if clearance is None else clearance[1],
        slope_cm_per_m=slope,
        bump_height_m=bumps,
    )
