"""Candidates and their labels for one decision frame, for each cost-grid variant (pure, no file I/O).

Everything shared by the variants is computed once: the cost map, the LiDAR label grid, the
driven path, and one ground-truth lookup (`eval.path_truth`: LiDAR labels first, then the
camera label image) for every cell that any candidate, or the driven path, passes through.

A cell is unsafe if its class is water, or a hard obstacle (tree, pole, fence, ...) that the
LiDAR also sees standing up: the cost map has something taller than max_step (0.3 m) in or next
to the cell. RELLIS labels sometimes mark flat ground as "person" or "tree" (e.g. the annotation
of people walking near the vehicle smears onto the ground); a hard obstacle lying flat on the
ground is not something the vehicle hits.

Puddles are **caution**, not unsafe (user decision 2026-11-22): the RELLIS driver drives through
them, so calling them unsafe made the labels contradict the driver. They don't shorten the safe
distance or cause "stop"; each candidate records where its first puddle is, so the evaluation
can report how often a chosen path goes through a puddle when a dry one existed.

The driven path is labelled the same way as a candidate. The driver really drove it, so if the
labels call it unsafe, that is a sign the labels (or the unsafe classes) are too strict.
"""

from dataclasses import dataclass, field

import numpy as np

from offroad.eval.metrics import HARD_OBSTACLE_CLASSES, hard_obstacle_xy, ids_of
from offroad.eval.path_truth import UNSEEN, path_truth
from offroad.mapping.costmap import CostMap, CostParams, geometric_cost_map, inflate
from offroad.mapping.elevation import GridSpec, build_elevation_map, lidar_to_flu, majority_label_grid
from offroad.nav.candidates import (
    CANDIDATE_LENGTH_M,
    RESAMPLE_SPACING_M,
    VARIANTS,
    Candidate,
    camera_half_fov_deg,
    generate_candidates,
    relaxed_cost,
    resample_and_cut,
    strict_cost,
)
from offroad.nav.candidate_metrics import CandidateMetrics, candidate_metrics, group_ids
from offroad.nav.driven_path import driven_path_flu
from offroad.nav.labels import CandidateLabels, cells_along, has_real_choice, label_candidate, should_stop

DRIVEN_MARGIN_M = 5.0  # compare candidates with the driven path up to CANDIDATE_LENGTH_M + this
IGNORE_CAMERA_CLASSES = ("void", "sky")  # camera labels that can't be the ground
STANDING_RADIUS_CELLS = 1  # a hard-obstacle label counts if something stands up within this many cells
UNSAFE_WATER_CLASSES = ("water",)  # deeper standing water
CAUTION_CLASSES = ("puddle",)  # driven through in RELLIS: recorded, but not unsafe


@dataclass
class FrameCandidates:
    """One variant's candidates for one frame, with their labels and terrain numbers (same order: left to right)."""

    variant: str
    candidates: list[Candidate]
    labels: list[CandidateLabels]
    metrics: list[CandidateMetrics] = field(default_factory=list)

    @property
    def stop(self) -> bool:
        return should_stop(self.labels)

    @property
    def real_choice(self) -> bool:
        return has_real_choice(self.labels)

    @property
    def best_driven_m(self) -> float | None:
        """Mean distance to the driven path of the candidate closest to it (None if no candidates)."""
        return min((lab.driven_mean_m for lab in self.labels), default=None)

    @property
    def closest_index(self) -> int | None:
        """Index of the candidate closest to the driven path: the driver's choice (None if no candidates)."""
        return int(np.argmin([lab.driven_mean_m for lab in self.labels])) if self.labels else None

    @property
    def any_matches_driver(self) -> bool:
        return any(lab.matches_driver for lab in self.labels)


@dataclass
class FrameResult:
    """Everything phase 2 computes for one decision frame.

    driven_xy: (M, 2) the driven path, FLU meters, up to CANDIDATE_LENGTH_M + DRIVEN_MARGIN_M.
    driven: the driven path's first CANDIDATE_LENGTH_M labelled like a candidate.
    variants: {"strict": ..., "relaxed": ...}.
    costmap: the geometric cost map of the frame (its ground_z places the paths on the camera image).
    driven_metrics: terrain and clearance numbers of the driven path's first CANDIDATE_LENGTH_M.
    """

    driven_xy: np.ndarray
    driven: CandidateLabels
    variants: dict[str, FrameCandidates]
    costmap: CostMap
    driven_metrics: CandidateMetrics | None = None


def unsafe_cells(classes: np.ndarray, cells: list[tuple[int, int]], standing: np.ndarray,
                 water_ids: list[int], hard_ids: list[int]) -> np.ndarray:
    """(C,) True for cells that are water, or a hard obstacle where something stands up.

    classes: (C,) ground-truth class per cell; standing: (nx, ny) bool, something taller than
    max_step in or next to the cell.
    """
    is_standing = np.array([standing[c] for c in cells], dtype=bool)
    return np.isin(classes, water_ids) | (np.isin(classes, hard_ids) & is_standing)


def candidates_for_frame(
    points_lidar: np.ndarray,
    point_labels: np.ndarray,
    label_image: np.ndarray | None,
    T_cam_from_lidar: np.ndarray,
    K: np.ndarray,
    poses: np.ndarray,
    frame_index: int,
    id_to_name: dict[int, str],
    image_width: int = 1920,
    variants: tuple[str, ...] = VARIANTS,
    spec: GridSpec = GridSpec(),
    params: CostParams = CostParams(),
) -> FrameResult:
    """Candidates and labels for each variant ("strict", "relaxed") of one frame, plus the driven path's labels.

    points_lidar / point_labels: the scan (N, 4) and its per-point classes; label_image: camera labels or None.
    poses: (N, 4, 4) T_world_from_lidar of the sequence; frame_index: this frame.
    image_width: camera image width in pixels (with K, it gives the camera's view for the candidates).
    """
    points_flu = lidar_to_flu(points_lidar)
    costmap = geometric_cost_map(points_flu, build_elevation_map(points_flu, spec), params)
    lidar_labels = majority_label_grid(points_flu, point_labels, spec, empty_label=UNSEEN)
    driven_xy = resample_and_cut(driven_path_flu(poses, frame_index, CANDIDATE_LENGTH_M + DRIVEN_MARGIN_M)[:, :2],
                                 RESAMPLE_SPACING_M, CANDIDATE_LENGTH_M + DRIVEN_MARGIN_M)
    driven = Candidate(resample_and_cut(driven_xy, RESAMPLE_SPACING_M, CANDIDATE_LENGTH_M), 0.0, float("nan"))

    half_fov = camera_half_fov_deg(K, image_width)
    grids = {"strict": lambda: strict_cost(costmap), "relaxed": lambda: relaxed_cost(costmap, params)}
    candidates = {variant: generate_candidates(grids[variant](), spec, half_fov_deg=half_fov) for variant in variants}

    # One ground-truth lookup for every cell any path passes through.
    paths = [driven] + [c for variant in variants for c in candidates[variant]]
    per_path = [cells_along(p.xy, spec) for p in paths]
    all_cells = list(dict.fromkeys(cell for cells, _ in per_path for cell in cells))
    cell_class, cell_unsafe, cell_caution = {}, {}, {}
    if all_cells:
        truth = path_truth(all_cells, lidar_labels, costmap.ground_z, spec, T_cam_from_lidar, K, label_image,
                           points_lidar, tuple(ids_of(id_to_name, IGNORE_CAMERA_CLASSES)) + (UNSEEN,))
        standing = inflate(costmap.obstacle, STANDING_RADIUS_CELLS)
        unsafe = unsafe_cells(truth.labels, all_cells, standing, ids_of(id_to_name, UNSAFE_WATER_CLASSES),
                              ids_of(id_to_name, HARD_OBSTACLE_CLASSES))
        caution = np.isin(truth.labels, ids_of(id_to_name, CAUTION_CLASSES))
        cell_class = dict(zip(all_cells, truth.labels))
        cell_unsafe = dict(zip(all_cells, unsafe))
        cell_caution = dict(zip(all_cells, caution))

    # For the clearance numbers: hard-obstacle points that stand up more than max_step, the same
    # rule as for "unsafe" (RELLIS labels flat ground near people as "person").
    obstacle_xy = hard_obstacle_xy(points_lidar, point_labels, ids_of(id_to_name, HARD_OBSTACLE_CLASSES),
                                   costmap.ground_z, spec, params.vehicle_height, min_height=params.max_step)
    ids = group_ids(id_to_name)
    labels, metrics = [], []
    for path, (cells, index) in zip(paths, per_path):
        classes = np.array([cell_class[cells[k]] if k >= 0 else UNSEEN for k in index], dtype=int)
        unsafe = np.array([bool(cell_unsafe[cells[k]]) if k >= 0 else False for k in index], dtype=bool)
        caution = np.array([bool(cell_caution[cells[k]]) if k >= 0 else False for k in index], dtype=bool)
        labels.append(label_candidate(path, classes, unsafe, driven_xy, caution))
        metrics.append(candidate_metrics(classes, path.xy, cells, obstacle_xy, costmap.ground_z,
                                         costmap.obstacle_height, spec, ids))

    results, k = {}, 1
    for variant in variants:
        n = len(candidates[variant])
        results[variant] = FrameCandidates(variant, candidates[variant], labels[k:k + n], metrics[k:k + n])
        k += n
    return FrameResult(driven_xy=driven_xy, driven=labels[0], variants=results, costmap=costmap,
                       driven_metrics=metrics[0])
