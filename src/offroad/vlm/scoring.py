"""Score a VLM answer against the camera label image of the same frame (pure, no file I/O).

For each grid cell we measure what share of its pixels belong to each class group (water,
mud, hard obstacles, ...). A hint or waypoint is then judged by the cell it names. All the
thresholds below are judgment calls; they are named constants so the report can state them.

RELLIS-3D labels plain trail ruts as "mud", so mud is not counted as a hazard for routes,
and "dirt_or_gravel" hints may match mud cells.
"""

from dataclasses import dataclass

import numpy as np

from offroad.vlm.answer import TERRAIN_TYPES
from offroad.vlm.grid import ImageGrid

# Class groups by ontology name (IDs are looked up from the ontology, not hardcoded).
CLASS_GROUPS = {
    "water": ("water", "puddle"),
    "mud": ("mud",),
    "bush": ("bush",),
    "good": ("dirt", "grass", "asphalt", "concrete"),
    "hard": ("tree", "pole", "vehicle", "object", "building", "log", "person", "fence", "barrier"),
    "person": ("person",),
}

# Ontology classes that count as a match for each terrain word the VLM can use.
TERRAIN_CLASSES = {
    "dirt_or_gravel": ("dirt", "asphalt", "concrete", "mud"),
    "short_grass": ("grass",),
    "tall_grass_or_bush": ("grass", "bush"),
    "mud": ("mud",),
    "water": ("water", "puddle"),
    "rocks": ("rubble",),
    "tree_or_log": ("tree", "log"),
    "other_obstacle": ("pole", "object", "building", "fence", "barrier"),
    "person_or_vehicle": ("person", "vehicle"),
}
assert set(TERRAIN_CLASSES) == set(TERRAIN_TYPES)

TERRAIN_MATCH_MIN = 0.2  # a hint's terrain is right if its classes cover >= 20% of the cell
WATER_HINT_MIN = 0.1  # a "water" hint is right if the cell is >= 10% water/puddle
AVOID_HAZARD_MIN = 0.3  # an avoid/impassable hint is justified if water+mud+hard+bush >= 30%
PREFERRED_GOOD_MIN = 0.5  # a preferred hint is right if >= 50% good ground ...
PREFERRED_BAD_MAX = 0.1  # ... and < 10% water and < 10% hard obstacle
WATER_CELL_MIN = 0.3  # a cell "is water" for the recall check if >= 30% water/puddle
ROUTE_HAZARD_MIN = 0.2  # a route cell is hazardous if >= 20% water or >= 20% hard obstacle
PERSON_PRESENT_MIN = 0.02  # a frame "has a person" if some cell is >= 2% person

CAUTIOUS_LEVELS = ("caution", "avoid", "impassable")


def class_ids(id_to_name: dict[int, str], names: tuple[str, ...]) -> list[int]:
    """Ontology IDs of the classes with these names (names missing from the ontology are skipped)."""
    return [i for i, name in id_to_name.items() if name in names]


def cell_fractions(
    label_image: np.ndarray, grid: ImageGrid, id_to_name: dict[int, str], groups: dict[str, tuple[str, ...]]
) -> dict[str, dict[str, float]]:
    """Share of each cell's pixels in each group: {"F6": {"water": 0.4, ...}, ...}.

    label_image: (H, W) class IDs, same size as the grid.
    groups: group name -> ontology class names, e.g. CLASS_GROUPS or TERRAIN_CLASSES.
    """
    if label_image.shape[:2] != (grid.height, grid.width):
        raise ValueError(f"label image is {label_image.shape[:2]}, grid expects {(grid.height, grid.width)}")
    group_ids = {group: class_ids(id_to_name, names) for group, names in groups.items()}
    fractions = {}
    for r in range(grid.rows):
        for c in range(grid.cols):
            u0, v0, u1, v1 = grid.bounds(r, c)
            patch = label_image[round(v0):round(v1), round(u0):round(u1)]
            fractions[grid.label(r, c)] = {
                group: float(np.isin(patch, ids).mean()) for group, ids in group_ids.items()
            }
    return fractions


def straight_route(grid: ImageGrid, goal_cell: str) -> list[str]:
    """Naive comparison route: from the bottom row straight up the goal's column to the goal's row."""
    goal_row, goal_col = grid.parse(goal_cell)
    return [grid.label(r, goal_col) for r in range(grid.rows - 1, goal_row - 1, -1)]


def mean_share(cells: list[str], fractions: dict[str, dict[str, float]], group: str) -> float | None:
    """Mean share of `group` over the cells, skipping the first (the cell under the vehicle)."""
    rest = cells[1:]
    return float(np.mean([fractions[c][group] for c in rest])) if rest else None


@dataclass
class FrameScore:
    """Counts for one answer; sum them over frames, then divide (e.g. water_hints_correct / water_hints).

    Hint counts are per cell. route_* and straight_* are shares (None if not defined).
    """

    hints: int = 0
    hints_terrain_matched: int = 0
    water_hints: int = 0
    water_hints_correct: int = 0
    avoid_hints: int = 0
    avoid_hints_on_hazard: int = 0
    preferred_hints: int = 0
    preferred_hints_on_good: int = 0
    water_cells: int = 0
    water_cells_flagged: int = 0
    has_person: bool = False
    person_mentioned: bool = False
    route_water: float | None = None
    route_hazard_cells: float | None = None
    straight_water: float | None = None


def score_answer(
    answer: dict,
    goal_cell: str | None,
    grid: ImageGrid,
    groups: dict[str, dict[str, float]],
    terrains: dict[str, dict[str, float]],
) -> FrameScore:
    """Score one checked answer (as saved by `runner.answer_to_dict`).

    groups: `cell_fractions` with CLASS_GROUPS; terrains: `cell_fractions` with TERRAIN_CLASSES.
    """
    score = FrameScore()
    hints = {h["cell"]: h for h in answer["hints"]}
    for cell, hint in hints.items():
        g = groups[cell]
        score.hints += 1
        score.hints_terrain_matched += terrains[cell][hint["terrain"]] >= TERRAIN_MATCH_MIN
        if hint["terrain"] == "water":
            score.water_hints += 1
            score.water_hints_correct += g["water"] >= WATER_HINT_MIN
        if hint["traversability"] in ("avoid", "impassable"):
            score.avoid_hints += 1
            score.avoid_hints_on_hazard += g["water"] + g["mud"] + g["hard"] + g["bush"] >= AVOID_HAZARD_MIN
        if hint["traversability"] == "preferred":
            score.preferred_hints += 1
            score.preferred_hints_on_good += (
                g["good"] >= PREFERRED_GOOD_MIN and g["water"] < PREFERRED_BAD_MAX and g["hard"] < PREFERRED_BAD_MAX
            )

    # Recall: water cells in the bottom half of the image (the ground near the vehicle).
    for cell, g in groups.items():
        if grid.parse(cell)[0] >= grid.rows // 2 and g["water"] >= WATER_CELL_MIN:
            score.water_cells += 1
            hint = hints.get(cell)
            score.water_cells_flagged += hint is not None and (
                hint["terrain"] in ("water", "mud") or hint["traversability"] in CAUTIOUS_LEVELS
            )

    score.has_person = any(g["person"] >= PERSON_PRESENT_MIN for g in groups.values())
    score.person_mentioned = "person" in answer["reasoning"].lower() or "people" in answer["reasoning"].lower() or any(
        h["terrain"] == "person_or_vehicle" for h in answer["hints"]
    )

    route = answer["waypoints"]
    score.route_water = mean_share(route, groups, "water")
    if len(route) > 1:
        score.route_hazard_cells = float(np.mean([
            groups[c]["water"] >= ROUTE_HAZARD_MIN or groups[c]["hard"] >= ROUTE_HAZARD_MIN for c in route[1:]
        ]))
    if goal_cell is not None:
        score.straight_water = mean_share(straight_route(grid, goal_cell), groups, "water")
    return score


def summarize(scores: list[FrameScore]) -> dict[str, tuple[float | None, str]]:
    """Totals over many frames: metric name -> (value, short text like "30/40" or "0.31 (n=8)").

    Ratios are pooled over all hints/cells (not averaged per frame); shares are averaged over the
    frames where they are defined. A value is None when there is nothing to measure.
    """
    def ratio(hit: str, total: str) -> tuple[float | None, str]:
        h, t = sum(getattr(s, hit) for s in scores), sum(getattr(s, total) for s in scores)
        return (h / t if t else None), f"{h}/{t}"

    def share(field: str) -> tuple[float | None, str]:
        values = [getattr(s, field) for s in scores if getattr(s, field) is not None]
        return (float(np.mean(values)) if values else None), f"n={len(values)}"

    with_person = [s for s in scores if s.has_person]
    mentioned = sum(s.person_mentioned for s in with_person)
    return {
        "hint terrain matches labels": ratio("hints_terrain_matched", "hints"),
        "water hints on water": ratio("water_hints_correct", "water_hints"),
        "avoid hints on hazards": ratio("avoid_hints_on_hazard", "avoid_hints"),
        "preferred hints on good ground": ratio("preferred_hints_on_good", "preferred_hints"),
        "near water cells flagged": ratio("water_cells_flagged", "water_cells"),
        "person frames mentioning one": ((mentioned / len(with_person)) if with_person else None,
                                         f"{mentioned}/{len(with_person)}"),
        "route water share": share("route_water"),
        "straight-ahead water share": share("straight_water"),
        "route cells on hazards": share("route_hazard_cells"),
    }
