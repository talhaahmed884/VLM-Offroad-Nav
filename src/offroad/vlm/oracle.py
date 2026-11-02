"""An "oracle" VLM answer made from the camera label image: what perfect perception would say (pure).

Used to find the upper bound of the hybrid planner. If even perfect hints barely change the
paths, the limit is the fusion design (coverage, weights, blind zone), not the VLM. Unlike
the real VLM it hints every grid cell that shows ground (no cap), and gives no waypoints, so
it tests the cost hints alone.

Per cell: the most cautious hazard covering at least `hazard_min_share` of the cell wins;
otherwise the drivable ground type covering the largest share (at least `ground_min_share`);
otherwise no hint (sky, mixed cells). The mapping follows the prompt's vehicle briefing.
"""

from offroad.vlm.grid import ImageGrid
from offroad.vlm.scoring import cell_fractions

# (ontology class names, terrain, traversability), hazards first, most cautious first.
HAZARD_RULES = (
    (("person", "vehicle"), "person_or_vehicle", "impassable"),
    (("tree", "log"), "tree_or_log", "impassable"),
    (("pole", "fence", "barrier", "building", "object"), "other_obstacle", "impassable"),
    (("water", "puddle"), "water", "avoid"),
    (("rubble",), "rocks", "avoid"),
    (("mud",), "mud", "caution"),
)
GROUND_RULES = (
    (("dirt", "asphalt", "concrete"), "dirt_or_gravel", "preferred"),
    (("grass",), "short_grass", "ok"),
    (("bush",), "tall_grass_or_bush", "ok"),
)
HAZARD_MIN_SHARE = 0.2
GROUND_MIN_SHARE = 0.5


def oracle_answer(
    label_image,
    grid: ImageGrid,
    id_to_name: dict[int, str],
    hazard_min_share: float = HAZARD_MIN_SHARE,
    ground_min_share: float = GROUND_MIN_SHARE,
) -> dict:
    """An answer dict like the runner saves ({"waypoints", "hints", "reasoning"}), from an (H, W) label image."""
    rules = HAZARD_RULES + GROUND_RULES
    shares = cell_fractions(label_image, grid, id_to_name, {str(k): names for k, (names, _, _) in enumerate(rules)})

    hints = []
    for cell in grid.all_labels():
        share = [shares[cell][str(k)] for k in range(len(rules))]
        chosen = next((k for k in range(len(HAZARD_RULES)) if share[k] >= hazard_min_share), None)
        if chosen is None:
            ground = range(len(HAZARD_RULES), len(rules))
            best = max(ground, key=lambda k: share[k])
            chosen = best if share[best] >= ground_min_share else None
        if chosen is not None:
            _, terrain, level = rules[chosen]
            hints.append({"cell": cell, "terrain": terrain, "traversability": level})
    return {"waypoints": [], "hints": hints, "reasoning": "oracle: hints from the camera label image"}
