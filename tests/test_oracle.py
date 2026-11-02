"""Tests for oracle hints made from label images. No dataset needed."""

import numpy as np

from offroad.vlm.answer import TERRAIN_TYPES, TRAVERSABILITY_LEVELS
from offroad.vlm.grid import ImageGrid
from offroad.vlm.oracle import GROUND_RULES, HAZARD_RULES, oracle_answer

ID_TO_NAME = {1: "dirt", 3: "grass", 4: "tree", 7: "sky", 17: "person", 19: "bush", 31: "puddle", 33: "mud"}
DIRT, GRASS, TREE, SKY, PERSON, BUSH, PUDDLE, MUD = 1, 3, 4, 7, 17, 19, 31, 33
GRID = ImageGrid(height=80, width=120)  # 8 x 12 cells of 10 x 10 px


def labels_with(cells: dict[str, int | tuple] = None, fill: int = GRASS) -> np.ndarray:
    """Label image filled with `fill`; a cell maps to a class, or to (class, rows) to fill only its top rows."""
    image = np.full((80, 120), fill, dtype=np.uint8)
    for cell, value in (cells or {}).items():
        class_id, rows = value if isinstance(value, tuple) else (value, 10)
        r, c = GRID.parse(cell)
        image[r * 10:r * 10 + rows, c * 10:(c + 1) * 10] = class_id
    return image


def hints_of(image: np.ndarray) -> dict[str, tuple[str, str]]:
    answer = oracle_answer(image, GRID, ID_TO_NAME)
    return {h["cell"]: (h["terrain"], h["traversability"]) for h in answer["hints"]}


def test_rules_use_only_allowed_words():
    for _, terrain, level in HAZARD_RULES + GROUND_RULES:
        assert terrain in TERRAIN_TYPES and level in TRAVERSABILITY_LEVELS


def test_plain_ground_cells():
    hints = hints_of(labels_with({"G5": DIRT, "G6": BUSH}))
    assert hints["G5"] == ("dirt_or_gravel", "preferred")
    assert hints["G6"] == ("tall_grass_or_bush", "ok")
    assert hints["G7"] == ("short_grass", "ok")


def test_hazards_win_over_ground():
    hints = hints_of(labels_with({"G5": (PUDDLE, 3), "G6": (TREE, 2), "G7": (MUD, 5), "G8": (TREE, 1)}))
    assert hints["G5"] == ("water", "avoid")  # 30% puddle beats 70% grass
    assert hints["G6"] == ("tree_or_log", "impassable")  # 20% is enough for a hazard
    assert hints["G7"] == ("mud", "caution")
    assert hints["G8"] == ("short_grass", "ok")  # 10% tree is not


def test_most_cautious_hazard_wins():
    image = labels_with({"G5": PUDDLE})
    image[60:63, 40:50] = PERSON  # 30% person in a puddle cell (G5 = pixel rows 60-69, columns 40-49)
    assert hints_of(image)["G5"] == ("person_or_vehicle", "impassable")


def test_sky_and_mixed_cells_get_no_hint():
    image = labels_with({"A1": SKY})
    image[60:64, 0:10] = BUSH  # G1: 40% bush, 40% grass, 20% sky -> no ground type reaches 50%
    image[68:70, 0:10] = SKY
    hints = hints_of(image)
    assert "A1" not in hints
    assert "G1" not in hints


def test_answer_has_no_waypoints():
    answer = oracle_answer(labels_with(), GRID, ID_TO_NAME)
    assert answer["waypoints"] == []
    assert len(answer["hints"]) == GRID.rows * GRID.cols  # every cell is grass here
