"""Tests for scoring VLM answers against label images, and for drawing them. No dataset needed."""

import numpy as np
import pytest

from offroad.viz.vlm_answer import colorize_labels, draw_answer_figure
from offroad.vlm.grid import ImageGrid
from offroad.vlm.scoring import (
    CLASS_GROUPS,
    TERRAIN_CLASSES,
    FrameScore,
    cell_fractions,
    class_ids,
    score_answer,
    straight_route,
    summarize,
)

ID_TO_NAME = {3: "grass", 4: "tree", 6: "water", 17: "person", 19: "bush", 31: "puddle", 33: "mud"}
GRASS, TREE, PERSON, PUDDLE = 3, 4, 17, 31
GRID = ImageGrid(height=80, width=120)  # 8 x 12 cells of 10 x 10 px


def label_image(cells: dict[str, int] = None) -> np.ndarray:
    """All grass, with whole cells set to other classes."""
    labels = np.full((80, 120), GRASS, dtype=np.uint8)
    for cell, class_id in (cells or {}).items():
        r, c = GRID.parse(cell)
        labels[r * 10:(r + 1) * 10, c * 10:(c + 1) * 10] = class_id
    return labels


def fractions(labels):
    return cell_fractions(labels, GRID, ID_TO_NAME, CLASS_GROUPS), cell_fractions(labels, GRID, ID_TO_NAME, TERRAIN_CLASSES)


def answer(waypoints=(), hints=(), reasoning=""):
    return {"waypoints": list(waypoints), "reasoning": reasoning,
            "hints": [{"cell": c, "terrain": t, "traversability": lv} for c, t, lv in hints]}


# --- cell fractions ---


def test_cell_fractions_per_cell():
    groups, _ = fractions(label_image({"G5": PUDDLE}))
    assert groups["G5"]["water"] == 1.0
    assert groups["G5"]["good"] == 0.0
    assert groups["G6"]["water"] == 0.0
    assert groups["G6"]["good"] == 1.0


def test_cell_fractions_partial_cell():
    labels = label_image()
    labels[60:65, 40:50] = PUDDLE  # top half of G5
    groups, _ = fractions(labels)
    assert groups["G5"]["water"] == pytest.approx(0.5)


def test_cell_fractions_checks_size():
    with pytest.raises(ValueError):
        cell_fractions(np.zeros((10, 10), dtype=np.uint8), GRID, ID_TO_NAME, CLASS_GROUPS)


def test_class_ids_skips_unknown_names():
    assert sorted(class_ids(ID_TO_NAME, ("water", "puddle", "lava"))) == [6, 31]


def test_straight_route():
    assert straight_route(GRID, "D6") == ["H6", "G6", "F6", "E6", "D6"]


# --- scoring one answer ---


def test_hint_checks():
    groups, terrains = fractions(label_image({"G5": PUDDLE, "F8": TREE}))
    score = score_answer(answer(
        waypoints=["H6", "G6"],
        hints=[("G5", "water", "avoid"),            # right: water, and a hazard
               ("G7", "water", "avoid"),            # wrong: grass
               ("F8", "tree_or_log", "impassable"),  # right
               ("E6", "short_grass", "preferred")],  # right
    ), "E6", GRID, groups, terrains)
    assert (score.hints, score.hints_terrain_matched) == (4, 3)
    assert (score.water_hints, score.water_hints_correct) == (2, 1)
    assert (score.avoid_hints, score.avoid_hints_on_hazard) == (3, 2)
    assert (score.preferred_hints, score.preferred_hints_on_good) == (1, 1)


def test_preferred_on_water_is_wrong():
    groups, terrains = fractions(label_image({"G5": PUDDLE}))
    score = score_answer(answer(hints=[("G5", "dirt_or_gravel", "preferred")]), None, GRID, groups, terrains)
    assert (score.preferred_hints, score.preferred_hints_on_good) == (1, 0)


def test_water_recall_only_counts_the_lower_half():
    groups, terrains = fractions(label_image({"G5": PUDDLE, "G6": PUDDLE, "B5": PUDDLE}))
    score = score_answer(answer(hints=[("G5", "mud", "caution")]), None, GRID, groups, terrains)
    assert (score.water_cells, score.water_cells_flagged) == (2, 1)  # B5 is in the top half


def test_person_detection():
    groups, terrains = fractions(label_image({"D6": PERSON}))
    assert score_answer(answer(reasoning="A person stands ahead."), None, GRID, groups, terrains).person_mentioned
    score = score_answer(answer(reasoning="Open grass."), None, GRID, groups, terrains)
    assert score.has_person and not score.person_mentioned
    groups, terrains = fractions(label_image())
    assert not score_answer(answer(), None, GRID, groups, terrains).has_person


def test_route_and_straight_water():
    groups, terrains = fractions(label_image({"G6": PUDDLE, "F6": PUDDLE}))
    score = score_answer(answer(waypoints=["H6", "G5", "F5", "E6"]), "E6", GRID, groups, terrains)
    assert score.route_water == 0.0  # went around the puddle
    assert score.straight_water == pytest.approx(2 / 3)  # G6, F6, E6 after the start
    assert score.route_hazard_cells == 0.0


def test_single_waypoint_has_no_route_share():
    groups, terrains = fractions(label_image())
    score = score_answer(answer(waypoints=["H6"]), None, GRID, groups, terrains)
    assert score.route_water is None and score.route_hazard_cells is None and score.straight_water is None


# --- summary ---


def test_summarize_pools_counts():
    scores = [FrameScore(water_hints=2, water_hints_correct=1, route_water=0.2, has_person=True, person_mentioned=True),
              FrameScore(water_hints=2, water_hints_correct=2, route_water=None, has_person=True)]
    summary = summarize(scores)
    assert summary["water hints on water"] == (0.75, "3/4")
    assert summary["route water share"] == (0.2, "n=1")
    assert summary["person frames mentioning one"] == (0.5, "1/2")
    assert summary["avoid hints on hazards"] == (None, "0/0")


# --- drawing ---


def test_colorize_labels():
    colors = colorize_labels(label_image({"G5": PUDDLE}), {GRASS: (0, 255, 0), PUDDLE: (0, 0, 255)})
    assert tuple(colors[0, 0]) == (0, 255, 0)
    assert tuple(colors[65, 45]) == (0, 0, 255)


def test_draw_answer_figure_shapes():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    a = {**answer(waypoints=["H6", "G6"], hints=[("G5", "water", "avoid")], reasoning="Go up."),
         "valid": True, "issues": [], "route_conflicts": []}
    with_labels = draw_answer_figure(image, label_image(), GRID, a, "G6", {GRASS: (0, 255, 0)}, "title")
    without = draw_answer_figure(image, None, GRID, a, None, {}, "title")
    assert with_labels.shape[1] == 240 and without.shape[1] == 120  # two panels vs one
    assert with_labels.shape[0] > 80 and with_labels.dtype == np.uint8  # header added on top
