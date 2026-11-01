"""Tests for the VLM answer checker and the prompt. No model or dataset needed."""

import json

import pytest

from offroad.vlm.answer import TERRAIN_TYPES, TRAVERSABILITY_LEVELS, CellHint, extract_json, parse_answer
from offroad.vlm.grid import ImageGrid
from offroad.vlm.prompt import MAX_CELLS_PER_HINT, MAX_HINTS, VEHICLE_BRIEFING, answer_template, build_prompt

GRID = ImageGrid(height=1200, width=1920)  # 8 rows (A-H) x 12 columns

CLEAN = json.dumps({
    "waypoints": ["H7", "G7", "F8", "E8"],
    "cost_hints": [
        {"cells": ["G4", "G5"], "terrain": "water", "traversability": "avoid"},
        {"cells": ["F10"], "terrain": "tall_grass_or_bush", "traversability": "ok"},
    ],
    "reasoning": "Stay on the trail and avoid the puddle.",
})


def test_clean_answer():
    answer = parse_answer(CLEAN, GRID)
    assert answer.valid
    assert answer.issues == []
    assert answer.waypoints == ["H7", "G7", "F8", "E8"]
    assert answer.hints["G4"] == CellHint("water", "avoid")
    assert answer.hints["G5"] == CellHint("water", "avoid")
    assert answer.hints["F10"] == CellHint("tall_grass_or_bush", "ok")
    assert answer.reasoning == "Stay on the trail and avoid the puddle."


def test_code_fence_is_not_an_issue():
    answer = parse_answer(f"```json\n{CLEAN}\n```", GRID)
    assert answer.valid
    assert answer.issues == []


def test_text_around_json_is_ignored_but_reported():
    answer = parse_answer(f"Sure! Here is my answer:\n{CLEAN}\nHope this helps.", GRID)
    assert answer.valid
    assert any("before" in issue for issue in answer.issues)
    assert any("after" in issue for issue in answer.issues)


def test_trailing_commas_are_repaired():
    text = '{"waypoints": ["H7", "G7",], "cost_hints": [], "reasoning": "x",}'
    answer = parse_answer(text, GRID)
    assert answer.waypoints == ["H7", "G7"]
    assert any("trailing commas" in issue for issue in answer.issues)


def test_braces_inside_strings_do_not_confuse_the_parser():
    text = '{"waypoints": ["H7"], "cost_hints": [], "reasoning": "a } inside { text"}'
    assert parse_answer(text, GRID).reasoning == "a } inside { text"


def test_no_json_gives_invalid_answer():
    answer = parse_answer("I cannot help with that.", GRID)
    assert not answer.valid
    assert any("no JSON" in issue for issue in answer.issues)


def test_empty_answer_is_reported_as_empty():
    for text in ["", "  \n "]:
        answer = parse_answer(text, GRID)
        assert not answer.valid
        assert answer.issues == ["the answer is empty"]


def test_unclosed_json_gives_invalid_answer():
    answer = parse_answer('{"waypoints": ["H7"', GRID)
    assert not answer.valid
    assert any("not closed" in issue for issue in answer.issues)


def test_bad_cells_are_dropped_and_lowercase_fixed():
    text = '{"waypoints": ["h7", "I3", "G13", 5, "g7"], "cost_hints": [], "reasoning": ""}'
    answer = parse_answer(text, GRID)
    assert answer.waypoints == ["H7", "G7"]
    assert sum("dropped" in issue for issue in answer.issues) == 3


def test_consecutive_repeated_waypoints_are_merged():
    text = '{"waypoints": ["H7", "H7", "G7", "H7"], "cost_hints": []}'
    assert parse_answer(text, GRID).waypoints == ["H7", "G7", "H7"]


def test_all_waypoints_bad_makes_answer_invalid():
    answer = parse_answer('{"waypoints": ["Z9"], "cost_hints": []}', GRID)
    assert not answer.valid
    assert "no usable waypoints" in answer.issues


def test_missing_fields_are_reported():
    answer = parse_answer('{"waypoints": ["H7"]}', GRID)
    assert answer.valid
    assert "'cost_hints' is missing" in answer.issues


def test_synonyms_are_normalized():
    text = json.dumps({
        "waypoints": ["H7"],
        "cost_hints": [
            {"cells": ["G4"], "terrain": "Puddle", "traversability": "Avoid"},
            {"cell": "F10", "terrain": "tall grass", "traversability": "passable"},
            {"cells": "E2", "terrain": "Tree", "traversability": "blocked"},
        ],
    })
    hints = parse_answer(text, GRID).hints
    assert hints["G4"] == CellHint("water", "avoid")
    assert hints["F10"] == CellHint("tall_grass_or_bush", "ok")
    assert hints["E2"] == CellHint("tree_or_log", "impassable")


def test_unknown_words_drop_the_hint():
    text = json.dumps({
        "waypoints": ["H7"],
        "cost_hints": [
            {"cells": ["G4"], "terrain": "lava", "traversability": "avoid"},
            {"cells": ["G5"], "terrain": "mud", "traversability": "maybe"},
        ],
    })
    answer = parse_answer(text, GRID)
    assert answer.hints == {}
    assert any("lava" in issue for issue in answer.issues)
    assert any("maybe" in issue for issue in answer.issues)


def test_conflicting_hints_keep_the_more_cautious():
    text = json.dumps({
        "waypoints": ["H7"],
        "cost_hints": [
            {"cells": ["G4"], "terrain": "short_grass", "traversability": "ok"},
            {"cells": ["G4"], "terrain": "mud", "traversability": "avoid"},
            {"cells": ["G4"], "terrain": "short_grass", "traversability": "preferred"},
        ],
    })
    answer = parse_answer(text, GRID)
    assert answer.hints["G4"] == CellHint("mud", "avoid")
    assert any("more cautious" in issue for issue in answer.issues)


def test_route_conflicts_lists_waypoints_in_own_avoid_cells():
    # The real 00000/120 answer from Qwen3-VL-4B: "steer left", then straight through F6/G6.
    text = json.dumps({
        "waypoints": ["H6", "G6", "F6", "E6", "D6"],
        "cost_hints": [
            {"cells": ["F6", "G6"], "terrain": "mud", "traversability": "avoid"},
            {"cells": ["F8"], "terrain": "water", "traversability": "impassable"},
            {"cells": ["E6"], "terrain": "short_grass", "traversability": "caution"},
        ],
    })
    answer = parse_answer(text, GRID)
    assert answer.route_conflicts == ["G6", "F6"]
    assert not any("conflict" in issue for issue in answer.issues)  # content, not a format issue


def test_no_route_conflicts_for_a_consistent_answer():
    assert parse_answer(CLEAN, GRID).route_conflicts == []


def test_non_object_json():
    answer = parse_answer("[1, 2, 3]", GRID)
    assert not answer.valid


def test_extract_json_returns_object():
    obj, issues = extract_json('{"a": 1}')
    assert obj == {"a": 1}
    assert issues == []


# --- prompt ---


def test_copied_template_is_caught():
    # A model that copies the template must not produce a usable answer, and every
    # placeholder must show up as an issue.
    answer = parse_answer(json.dumps(answer_template(GRID)), GRID)
    assert not answer.valid
    assert answer.hints == {}
    assert sum("dropped" in issue for issue in answer.issues) == 4  # 3 waypoints + 1 hint
    assert "no usable waypoints" in answer.issues


def test_template_key_order_reasoning_then_route_then_hints():
    # Reasoning first (describe before deciding); the route before the hints, so a cut-off answer keeps it.
    assert list(answer_template(GRID)) == ["reasoning", "waypoints", "cost_hints"]
    prompt = build_prompt(GRID)
    assert prompt.index('"reasoning"') < prompt.index('"waypoints"') < prompt.index('"cost_hints"')


def test_prompt_caps_the_hints():
    prompt = build_prompt(GRID)
    assert f"at most {MAX_HINTS} entries" in prompt
    assert f"at most {MAX_CELLS_PER_HINT} cells" in prompt
    assert "Skip the sky" in prompt


def test_template_has_no_real_cells():
    # Nothing in the prompt's format block may be a valid cell the model could copy.
    template = json.dumps(answer_template(GRID))
    for label in GRID.all_labels():
        assert f'"{label}"' not in template


def test_prompt_lists_every_allowed_word_and_the_grid_size():
    prompt = build_prompt(GRID)
    for word in TERRAIN_TYPES + TRAVERSABILITY_LEVELS:
        assert word in prompt
    assert "A to H" in prompt
    assert "A1 to H12" in prompt
    assert "H6 and H7" in prompt  # where the vehicle is


def test_prompt_goal():
    assert "cell F7" in build_prompt(GRID, goal_cell="f7")
    assert "20 m straight ahead" in build_prompt(GRID)
    with pytest.raises(ValueError):
        build_prompt(GRID, goal_cell="Z99")


def test_vehicle_briefing_can_be_left_out():
    assert VEHICLE_BRIEFING in build_prompt(GRID)
    assert VEHICLE_BRIEFING not in build_prompt(GRID, vehicle_briefing=None)


def test_prompt_works_for_small_and_odd_grids():
    for rows, cols in [(2, 3), (5, 5), (26, 20)]:
        grid = ImageGrid(height=600, width=600, rows=rows, cols=cols)
        assert not parse_answer(json.dumps(answer_template(grid)), grid).valid
        assert build_prompt(grid)
