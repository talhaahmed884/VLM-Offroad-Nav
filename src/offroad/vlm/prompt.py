"""The text prompt sent to the VLM together with the gridded camera image (pure, no file I/O).

The prompt explains the grid and the vehicle, gives the goal, and asks for the JSON format
that `answer.parse_answer` checks.

The format is shown as a template with <placeholders>, not as a filled-in example: in the
first real run (2026-11-01) Qwen3-VL-4B copied a concrete example's cells word for word
instead of looking at the image. Placeholders can't be copied unnoticed, because the
checker drops them and reports each one.

The keys are ordered reasoning -> waypoints -> cost_hints: the reasoning makes the model
describe what it sees before it picks a route, and the route comes before the hints so an
answer that runs too long loses hints rather than the route. The hints are capped, because
asked for "every hazard", Qwen3-VL-4B listed every grid column top to bottom (sky included)
until it ran out of tokens.
"""

import json

from offroad.vlm.answer import TERRAIN_TYPES, TRAVERSABILITY_LEVELS
from offroad.vlm.grid import ROW_LETTERS, ImageGrid

# A one-sentence briefing on what the vehicle can and cannot drive through. It is fair
# context (an operator would know this), but it does tell the model part of the answer;
# pass vehicle_briefing=None to build_prompt to leave it out.
VEHICLE_BRIEFING = (
    "The vehicle can drive through short and tall grass and small, soft bushes. It must not "
    "hit trees, logs, rocks, fences, people, vehicles, or deep water, and it can get stuck in "
    "mud and puddles, so avoid those when there is a reasonable alternative."
)

# Limits on cost_hints stated in the prompt, so answers stay short (and fast).
MAX_HINTS = 6
MAX_CELLS_PER_HINT = 8


def answer_template(grid: ImageGrid) -> dict:
    """The answer format with <placeholders> in place of values (nothing in it is a valid answer)."""
    last_row = ROW_LETTERS[grid.rows - 1]
    return {
        "reasoning": "<one or two sentences: what you see ahead, and which way is best>",
        "waypoints": [f"<nearest cell, in row {last_row}>", "<next cell>", "<cell closest to the goal>"],
        "cost_hints": [
            {"cells": ["<cell>", "<cell>"], "terrain": "<terrain word>", "traversability": "<traversability word>"},
        ],
    }


def build_prompt(
    grid: ImageGrid,
    goal_cell: str | None = None,
    goal_distance_m: float = 20.0,
    vehicle_briefing: str | None = VEHICLE_BRIEFING,
) -> str:
    """Prompt text for one gridded image.

    goal_cell: grid cell where the goal appears in the image (e.g. "F7"), if known.
    goal_distance_m: how far ahead the goal is, meters (said in the prompt).
    vehicle_briefing: sentence about what the vehicle can drive through; None leaves it out.
    """
    last_row = ROW_LETTERS[grid.rows - 1]
    center_left, center_right = grid.label(grid.rows - 1, (grid.cols - 1) // 2), grid.label(grid.rows - 1, grid.cols // 2)
    vehicle_cells = center_left if center_left == center_right else f"{center_left} and {center_right}"
    if goal_cell is not None:
        goal = f"Drive toward cell {grid.label(*grid.parse(goal_cell))}, about {goal_distance_m:.0f} m straight ahead."
    else:
        goal = f"Drive toward a point about {goal_distance_m:.0f} m straight ahead of the vehicle."

    lines = [
        "You are the high-level route guide for an off-road ground vehicle (about 1.5 m wide). "
        "The image comes from its front camera.",
        "",
        f"A grid is drawn on the image: rows A to {last_row} from top to bottom, columns 1 to {grid.cols} "
        f"from left to right. Each cell's name (for example \"{grid.label(grid.rows // 2, grid.cols // 2)}\") "
        "is printed in its top-left corner. The vehicle is just below the bottom edge of the image, under "
        f"{vehicle_cells}. Lower rows are closer to the vehicle; higher rows are farther away.",
        "",
        goal,
    ]
    if vehicle_briefing:
        lines += ["", vehicle_briefing]
    lines += [
        "",
        "Look carefully at this image, then answer with ONLY a JSON object, with no other text, in this "
        "format. Replace every <...> with your own values based on what you see; do not copy the placeholders.",
        json.dumps(answer_template(grid), indent=2),
        "",
        "Rules:",
        "- \"reasoning\" comes first: one or two short sentences about the hazards ahead and which way is best.",
        f"- \"waypoints\": 3 to 8 cells on drivable ground, ordered from the nearest (row {last_row}) to the goal. "
        "Consecutive waypoints should be in the same or neighbouring cells. Steer around water, mud, and "
        "obstacles unless there is no other way.",
        f"- \"cost_hints\": at most {MAX_HINTS} entries, each with at most {MAX_CELLS_PER_HINT} cells. Only mark ground "
        "cells that matter for the route: the hazards you mentioned (water, mud, rocks, trees, people, ...) and "
        "the ground you chose to drive on. Skip the sky, distant scenery, and cells far from the route. Each "
        "entry has \"cells\" (a list of cell names), \"terrain\" and \"traversability\".",
        f"- \"terrain\" must be one of: {', '.join(TERRAIN_TYPES)}.",
        f"- \"traversability\" must be one of: {', '.join(TRAVERSABILITY_LEVELS)} (from best to worst).",
        f"- Use only cell names that exist in the grid (A1 to {grid.label(grid.rows - 1, grid.cols - 1)}).",
    ]
    return "\n".join(lines)
