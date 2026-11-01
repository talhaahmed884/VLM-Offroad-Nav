"""The VLM's answer format, and a checker that turns raw model text into a clean, validated answer.

Expected answer (a JSON object, see `prompt.py`):
    {
      "waypoints": ["H7", "G7", "F7"],
      "cost_hints": [{"cells": ["G4", "G5"], "terrain": "water", "traversability": "avoid"}],
      "reasoning": "..."
    }

`parse_answer` never raises: it repairs what it safely can (code fences, extra text around
the JSON, trailing commas, lowercase cell names, common synonyms), drops what it can't
(unknown cells or words), and lists every problem in `issues`. Pure functions, no file I/O.
"""

import json
import re
from dataclasses import dataclass, field

from offroad.vlm.grid import ImageGrid

TERRAIN_TYPES = (
    "dirt_or_gravel",
    "short_grass",
    "tall_grass_or_bush",
    "mud",
    "water",
    "rocks",
    "tree_or_log",
    "other_obstacle",
    "person_or_vehicle",
)

# From best to worst. Phase 5 turns these into cost multipliers for the planner.
TRAVERSABILITY_LEVELS = ("preferred", "ok", "caution", "avoid", "impassable")

# Words models commonly use instead of ours (after lowercasing, spaces/hyphens -> underscores).
TERRAIN_SYNONYMS = {
    "dirt": "dirt_or_gravel", "gravel": "dirt_or_gravel", "trail": "dirt_or_gravel",
    "path": "dirt_or_gravel", "road": "dirt_or_gravel", "ground": "dirt_or_gravel",
    "grass": "short_grass", "lawn": "short_grass",
    "tall_grass": "tall_grass_or_bush", "bush": "tall_grass_or_bush", "bushes": "tall_grass_or_bush",
    "shrub": "tall_grass_or_bush", "shrubs": "tall_grass_or_bush", "vegetation": "tall_grass_or_bush",
    "brush": "tall_grass_or_bush",
    "puddle": "water", "puddles": "water", "pond": "water",
    "rock": "rocks", "rubble": "rocks", "stones": "rocks",
    "tree": "tree_or_log", "trees": "tree_or_log", "log": "tree_or_log", "logs": "tree_or_log",
    "obstacle": "other_obstacle", "fence": "other_obstacle", "barrier": "other_obstacle",
    "person": "person_or_vehicle", "people": "person_or_vehicle", "vehicle": "person_or_vehicle",
}
TRAVERSABILITY_SYNONYMS = {
    "prefer": "preferred", "best": "preferred", "good": "preferred",
    "passable": "ok", "fine": "ok", "traversable": "ok",
    "careful": "caution", "risky": "caution",
    "avoid_if_possible": "avoid", "bad": "avoid",
    "blocked": "impassable", "not_passable": "impassable", "untraversable": "impassable",
}


@dataclass(frozen=True)
class CellHint:
    """What the VLM says about one grid cell."""

    terrain: str  # one of TERRAIN_TYPES
    traversability: str  # one of TRAVERSABILITY_LEVELS


@dataclass
class VlmAnswer:
    """A checked VLM answer.

    waypoints: cell names from near to far, normalized (e.g. "F6"), consecutive repeats removed.
    hints: cell name -> CellHint; if the model gave a cell twice, the more cautious hint is kept.
    reasoning: the model's short explanation ("" if missing).
    issues: every problem found, whether it was repaired or the item was dropped.
    """

    waypoints: list[str] = field(default_factory=list)
    hints: dict[str, CellHint] = field(default_factory=dict)
    reasoning: str = ""
    issues: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Usable for planning: at least one waypoint survived checking."""
        return len(self.waypoints) > 0

    @property
    def route_conflicts(self) -> list[str]:
        """Waypoints in cells the model itself marked "avoid" or "impassable".

        A self-consistency measure, not a format problem: e.g. Qwen3-VL-4B said "steer left"
        and marked F6/G6 as mud to avoid, then routed straight through F6 and G6.
        """
        bad_levels = ("avoid", "impassable")
        return [cell for cell in self.waypoints if cell in self.hints and self.hints[cell].traversability in bad_levels]


def extract_json(text: str) -> tuple[dict | None, list[str]]:
    """Find and decode the first JSON object in model text. Returns (object or None, issues)."""
    issues = []
    if not text.strip():
        return None, ["the answer is empty"]
    start = text.find("{")
    if start < 0:
        return None, ["no JSON object found in the answer"]
    if text[:start].strip().strip("`").strip().lower() not in ("", "json"):
        issues.append("extra text before the JSON was ignored")

    end = _matching_brace(text, start)
    if end is None:
        return None, issues + ["the JSON object is not closed"]
    if text[end + 1:].strip().strip("`").strip():
        issues.append("extra text after the JSON was ignored")
    snippet = text[start:end + 1]

    try:
        return json.loads(snippet), issues
    except json.JSONDecodeError:
        pass
    repaired = re.sub(r",\s*([}\]])", r"\1", snippet)  # trailing commas, a common model mistake
    try:
        obj = json.loads(repaired)
        return obj, issues + ["trailing commas were removed"]
    except json.JSONDecodeError as error:
        return None, issues + [f"invalid JSON: {error.msg}"]


def parse_answer(text: str, grid: ImageGrid) -> VlmAnswer:
    """Turn raw model text into a checked `VlmAnswer` (see the module docstring)."""
    answer = VlmAnswer()
    obj, answer.issues = extract_json(text)
    if not isinstance(obj, dict):
        if obj is not None:
            answer.issues.append("the JSON is not an object")
        return answer

    answer.waypoints = _parse_waypoints(obj.get("waypoints"), grid, answer.issues)
    answer.hints = _parse_hints(obj.get("cost_hints"), grid, answer.issues)
    reasoning = obj.get("reasoning", "")
    answer.reasoning = reasoning if isinstance(reasoning, str) else str(reasoning)
    if not answer.waypoints:
        answer.issues.append("no usable waypoints")
    return answer


def _matching_brace(text: str, start: int) -> int | None:
    """Index of the '}' that closes the '{' at `start`, skipping braces inside strings."""
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _normalize_cell(value, grid: ImageGrid, issues: list[str], where: str) -> str | None:
    if not isinstance(value, str):
        issues.append(f"{where}: {value!r} is not a cell name, dropped")
        return None
    try:
        return grid.label(*grid.parse(value))
    except ValueError:
        issues.append(f"{where}: {value!r} is not a cell in the grid, dropped")
        return None


def _normalize_word(value, allowed: tuple[str, ...], synonyms: dict[str, str]) -> str | None:
    if not isinstance(value, str):
        return None
    word = re.sub(r"[\s\-]+", "_", value.strip().lower())
    if word in allowed:
        return word
    return synonyms.get(word)


def _parse_waypoints(raw, grid: ImageGrid, issues: list[str]) -> list[str]:
    if raw is None:
        issues.append("'waypoints' is missing")
        return []
    if not isinstance(raw, list):
        issues.append("'waypoints' is not a list")
        return []
    waypoints = []
    for value in raw:
        cell = _normalize_cell(value, grid, issues, "waypoints")
        if cell is not None and (not waypoints or waypoints[-1] != cell):
            waypoints.append(cell)
    return waypoints


def _parse_hints(raw, grid: ImageGrid, issues: list[str]) -> dict[str, CellHint]:
    if raw is None:
        issues.append("'cost_hints' is missing")
        return {}
    if not isinstance(raw, list):
        issues.append("'cost_hints' is not a list")
        return {}
    hints: dict[str, CellHint] = {}
    for k, item in enumerate(raw):
        where = f"cost_hints[{k}]"
        if not isinstance(item, dict):
            issues.append(f"{where}: not an object, dropped")
            continue
        terrain = _normalize_word(item.get("terrain"), TERRAIN_TYPES, TERRAIN_SYNONYMS)
        level = _normalize_word(item.get("traversability"), TRAVERSABILITY_LEVELS, TRAVERSABILITY_SYNONYMS)
        if terrain is None:
            issues.append(f"{where}: unknown terrain {item.get('terrain')!r}, dropped")
            continue
        if level is None:
            issues.append(f"{where}: unknown traversability {item.get('traversability')!r}, dropped")
            continue
        cells = item.get("cells", item.get("cell"))
        if isinstance(cells, str):
            cells = [cells]
        if not isinstance(cells, list) or not cells:
            issues.append(f"{where}: no cells given, dropped")
            continue
        for value in cells:
            cell = _normalize_cell(value, grid, issues, where)
            if cell is None:
                continue
            hint = CellHint(terrain, level)
            if cell in hints and hints[cell] != hint:
                issues.append(f"{where}: {cell} was already described; kept the more cautious hint")
                hint = max(hints[cell], hint, key=lambda h: TRAVERSABILITY_LEVELS.index(h.traversability))
            hints[cell] = hint
    return hints
