"""Text prompts for the path selector, the history summary, and a forgiving answer checker (pure).

Two modes (docs/approach-2-plan.md, section 2):
- **Mode A**, "keep going": no destination; choose a safe candidate and keep driving.
- **Mode B**, "keep going + a coarse direction": the same, plus a direction word ("bear left",
  "slight left", "straight", "slight right", "bear right") to follow when it's safe.

The answer is JSON: {"safe_until": {color: "all" | "tick 2" | "tick 1" | "none"}, "choice":
color, "action": "continue" | "stop", "reasoning": one sentence}. The per-candidate judgement
comes first: with the choice first, the zero-shot 4B justified its pick by calling every other
candidate "none" (first try on 6 validation frames). Lessons from approach 1: the format uses
placeholders instead of a concrete example (small models copy examples), and the reasoning is
capped at one short sentence (the safe_until list is at most 4 entries, so it can't run away).

The candidates are listed in a random order (seeded per frame), not left to right, so the
model can't learn "the first one listed" or "the leftmost one" (section 8 of the plan).

The history describes earlier decisions without colors (colors are re-assigned every frame):
when it was, whether the vehicle continued or stopped, and which way the chosen path went.
"""

import json
import re
from dataclasses import dataclass, field

import numpy as np

from offroad.nav.driven_path import direction_word
from offroad.nav.labels import SAFE_ANSWERS

MODES = ("A", "B")
ACTIONS = ("continue", "stop")
HISTORY_LENGTH = 3  # setting 9: the last 3 decisions
REASONING_MAX_WORDS = 30

DIRECTION_MEANINGS = {
    "bear left": "turn clearly to the left",
    "slight left": "drift a little to the left",
    "straight": "keep going straight ahead",
    "slight right": "drift a little to the right",
    "bear right": "turn clearly to the right",
}

_INTRO = """You are driving an off-road vehicle. The image is from its front camera.
Your task: {task}

Colored lines drawn on the ground are candidate paths that a planner found drivable from its 3D sensor. Each is up to 15 m long and is labelled with its color name at its far end. The vehicle is just below the bottom of the image: the camera can't see the first 4 m, so the lines start at the bottom edge. White dots on each line mark 5 m (tick 1, at or just below the bottom edge), 10 m (tick 2) and 15 m (the end).

Candidates: {colors}.
Recent decisions (oldest first): {history}
"""

_RULES = """First judge each candidate on its own (not by whether you will choose it): how far along it is safe to drive.
- "all": safe all the way to its end,
- "tick 2": safe to the 10 m dot, but not further,
- "tick 1": safe to the 5 m dot, but not further,
- "none": not safe even to the 5 m dot.
Unsafe means: deep water, or something the vehicle would hit (tree trunk, rock, fence, pole, log, building, vehicle, person). Grass, low bushes, mud and puddles are drivable, but prefer a dry path when one is as good.

Then choose one candidate to drive on. {preference} The action is "stop" only if no candidate is safe to at least the 5 m dot; otherwise "continue".

Answer with JSON only, exactly in this format (one safe_until entry per candidate):
{answer_format}"""

_FORMAT_WITH_REASONING = ('{"safe_until": {"<color>": "<all, tick 2, tick 1 or none>"}, "choice": "<color>", '
                          '"action": "<continue or stop>", "reasoning": "<one short sentence>"}')
# Phase 5 (user decision 2026-11-23): the fine-tuned model is trained and asked without the reasoning
# field, since the training data has no human-written reasons.
_FORMAT_WITHOUT_REASONING = ('{"safe_until": {"<color>": "<all, tick 2, tick 1 or none>"}, "choice": "<color>", '
                             '"action": "<continue or stop>"}')

# Wordings of the task and the preference (phase 5: training uses all of them, so the model learns
# the meaning rather than one exact sentence). Wording 0 is the one used in phases 3-4 and for evaluation.
_TASKS_A = (
    "keep driving forward safely until you are told to stop. There is no destination.",
    "drive on safely and keep going until someone tells you to stop. There is no goal location.",
    "continue driving forward safely; there is no destination, so keep going until told to stop.",
)
_TASKS_B = (
    "keep driving forward safely until you are told to stop. There is no destination, but the "
    "operator asked you to go \"{word}\" ({meaning}).",
    "drive on safely and keep going until someone tells you to stop. There is no goal location, but "
    "the operator wants you to go \"{word}\" ({meaning}).",
    "continue driving forward safely until told to stop. There is no destination; the operator's "
    "direction is \"{word}\" ({meaning}).",
)
_PREFERS_A = (
    "Prefer following the trail or track when there is one.",
    "If there is a trail or track, prefer staying on it.",
    "When a trail or track is visible, prefer the candidate that follows it.",
)
_PREFERS_B = (
    "Among the safe candidates, choose the one that best follows the direction \"{word}\".",
    "Of the safe candidates, pick the one that matches the direction \"{word}\" best.",
    "Choose the safe candidate that goes most in the direction \"{word}\".",
)
WORDINGS = len(_TASKS_A)
_TASK_A, _TASK_B, _PREFER_A, _PREFER_B = _TASKS_A[0], _TASKS_B[0], _PREFERS_A[0], _PREFERS_B[0]


@dataclass
class HistoryEntry:
    """One earlier decision: when (seconds), continue / stop, and the chosen path's end bearing (+ = left; None if stopped)."""

    timestamp: float
    action: str
    bearing_deg: float | None


def history_text(entries: list[HistoryEntry], now: float, length: int = HISTORY_LENGTH) -> str:
    """The last `length` decisions as one line of text, oldest first; "none" if there are none."""
    parts = []
    for e in entries[-length:]:
        when = f"{now - e.timestamp:.1f} s ago"
        if e.action == "stop" or e.bearing_deg is None:
            parts.append(f"{when}: stopped")
        else:
            parts.append(f"{when}: continued {direction_word(e.bearing_deg)}")
    return "; ".join(parts) + "." if parts else "none."


WRONG_TURN_DEG = 35.0  # the made-up turn in a wrong history: "bear left" / "bear right"


def wrong_history(entries: list[HistoryEntry], seed: int) -> tuple[list[HistoryEntry], str | None]:
    """The history with its latest decision replaced by a sharp turn the vehicle did not take (phase 4 test).

    The made-up turn goes to the side opposite the real latest decision's turn, or to a random
    (seeded) side if it went straight or stopped. Returns (new entries, the made-up side "left" /
    "right"), or (entries, None) if there is no history yet. The input list is not changed.
    In closed loop the history holds the model's own mistakes; this checks whether a wrong
    entry pulls the next choice toward it.
    """
    if not entries:
        return list(entries), None
    last = entries[-1]
    if last.bearing_deg is not None and direction_word(last.bearing_deg) != "straight":
        side = "right" if last.bearing_deg > 0 else "left"
    else:
        side = ("left", "right")[int(np.random.default_rng(seed).integers(2))]
    fake = HistoryEntry(last.timestamp, "continue", WRONG_TURN_DEG if side == "left" else -WRONG_TURN_DEG)
    return list(entries[:-1]) + [fake], side


def build_prompt(colors: list[str], mode: str = "A", direction: str | None = None,
                 history: str = "none.", reasoning: bool = True, wording: int = 0) -> str:
    """The text prompt for one frame.

    colors: candidate color names in the order to list them (already shuffled by the caller).
    direction: Mode B's word (required in Mode B, ignored in Mode A). history: from `history_text`.
    reasoning: ask for the one-sentence reasoning (phases 3-4: yes; the fine-tuned model: no).
    wording: which wording of the task and preference (0 = the evaluation wording; training uses all).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not 0 <= wording < WORDINGS:
        raise ValueError(f"wording must be 0..{WORDINGS - 1}, got {wording}")
    if mode == "B":
        if direction not in DIRECTION_MEANINGS:
            raise ValueError(f"Mode B needs a direction word from {list(DIRECTION_MEANINGS)}, got {direction!r}")
        task = _TASKS_B[wording].format(word=direction, meaning=DIRECTION_MEANINGS[direction])
        preference = _PREFERS_B[wording].format(word=direction)
    else:
        task, preference = _TASKS_A[wording], _PREFERS_A[wording]
    answer_format = _FORMAT_WITH_REASONING if reasoning else _FORMAT_WITHOUT_REASONING
    return _INTRO.format(task=task, colors=", ".join(colors), history=history) + "\n" + \
        _RULES.format(preference=preference, answer_format=answer_format)


# --- the answer checker ---


@dataclass
class ParsedAnswer:
    """A checked answer. valid = it has a known choice, an action, and a tick for every candidate.

    choice: a candidate color or None; action: "continue" / "stop" or None; safe_until: {color:
    tick answer} for the candidates it answered; problems: what was wrong or missing.
    """

    valid: bool
    choice: str | None
    action: str | None
    safe_until: dict[str, str]
    reasoning: str
    problems: list[str] = field(default_factory=list)


def _first_json_object(text: str) -> dict | None:
    """The first JSON object in the text (code fences and text around it are ignored)."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _color_in(value, colors: list[str]) -> str | None:
    """The one candidate color named in a value like "red", "Red path" or "the RED line"; None if none or several."""
    if not isinstance(value, str):
        return None
    found = [c for c in colors if re.search(rf"\b{re.escape(c)}\b", value.lower())]
    return found[0] if len(found) == 1 else None


def normalize_tick(value) -> str | None:
    """Map answers like "all", "Tick 2", "tick2", "2", "10 m", "none", 0 to a SAFE_ANSWERS value; None if unclear."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        value = str(value)
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in SAFE_ANSWERS:
        return v
    if v in ("all", "end", "15", "15 m", "15m", "tick 3", "tick3", "3") or "all" in v.split():
        return "all"
    if v in ("tick2", "2", "10", "10 m", "10m"):
        return "tick 2"
    if v in ("tick1", "1", "5", "5 m", "5m"):
        return "tick 1"
    if v in ("none", "0", "0 m", "no", "unsafe"):
        return "none"
    return None


def parse_answer(text: str, colors: list[str]) -> ParsedAnswer:
    """Check a model reply against the frame's candidate colors (forgiving about case and wording)."""
    data = _first_json_object(text)
    if data is None:
        return ParsedAnswer(False, None, None, {}, "", ["no JSON object"])
    data = {str(k).strip().lower(): v for k, v in data.items()}
    problems = []

    choice = _color_in(data.get("choice"), colors)
    if choice is None:
        problems.append(f"choice {data.get('choice')!r} is not one candidate color")

    action = data.get("action")
    action = action.strip().lower() if isinstance(action, str) else None
    action = {"go": "continue", "drive": "continue", "halt": "stop"}.get(action, action)
    if action not in ACTIONS:
        problems.append(f"action {data.get('action')!r} is not continue / stop")
        action = None

    safe_until = {}
    raw = data.get("safe_until")
    if isinstance(raw, dict):
        for key, value in raw.items():
            color = _color_in(str(key), colors)
            tick = normalize_tick(value)
            if color is not None and tick is not None:
                safe_until[color] = tick
    missing = [c for c in colors if c not in safe_until]
    if missing:
        problems.append(f"no safe_until for {', '.join(missing)}")

    reasoning = data.get("reasoning")
    reasoning = " ".join(str(reasoning).split()[:REASONING_MAX_WORDS]) if reasoning is not None else ""
    return ParsedAnswer(not problems, choice, action, safe_until, reasoning, problems)


def chosen_bearing(answer: ParsedAnswer, colors: list[str], end_bearings_deg: list[float]) -> float | None:
    """End bearing of the chosen candidate (for the history), or None if it stopped or chose nothing valid."""
    if answer.action == "stop" or answer.choice is None:
        return None
    return float(end_bearings_deg[colors.index(answer.choice)])


def shuffled(items: list, seed: int) -> list:
    """The items in a random order, repeatable for a seed."""
    return [items[k] for k in np.random.default_rng(seed).permutation(len(items))]
