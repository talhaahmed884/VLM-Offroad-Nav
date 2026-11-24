"""Tests for the selector prompts, history text, answer checker, and scoring. No model needed."""

import numpy as np
import pytest

from offroad.nav.labels import CandidateLabels
from offroad.nav.prompts import (
    HistoryEntry,
    build_prompt,
    chosen_bearing,
    history_text,
    normalize_tick,
    parse_answer,
    shuffled,
)
from offroad.nav.scoring import score_answer

COLORS = ["red", "blue", "yellow"]


def labels(safe_until="all", match=False, puddle=None) -> CandidateLabels:
    return CandidateLabels(0.5 if match else 3.0, match, None, None, puddle, 15.0, safe_until, 0.0)


# --- prompts ---


def test_prompt_modes():
    a = build_prompt(COLORS, "A")
    b = build_prompt(COLORS, "B", direction="slight left", history="2.0 s ago: continued straight.")
    assert "Candidates: red, blue, yellow." in a and "trail" in a and "operator" not in a
    assert '"slight left"' in b and "drift a little to the left" in b and "2.0 s ago" in b
    assert '"<color>"' in a and "red" not in a.split("Answer with JSON")[1]  # placeholders, no real example
    with pytest.raises(ValueError):
        build_prompt(COLORS, "B")  # Mode B needs a direction
    with pytest.raises(ValueError):
        build_prompt(COLORS, "C")


def test_history_text_is_short_and_colorless():
    entries = [HistoryEntry(t, "continue", b) for t, b in ((1.0, 0.0), (2.0, 15.0), (3.0, -2.0), (4.0, None))]
    entries[-1].action = "stop"
    text = history_text(entries, now=5.0)
    assert text == "3.0 s ago: continued slight left; 2.0 s ago: continued straight; 1.0 s ago: stopped."
    assert history_text([], now=5.0) == "none."


def test_shuffled_is_repeatable_permutation():
    assert shuffled(COLORS, 7) == shuffled(COLORS, 7) and sorted(shuffled(COLORS, 7)) == sorted(COLORS)


# --- answer checker ---


def test_parse_a_good_answer_in_a_code_fence():
    reply = ('Here you go:\n```json\n{"safe_until": {"Red": "all", "blue path": "Tick 2", "yellow": "none"}, '
             '"choice": "Red", "action": "Continue", "reasoning": "clear trail"}\n```')
    answer = parse_answer(reply, COLORS)
    assert answer.valid and answer.choice == "red" and answer.action == "continue"
    assert answer.safe_until == {"red": "all", "blue": "tick 2", "yellow": "none"}


def test_parse_problems():
    assert parse_answer("no json here", COLORS).problems == ["no JSON object"]
    answer = parse_answer('{"choice": "green", "action": "maybe", "safe_until": {"red": "all"}}', COLORS)
    assert not answer.valid and answer.choice is None and answer.action is None
    assert len(answer.problems) == 3  # unknown color, unknown action, missing ticks for blue and yellow
    both = parse_answer('{"choice": "red or blue", "action": "stop", "safe_until": {}}', COLORS)
    assert both.choice is None  # two colors named: not a choice


@pytest.mark.parametrize("value, tick", [
    ("all", "all"), ("ALL", "all"), ("tick2", "tick 2"), ("10 m", "tick 2"), (2, "tick 2"), ("tick 1", "tick 1"),
    ("5m", "tick 1"), ("none", "none"), (0, "none"), ("maybe", None), (None, None), (True, None),
])
def test_normalize_tick(value, tick):
    assert normalize_tick(value) == tick


def test_chosen_bearing():
    answer = parse_answer('{"choice": "blue", "action": "continue", "safe_until": {}}', COLORS)
    assert chosen_bearing(answer, COLORS, [10.0, -3.0, 5.0]) == -3.0
    answer.action = "stop"
    assert chosen_bearing(answer, COLORS, [10.0, -3.0, 5.0]) is None


# --- scoring ---


def test_score_answer():
    labs = [labels("all", match=True, puddle=8.0), labels("tick 1"), labels("all")]
    reply = '{"safe_until": {"red": "tick 2", "blue": "all", "yellow": "all"}, "choice": "red", "action": "continue"}'
    score = score_answer(parse_answer(reply, COLORS), COLORS, labs, closest_index=0)
    assert score.valid and score.choice_matches_driver and score.choice_is_closest
    assert score.choice_safe and score.chose_puddle and score.puddle_frame  # yellow was dry and safe
    assert score.ticks_answered == 3 and score.ticks_exact == 1  # only yellow right
    assert score.ticks_over == 1 and score.ticks_under == 1  # blue too optimistic, red too cautious
    assert score.ticks_within_one == 2 and score.action_correct and not score.stop_label


def test_score_stop_and_invalid_choice():
    labs = [labels("none"), labels("none")]
    reply = '{"safe_until": {"red": "none", "blue": "none"}, "choice": "purple", "action": "stop"}'
    score = score_answer(parse_answer(reply, ["red", "blue"]), ["red", "blue"], labs, closest_index=1)
    assert score.stop_label and score.action_correct
    assert score.choice_matches_driver is None and score.choice_safe is None and not score.puddle_frame
    assert np.isclose(score.ticks_exact, 2)
