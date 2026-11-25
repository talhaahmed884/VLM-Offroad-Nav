"""Tests for the phase 4 evaluation, the per-path terrain numbers, and the wrong history. No dataset or model needed."""

import numpy as np
import pytest

from offroad.mapping.elevation import GridSpec
from offroad.nav.candidate_metrics import NEAR_MISS_M, candidate_metrics
from offroad.nav.evaluation import (
    answer_weights,
    binary_scores,
    choose_cheapest,
    choose_driver,
    choose_random,
    choose_straightest,
    choose_toward_word,
    direction_test,
    frame_numbers,
    history_test,
    mcnemar_p,
    mean_interval,
    offset_from_driven,
    one_hot,
    paired_comparison,
    summarize_actions,
    summarize_choices,
    summarize_speed,
    summarize_ticks,
    wilson_interval,
)
from offroad.nav.labels import cells_along
from offroad.nav.prompts import WRONG_TURN_DEG, HistoryEntry, wrong_history

SPEC = GridSpec(x_min=-5.0, x_max=25.0, y_min=-12.0, y_max=12.0, resolution=0.25)


def line(y_end: float, length: float = 15.0) -> list[list[float]]:
    xs = np.arange(0.0, length + 1e-9, 0.25)
    return np.column_stack([xs, xs / length * y_end]).tolist()


def candidate(bearing, cost=1.0, match=False, safe="all", puddle=None, unsafe=None, water=0.0, clearance=None):
    return {"end_bearing_deg": bearing, "cost_per_m": cost, "matches_driver": match, "safe_until": safe,
            "first_caution_m": puddle, "first_unsafe_m": unsafe, "xy": line(15.0 * np.tan(np.radians(bearing))),
            "metrics": {"share_water": water, "share_puddle": 0.0, "share_mud": 0.0, "share_grass_bush": 1.0,
                        "share_tree_log": 0.0, "share_unknown": 0.0, "clearance_m": clearance,
                        "slope_cm_per_m": 2.0, "bump_height_m": 0.05}}


def frame(cands, closest, direction="straight", stop=False, real=True, key=("00004", 1)):
    return {"sequence": key[0], "frame": key[1], "candidates": cands, "closest_to_driver": closest,
            "direction": direction, "stop": stop, "real_choice": real, "driven": {"xy": line(0.0, 20.0)}}


def three(**kw):
    """Left (20°, cheap, puddle), straight (0°, the driver's), right (-20°, water, close obstacle)."""
    return frame([candidate(20.0, cost=0.5, puddle=6.0),
                  candidate(0.0, cost=1.0, match=True),
                  candidate(-20.0, cost=2.0, water=0.5, clearance=0.4, safe="tick 1", unsafe=7.0)], closest=1, **kw)


def answer(choice, colors=("red", "blue", "yellow"), action="continue", safe=None, **kw):
    return {"asked": True, "choice": choice, "colors": list(colors), "action": action,
            "safe_until": safe or {}, **kw}


# --- choosers ---


def test_baseline_choosers():
    f = three()
    assert np.allclose(choose_random(f), 1 / 3)
    assert np.argmax(choose_straightest(f)) == 1 and np.argmax(choose_cheapest(f)) == 0
    assert np.argmax(choose_driver(f)) == 1
    assert np.argmax(choose_toward_word(f, "slight left")) == 0 and np.argmax(choose_toward_word(f, "bear right")) == 2


def test_answer_weights():
    f = three()
    assert np.argmax(answer_weights(f, answer("yellow"))) == 2
    assert answer_weights(f, answer("green")).sum() == 0 and answer_weights(f, None).sum() == 0
    with pytest.raises(ValueError):
        answer_weights(f, answer("red", colors=("red", "blue")))  # candidates changed since the run


# --- numbers ---


def test_offset_from_driven():
    straight = line(0.0)
    assert offset_from_driven(straight, line(0.0, 20.0), 10.0) == pytest.approx(0.0)
    assert offset_from_driven(line(3.0), line(0.0, 20.0), 15.0) == pytest.approx(2.956, abs=0.01)  # 15 m along a slanted line
    assert offset_from_driven(line(0.0, 8.0), line(0.0, 20.0), 10.0) is None  # candidate too short
    assert offset_from_driven([[0.0, 0.0], [14.98, 0.0]], line(0.0, 20.0), 15.0) == pytest.approx(0.02, abs=0.01)  # end counts


def test_frame_numbers_weighted():
    f = three()
    one = frame_numbers(f, choose_driver(f))
    assert one["matches_driver"] == 1.0 and one["is_closest"] == 1.0 and one["offset_10m"] == pytest.approx(0.0)
    rand = frame_numbers(f, choose_random(f))
    assert rand["matches_driver"] == pytest.approx(1 / 3) and rand["share_water"] == pytest.approx(0.5 / 3)
    assert rand["near_miss"] == pytest.approx(1 / 3) and rand["clearance_m"] == pytest.approx(0.4)  # None skipped
    assert frame_numbers(f, np.zeros(3)) is None


def test_summaries():
    frames = [three(direction="bear left"), three()]
    s = summarize_choices(frames, [choose_driver(f) for f in frames])
    assert s["choice matches the driver [95% interval]"].startswith("100% [") and s["frames with a choice"] == "2/2"
    s = summarize_choices(frames, [choose_cheapest(frames[0]), np.zeros(3)])
    assert s["frames with a choice"] == "1/2" and s["chose a puddle when a dry path existed"] == "100%"

    ticks = summarize_ticks(frames, [{0: "all", 1: "all", 2: "all"}, {2: "none"}])
    assert ticks["ticks answered"] == "4/6" and ticks["unsafe stretch called safe past it"] == "1/2"
    assert ticks['label "tick 1": answered exactly'] == "0/2" and ticks['label "all": answered exactly'] == "2/2"

    actions = summarize_actions([three(stop=True), three()], ["stop", "stop"])
    assert actions["stop frames (label) / said stop / both"] == "1 / 2 / 1"
    assert actions["stop recall / precision"] == "1/1 / 1/2" and actions["action correct"] == "50%"
    assert actions["stop F1"] == "0.67"


# --- error measures for the report ---

def test_intervals_and_mcnemar():
    lo, hi = wilson_interval(218, 328)  # the reported model's test accuracy, 66.5%
    assert (round(lo, 3), round(hi, 3)) == (0.612, 0.714)
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    m, lo, hi = mean_interval([1.0, 2.0, 3.0])
    assert m == 2.0 and hi - m == pytest.approx(1.96 / np.sqrt(3))
    assert mcnemar_p(0, 0) == 1.0 and mcnemar_p(5, 5) == 1.0
    assert mcnemar_p(10, 0) == pytest.approx(2 / 2 ** 10)  # all 10 disagreements one way
    assert mcnemar_p(110, 35) < 1e-9 and mcnemar_p(50, 51) > 0.9


def test_binary_scores():
    s = binary_scores([True, True, False, False], [True, False, True, False])
    assert (s["tp"], s["fp"], s["fn"]) == (1, 1, 1) and s["precision"] == s["recall"] == s["f1"] == 0.5
    none_flagged = binary_scores([True, False], [False, False])
    assert none_flagged["precision"] is None and none_flagged["recall"] == 0.0 and none_flagged["f1"] is None
    assert binary_scores([False], [False])["recall"] is None


def test_new_choice_rows():
    f = three()  # driver's candidate is the middle one (index 1)
    left = frame_numbers(f, one_hot(3, 0))
    assert left["wrong_left"] == 1.0 and left["wrong_right"] == 0.0 and left["pos_left"] == 1.0
    assert left["ade_m"] == pytest.approx(np.mean([left[f"offset_{d}m"] for d in (5, 10, 15)]))
    frames = [three(), three(direction="slight right"), three()]
    s = summarize_choices(frames, [one_hot(3, 1), one_hot(3, 2), one_hot(3, 0)])
    assert s["wrong choices: left / right of the driver's candidate"] == "50% / 50%"
    assert s["chosen position (2+ candidates): leftmost / middle / rightmost"] == "33% / 33% / 33%"
    assert s["... in straight frames / turn frames (driver's word)"].startswith("50% [")
    assert s["distance at 15 m: median / 90th percentile"].endswith(" m")

    ticks = summarize_ticks([three()], [{0: "all", 1: "tick 2", 2: "all"}])
    assert ticks['hazard flags ("not safe to the end"): labelled / flagged / both'] == "1 / 1 / 0"
    assert ticks["hazard flags: precision / recall / F1"] == "0.00 / 0.00 / 0.00"
    assert ticks["mean tick error (ticks)"] == f"{(1 + 2) / 3:.3f}"


def test_paired_comparison_and_speed():
    frames = [three(), three(), three()]
    driver = [choose_driver(f) for f in frames]
    straightest = [one_hot(3, 1), one_hot(3, 0), one_hot(3, 0)]
    p = paired_comparison(frames, driver, straightest)
    assert p["only the first right"] == "2" and p["only the second right"] == "0" and p["both right / both wrong"] == "1 / 0"
    assert p["p-value (exact McNemar)"] == "0.500"
    speed = summarize_speed([{"asked": True, "seconds": s, "new_tokens": 40} for s in (1.0, 2.0, 3.0)] + [{"asked": False}])
    assert speed["answers timed"] == "3" and speed["seconds per answer: mean / median / 95th percentile"].startswith("2.00 / 2.00")


def test_direction_and_history_tests():
    frames = [three(), three(), frame([candidate(0.0, match=True)], closest=0)]
    left = [answer("red"), answer("blue"), answer("red", colors=("red",))]
    right = [answer("yellow"), answer("blue"), answer("red", colors=("red",))]
    d = direction_test(frames, left, right, "bear left", "bear right")
    assert d["frames (2+ candidates, both answered)"] == "2"  # the one-candidate frame is left out
    assert d['choice further left with "bear left" (followed)'] == "50%"
    assert d["same candidate both times (ignored)"] == "50%"

    own = [answer("blue"), answer("blue"), None]
    wrong = [answer("red", history_injected="left"), answer("red", history_injected="right"), None]
    h = history_test(frames, own, wrong)
    assert h["frames (2+ candidates, both answered)"] == "2"
    assert h["moved toward the made-up turn"] == "50%" and h["moved away from it"] == "50%"


# --- terrain numbers ---


def test_candidate_metrics():
    xy = np.array(line(0.0, 10.0))
    cells, _ = cells_along(xy, SPEC)
    classes = np.array([1] * 20 + [6] * 10 + [0] * 11)  # 20 dirt, 10 water, 11 unknown points
    ids = {"water": [6], "puddle": [31], "mud": [33], "grass_bush": [3, 19], "tree_log": [4]}
    obstacles = np.array([[5.0, 0.5], [9.0, 3.0]])
    ground = np.full(SPEC.shape, -1.2)
    m = candidate_metrics(classes, xy, cells, obstacles, ground, np.zeros(SPEC.shape), SPEC, ids)
    assert m.share_water == pytest.approx(10 / 41) and m.share_unknown == pytest.approx(11 / 41)
    assert m.clearance_m == pytest.approx(0.5) and m.near_miss == (0.5 < NEAR_MISS_M)
    assert m.slope_cm_per_m == pytest.approx(0.0)
    none = candidate_metrics(classes, xy, cells, np.zeros((0, 2)), ground, np.zeros(SPEC.shape), SPEC, ids)
    assert none.clearance_m is None and not none.near_miss


# --- wrong history ---


def test_wrong_history():
    entries = [HistoryEntry(1.0, "continue", 0.0), HistoryEntry(2.0, "continue", 20.0)]
    fake, side = wrong_history(entries, seed=0)
    assert side == "right" and fake[-1].bearing_deg == -WRONG_TURN_DEG  # opposite to the real left turn
    assert fake[0] == entries[0] and entries[-1].bearing_deg == 20.0  # the input is not changed
    _, side = wrong_history([HistoryEntry(1.0, "continue", 2.0)], seed=3)
    assert side in ("left", "right") and wrong_history([HistoryEntry(1.0, "stop", None)], 3)[1] == side
    assert wrong_history([], seed=0) == ([], None)
