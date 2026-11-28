"""Tests for the step 6.5 run plan and scoring (no MAVS needed)."""

from offroad.sim.evaluation import (SCENE_ROUTES, WORD_BLIND, chooser_summary, decision_stats, fork_success,
                                    is_success, planned_runs, run_row)


def test_planned_runs():
    runs = planned_runs()
    assert len(runs) == 3 * 6 + 2 * 2 * 6  # baselines in Mode A only; the VLMs in both modes
    assert not [r for r in runs if r.chooser in WORD_BLIND and r.mode == "B"]
    assert [r.chooser for r in runs][:18] == ["straightest"] * 6 + ["cheapest"] * 6 + ["oracle"] * 6  # fast ones first
    vlm = [r.chooser for r in runs if r.chooser not in WORD_BLIND]
    assert vlm == ["zeroshot"] * 12 + ["finetuned"] * 12  # each VLM's runs together: loaded once
    assert {(r.scene, r.route) for r in runs} == set(SCENE_ROUTES)
    assert len(planned_runs(["finetuned"], ["B"], ["fork"])) == 2


def test_success_rules():
    assert is_success("curves", {"outcome": "finished"}) and not is_success("curves", {"outcome": "stopped"})
    assert is_success("blocked", {"outcome": "stopped"}) and not is_success("blocked", {"outcome": "collision"})
    assert not is_success("blocked", {"outcome": "finished"})
    assert fork_success({"outcome": "off_trail"}, {"outcome": "finished"})
    assert not fork_success({"outcome": "off_trail"}, None)


def decision(chosen, bearings, answer=None, n=None):
    d = {"chosen": chosen, "end_bearings_deg": bearings, "n_candidates": len(bearings) if n is None else n}
    if answer is not None:
        d["answer"] = answer
    return d


def test_decision_stats():
    a_ok = {"valid": True, "action": "continue", "safe_until": {"red": "all", "blue": "all"}, "seconds": 3.0}
    a_stop = {"valid": True, "action": "stop", "safe_until": {"red": "none"}, "seconds": 5.0}
    stats = decision_stats([decision(1, [10.0, 1.0], a_ok), decision(0, [10.0, 1.0], a_ok),
                            decision(None, [3.0], a_stop), decision(None, [], n=0)])
    assert stats["decisions"] == 4 and stats["asked"] == 3 and stats["no_candidates"] == 1
    assert stats["chose_straightest"] == round(1 / 3, 3)
    assert stats["said_stop"] == 1 and stats["all_ticks_all"] == round(2 / 3, 3) and stats["seconds"] == round(11 / 3, 2)
    assert "valid" not in decision_stats([decision(0, [1.0])])  # a baseline has no answers


def row(scene, route, outcome, mode="A", completion=1.0):
    return run_row(scene, route, "finetuned", mode, {"outcome": outcome, "completion": completion,
                                                     "mean_offset_m": 0.5, "min_gap_m": 1.0}, [])


def test_chooser_summary_counts_the_fork_once_in_mode_a():
    rows = [row("trail01", None, "finished"), row("curves", None, "off_trail", completion=0.5),
            row("fork", "left", "off_trail", completion=0.4), row("fork", "right", "finished"),
            row("blocked", None, "stopped", completion=0.3), row("halfblocked", None, "collision", completion=0.3)]
    s = chooser_summary(rows)
    assert s["tests"] == 5 and s["successes"] == 3  # trail01, fork (either branch), blocked
    assert s["collisions"] == 1 and s["off_trail"] == 2 and s["stopped_wrongly"] == 0
    rows_b = [row("fork", "left", "off_trail", mode="B"), row("fork", "right", "finished", mode="B")]
    assert chooser_summary(rows_b)["tests"] == 2 and chooser_summary(rows_b)["successes"] == 1
