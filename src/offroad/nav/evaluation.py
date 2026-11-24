"""Phase 4 evaluation: score selector answers and simple baselines on the same frames (pure, no file I/O).

Inputs are the JSON records the scripts write: one *frame* record per decision frame from
scripts/build_candidates.py (candidates left to right, with labels and terrain numbers, and the
driven path), and one *answer* record per frame from scripts/run_selector.py.

A choice is a weight per candidate: one-hot for a model's answer or a baseline, uniform for the
random baseline (so "random" is the exact average over all candidates, not one lucky draw), all
zeros for "no valid choice". Every metric of a choice is the weighted mean over candidates.

Baselines (docs/approach-2-plan.md, phase 4; the VLM must beat them to show it reads the image):
- random; the straightest candidate; the cheapest (lowest mean cost: the geometric planner's own
  pick, roughly what approach 1's planner would drive);
- Mode B: "toward the word": the candidate whose end bearing is closest to the word's typical
  bearing, without looking at the image;
- the driver (oracle): the candidate closest to the driven path, the best a chooser can do;
- ticks: "always all"; action: "always continue".
"""

import numpy as np

from offroad.nav.candidate_metrics import NEAR_MISS_M
from offroad.nav.candidates import FULL_LENGTH_TOLERANCE_M
from offroad.nav.driven_path import point_at_distance
from offroad.nav.labels import SAFE_ANSWERS, TICKS_M

# Typical end bearing for each Mode B word (+ = left): the middle of its range (straight < 10°,
# slight 10-25°), and 35° for "bear", beyond the ±30° fan.
WORD_BEARING_DEG = {"bear left": 35.0, "slight left": 17.5, "straight": 0.0, "slight right": -17.5, "bear right": -35.0}
TERRAIN = ("water", "puddle", "mud", "grass_bush", "tree_log", "unknown")


# --- choices as weights over the candidates ---


def one_hot(n: int, k: int | None) -> np.ndarray:
    w = np.zeros(n)
    if k is not None:
        w[k] = 1.0
    return w


def choose_random(frame: dict, word: str | None = None) -> np.ndarray:
    n = len(frame["candidates"])
    return np.full(n, 1.0 / n) if n else np.zeros(0)


def choose_straightest(frame: dict, word: str | None = None) -> np.ndarray:
    bearings = [abs(c["end_bearing_deg"]) for c in frame["candidates"]]
    return one_hot(len(bearings), int(np.argmin(bearings)) if bearings else None)


def choose_cheapest(frame: dict, word: str | None = None) -> np.ndarray:
    costs = [c["cost_per_m"] for c in frame["candidates"]]
    return one_hot(len(costs), int(np.argmin(costs)) if costs else None)


def choose_toward_word(frame: dict, word: str | None) -> np.ndarray:
    """The candidate whose end bearing is closest to the word's typical bearing (straight if no word)."""
    target = WORD_BEARING_DEG.get(word, 0.0)
    gaps = [abs(c["end_bearing_deg"] - target) for c in frame["candidates"]]
    return one_hot(len(gaps), int(np.argmin(gaps)) if gaps else None)


def choose_driver(frame: dict, word: str | None = None) -> np.ndarray:
    return one_hot(len(frame["candidates"]), frame["closest_to_driver"])


BASELINES = {
    "A": {"random": choose_random, "straightest": choose_straightest, "cheapest": choose_cheapest,
          "driver (oracle)": choose_driver},
    "B": {"random": choose_random, "straightest": choose_straightest, "cheapest": choose_cheapest,
          "toward the word": choose_toward_word, "driver (oracle)": choose_driver},
}


def driven_as_frame(frame: dict) -> dict:
    """The frame with the driven path as its only candidate (a reference column: what the driver had).

    The driven path's first 15 m are labelled like a candidate (build_candidates.py), so the same
    numbers apply; it is not one of the choices the model sees.
    """
    driven = frame["driven"]
    length = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(np.asarray(driven["xy"]), axis=0), axis=1)])
    path = {**driven, "xy": np.asarray(driven["xy"])[length <= TICKS_M[-1] + 1e-6].tolist(),
            "end_bearing_deg": 0.0, "cost_per_m": float("nan")}
    return {**frame, "candidates": [path], "closest_to_driver": 0}


def answer_weights(frame: dict, answer: dict | None) -> np.ndarray:
    """The model's choice as weights (zeros if it wasn't asked or named no candidate)."""
    n = len(frame["candidates"])
    if not answer or not answer.get("asked") or answer.get("choice") not in answer.get("colors", []):
        return np.zeros(n)
    if len(answer["colors"]) != n:
        raise ValueError(f"{frame['sequence']}/{frame['frame']}: the answer has {len(answer['colors'])} colors "
                         f"but the frame {n} candidates; rebuild the candidates or rerun the selector")
    return one_hot(n, answer["colors"].index(answer["choice"]))


# --- numbers for one frame ---


def offset_from_driven(candidate_xy, driven_xy, distance: float) -> float | None:
    """Distance (m) between the points `distance` m along the candidate and along the driven path.

    A full-length candidate (ends within FULL_LENGTH_TOLERANCE_M of 15 m) uses its end for 15 m.
    None if either path is shorter.
    """
    xy = np.asarray(candidate_xy, dtype=float)
    length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
    if distance - FULL_LENGTH_TOLERANCE_M <= length < distance:
        distance_on_candidate = length
    else:
        distance_on_candidate = distance
    a = point_at_distance(xy, distance_on_candidate)
    b = point_at_distance(np.asarray(driven_xy, dtype=float), distance)
    return None if a is None or b is None else float(np.linalg.norm(a - b))


def puddle_frame(frame: dict) -> bool:
    """Some candidate crosses a puddle and some candidate is dry and safe to tick 1 (as in nav.scoring)."""
    cands = frame["candidates"]
    return (any(c["first_caution_m"] is not None for c in cands)
            and any(c["first_caution_m"] is None and c["safe_until"] != "none" for c in cands))


def frame_numbers(frame: dict, weights: np.ndarray) -> dict[str, float | None] | None:
    """Weighted numbers of the chosen candidate(s); None if there is no choice (all weights zero)."""
    total = float(np.sum(weights))
    if total == 0:
        return None
    w = np.asarray(weights, dtype=float) / total
    cands = frame["candidates"]

    def mean(values) -> float | None:
        v = np.array([np.nan if x is None else float(x) for x in values])
        ok = ~np.isnan(v) & (w > 0)
        return float((w[ok] * v[ok]).sum() / w[ok].sum()) if ok.any() else None

    metrics = [c.get("metrics") or {} for c in cands]
    numbers = {
        "matches_driver": mean([c["matches_driver"] for c in cands]),
        "is_closest": mean([k == frame["closest_to_driver"] for k in range(len(cands))]),
        "safe_tick1": mean([c["safe_until"] != "none" for c in cands]),
        "safe_all": mean([c["safe_until"] == "all" for c in cands]),
        "puddle": mean([c["first_caution_m"] is not None for c in cands]),
        "clearance_m": mean([m.get("clearance_m") for m in metrics]),
        "near_miss": mean([m.get("clearance_m") is not None and m["clearance_m"] < NEAR_MISS_M for m in metrics]),
        "slope_cm_per_m": mean([m.get("slope_cm_per_m") for m in metrics]),
    }
    for group in TERRAIN:
        numbers[f"share_{group}"] = mean([m.get(f"share_{group}") for m in metrics])
    for d in TICKS_M:
        numbers[f"offset_{d:g}m"] = mean([offset_from_driven(c["xy"], frame["driven"]["xy"], d) for c in cands])
    return numbers


# --- summaries over frames ---


def _pct(values: list[float], digits: int = 0) -> str:
    return f"{100 * np.mean(values):.{digits}f}%" if values else "-"


def _num(values: list[float], unit: str = "", digits: int = 2) -> str:
    return f"{np.mean(values):.{digits}f}{unit}" if values else "-"


def summarize_choices(frames: list[dict], weights: list[np.ndarray]) -> dict[str, str]:
    """One table column (metric -> text) for a chooser over the frames (frames with candidates only)."""
    rows = [(f, frame_numbers(f, w)) for f, w in zip(frames, weights)]
    scored = [(f, n) for f, n in rows if n is not None]

    def col(key: str, subset=lambda f: True) -> list[float]:
        return [n[key] for f, n in scored if subset(f) and n[key] is not None]

    real = lambda f: f["real_choice"]  # noqa: E731
    turn = lambda f: f["direction"] not in (None, "straight")  # noqa: E731
    return {
        "frames with a choice": f"{len(scored)}/{len(rows)}",
        "choice matches the driver": _pct(col("matches_driver")),
        "... in frames with a real choice": _pct(col("matches_driver", real)),
        "... in turn frames (driver's word not straight)": _pct(col("matches_driver", turn)),
        "choice is the driver's closest candidate": _pct(col("is_closest")),
        "chosen path safe to tick 1 / to the end": f"{_pct(col('safe_tick1'))} / {_pct(col('safe_all'))}",
        "chose a puddle when a dry path existed": _pct(col("puddle", puddle_frame)),
        "water / puddle / mud on the path (share of length)":
            " / ".join(_pct(col(f"share_{g}"), 1) for g in ("water", "puddle", "mud")),
        "grass-bush / tree-log / no ground truth":
            " / ".join(_pct(col(f"share_{g}"), 1) for g in ("grass_bush", "tree_log", "unknown")),
        "closest hard obstacle (mean)": _num(col("clearance_m"), " m"),
        f"near misses (hard obstacle < {NEAR_MISS_M:g} m)": _pct(col("near_miss"), 1),
        "ground slope (cm per m)": _num(col("slope_cm_per_m"), digits=1),
        "distance from the driven path at 5 / 10 / 15 m":
            " / ".join(_num(col(f"offset_{d:g}m"), " m") for d in TICKS_M),
    }


def summarize_ticks(frames: list[dict], said: list[dict[int, str]]) -> dict[str, str]:
    """Tick answers vs labels. said[i]: {candidate index: tick answer} for frame i (missing = not answered)."""
    pairs = [(SAFE_ANSWERS.index(c["safe_until"]), SAFE_ANSWERS.index(s[k]), c["first_unsafe_m"] is not None)
             for f, s in zip(frames, said) for k, c in enumerate(f["candidates"]) if k in s]
    n_candidates = sum(len(f["candidates"]) for f in frames)
    if not pairs:
        return {"ticks answered": f"0/{n_candidates}"}
    truth, answer, unsafe = (np.array(x) for x in zip(*pairs))
    out = {
        "ticks answered": f"{len(pairs)}/{n_candidates}",
        "exact / within one tick": f"{_pct(list(answer == truth))} / {_pct(list(np.abs(answer - truth) <= 1))}",
        "too optimistic / too cautious": f"{_pct(list(answer > truth))} / {_pct(list(answer < truth))}",
        "unsafe stretch called safe past it":
            f"{int((answer[unsafe] > truth[unsafe]).sum())}/{int(unsafe.sum())}" if unsafe.any() else "- (none)",
    }
    for k, name in enumerate(SAFE_ANSWERS):  # per answer type: how often each true label is answered exactly
        here = truth == k
        out[f'label "{name}": answered exactly'] = (f"{int((answer[here] == k).sum())}/{int(here.sum())}"
                                                    if here.any() else "- (none)")
    return out


def summarize_actions(frames: list[dict], actions: list[str | None]) -> dict[str, str]:
    """continue / stop answers vs the stop label (no candidate safe to tick 1)."""
    pairs = [(f["stop"], a == "stop") for f, a in zip(frames, actions) if a is not None]
    label = [p[0] for p in pairs]
    said = [p[1] for p in pairs]
    hits = sum(lab and s for lab, s in pairs)
    return {
        "action correct": _pct([lab == s for lab, s in pairs]),
        "stop frames (label) / said stop / both": f"{sum(label)} / {sum(said)} / {hits}",
        "stop recall / precision": (f"{hits}/{sum(label)}" if sum(label) else "-") + " / " +
                                   (f"{hits}/{sum(said)}" if sum(said) else "-"),
    }


# --- the two behaviour tests ---


def _chosen_bearing(frame: dict, answer: dict | None) -> float | None:
    w = answer_weights(frame, answer)
    return None if w.sum() == 0 else float(frame["candidates"][int(np.argmax(w))]["end_bearing_deg"])


def direction_test(frames: list[dict], left_answers: list[dict | None], right_answers: list[dict | None],
                   left_word: str, right_word: str) -> dict[str, str]:
    """Same frames asked with a more-left and a more-right word: does the choice move with the word?

    Only frames with 2+ candidates and a valid choice in both runs count. "followed" = the choice
    under left_word ends further left than under right_word; "same" = the same candidate.
    """
    moves = []
    edge_left, edge_right = [], []
    for f, a, b in zip(frames, left_answers, right_answers):
        if len(f["candidates"]) < 2:
            continue
        bl, br = _chosen_bearing(f, a), _chosen_bearing(f, b)
        if bl is None or br is None:
            continue
        moves.append(np.sign(bl - br))
        bearings = [c["end_bearing_deg"] for c in f["candidates"]]
        edge_left.append(bl == max(bearings))
        edge_right.append(br == min(bearings))
    moves = np.array(moves)
    return {
        "frames (2+ candidates, both answered)": str(len(moves)),
        f'choice further left with "{left_word}" (followed)': _pct(list(moves > 0)),
        "same candidate both times (ignored)": _pct(list(moves == 0)),
        f'choice further left with "{right_word}" (reversed)': _pct(list(moves < 0)),
        f'leftmost with "{left_word}" / rightmost with "{right_word}"': f"{_pct(edge_left)} / {_pct(edge_right)}",
    }


def history_test(frames: list[dict], own_answers: list[dict | None], wrong_answers: list[dict | None]) -> dict[str, str]:
    """A wrong latest history entry (a sharp turn not taken): does the choice move toward it?

    Compares each frame's choice with a wrong history against the same frame with the model's own
    history. Only frames with 2+ candidates, a made-up entry, and a valid choice in both runs count.
    """
    toward, away, same = 0, 0, 0
    for f, own, wrong in zip(frames, own_answers, wrong_answers):
        if len(f["candidates"]) < 2 or not wrong or not wrong.get("history_injected"):
            continue
        b_own, b_wrong = _chosen_bearing(f, own), _chosen_bearing(f, wrong)
        if b_own is None or b_wrong is None:
            continue
        side = 1.0 if wrong["history_injected"] == "left" else -1.0
        move = np.sign(b_wrong - b_own) * side
        toward += move > 0
        away += move < 0
        same += move == 0
    n = toward + away + same
    return {
        "frames (2+ candidates, both answered)": str(n),
        "moved toward the made-up turn": _pct([1] * toward + [0] * (n - toward)),
        "moved away from it": _pct([1] * away + [0] * (n - away)),
        "same candidate": _pct([1] * same + [0] * (n - same)),
    }
