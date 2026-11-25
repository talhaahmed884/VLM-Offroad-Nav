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

Error measures for the report (added 2026-11-24): 95% confidence intervals (Wilson for shares,
normal approximation for means), distance errors (mean with interval, median, 90th percentile,
and the mean over 5 / 10 / 15 m like the average displacement error of trajectory papers), the
side of wrong choices, the chosen position, hazard flags from the ticks as precision / recall /
F1, answer speed, and paired comparisons of two choosers on the same frames (exact McNemar test).
"""

import math

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
    offsets = [[offset_from_driven(c["xy"], frame["driven"]["xy"], d) for d in TICKS_M] for c in cands]
    closest = frame["closest_to_driver"]
    wrong = [not c["matches_driver"] for c in cands]
    any_match = not all(wrong)
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
    for i, d in enumerate(TICKS_M):
        numbers[f"offset_{d:g}m"] = mean([o[i] for o in offsets])
    # Mean over 5 / 10 / 15 m (only paths that reach all three distances).
    numbers["ade_m"] = mean([None if None in o else float(np.mean(o)) for o in offsets])
    # Position among 2+ candidates (left to right), and the side of a wrong choice relative to the
    # driver's closest candidate (only where some candidate matches the driver).
    n = len(cands)
    for name, k in (("pos_left", 0), ("pos_right", n - 1)):
        numbers[name] = mean([j == k for j in range(n)]) if n >= 2 else None
    numbers["pos_middle"] = mean([0 < j < n - 1 for j in range(n)]) if n >= 2 else None
    wrong_weight = float(sum(w[j] for j in range(n) if wrong[j]))
    numbers["wrong_weight"] = wrong_weight if any_match else None
    numbers["wrong_left"] = float(sum(w[j] for j in range(n) if wrong[j] and j < closest)) if any_match else None
    numbers["wrong_right"] = float(sum(w[j] for j in range(n) if wrong[j] and j > closest)) if any_match else None
    return numbers


# --- statistics ---


Z_95 = 1.96


def wilson_interval(successes: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% confidence interval of a share (Wilson score); successes may be fractional (random baseline)."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mean_interval(values: list[float], z: float = Z_95) -> tuple[float, float, float]:
    """(mean, low, high): the mean with its 95% confidence interval (normal approximation)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    half = z * float(v.std(ddof=1)) / math.sqrt(v.size) if v.size > 1 else 0.0
    return float(v.mean()), float(v.mean()) - half, float(v.mean()) + half


def mcnemar_p(only_first: int, only_second: int) -> float:
    """Two-sided exact McNemar p-value: are two choosers equally often right on the same frames?

    only_first / only_second: frames where only the first / only the second chooser is right.
    """
    n = only_first + only_second
    if n == 0:
        return 1.0
    k = min(only_first, only_second)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def binary_scores(truth: list[bool], predicted: list[bool]) -> dict[str, float | int | None]:
    """Precision, recall and F1 of a yes/no detector (None where undefined), with the counts."""
    t, p = np.asarray(truth, dtype=bool), np.asarray(predicted, dtype=bool)
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall) if precision and recall else
          (0.0 if precision is not None and recall is not None else None))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def paired_comparison(frames: list[dict], first: list[np.ndarray], second: list[np.ndarray]) -> dict[str, str]:
    """Two one-hot choosers on the same frames: where only one is right (driver match), and the McNemar p.

    Frames where either has no choice are left out. Not for the random baseline (not one-hot).
    """
    only_first = only_second = both = neither = 0
    for f, a, b in zip(frames, first, second):
        if a.sum() == 0 or b.sum() == 0:
            continue
        ra = f["candidates"][int(np.argmax(a))]["matches_driver"]
        rb = f["candidates"][int(np.argmax(b))]["matches_driver"]
        only_first += ra and not rb
        only_second += rb and not ra
        both += ra and rb
        neither += not ra and not rb
    return {
        "frames": str(only_first + only_second + both + neither),
        "both right / both wrong": f"{both} / {neither}",
        "only the first right": str(only_first),
        "only the second right": str(only_second),
        "p-value (exact McNemar)": f"{mcnemar_p(only_first, only_second):.3f}",
    }


# --- summaries over frames ---


def _pct(values: list[float], digits: int = 0) -> str:
    return f"{100 * np.mean(values):.{digits}f}%" if values else "-"


def _num(values: list[float], unit: str = "", digits: int = 2) -> str:
    return f"{np.mean(values):.{digits}f}{unit}" if values else "-"


def _pct_ci(values: list[float]) -> str:
    """Share with its 95% interval, e.g. '66% [61, 71]'."""
    if not values:
        return "-"
    lo, hi = wilson_interval(float(np.sum(values)), len(values))
    return f"{100 * np.mean(values):.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"


def _num_ci(values: list[float], unit: str = " m") -> str:
    """Mean with its 95% interval, e.g. '1.43 m [1.29, 1.58]'."""
    if not values:
        return "-"
    m, lo, hi = mean_interval(values)
    return f"{m:.2f}{unit} [{lo:.2f}, {hi:.2f}]"


def _score(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _ratio(part: list[float], whole: list[float]) -> str:
    total = float(np.sum(whole))
    return f"{100 * float(np.sum(part)) / total:.0f}%" if total > 0 else "-"


def summarize_choices(frames: list[dict], weights: list[np.ndarray]) -> dict[str, str]:
    """One table column (metric -> text) for a chooser over the frames (frames with candidates only)."""
    rows = [(f, frame_numbers(f, w)) for f, w in zip(frames, weights)]
    scored = [(f, n) for f, n in rows if n is not None]

    def col(key: str, subset=lambda f: True) -> list[float]:
        return [n[key] for f, n in scored if subset(f) and n[key] is not None]

    real = lambda f: f["real_choice"]  # noqa: E731
    turn = lambda f: f["direction"] not in (None, "straight")  # noqa: E731
    straight = lambda f: not turn(f)  # noqa: E731
    far = col(f"offset_{TICKS_M[-1]:g}m")
    wrong = col("wrong_weight")
    return {
        "frames with a choice": f"{len(scored)}/{len(rows)}",
        "choice matches the driver [95% interval]": _pct_ci(col("matches_driver")),
        "... in frames with a real choice": _pct(col("matches_driver", real)),
        "... in straight frames / turn frames (driver's word)":
            f"{_pct_ci(col('matches_driver', straight))} / {_pct_ci(col('matches_driver', turn))}",
        "choice is the driver's closest candidate": _pct(col("is_closest")),
        "wrong choices: left / right of the driver's candidate":
            f"{_ratio(col('wrong_left'), wrong)} / {_ratio(col('wrong_right'), wrong)}",
        "chosen position (2+ candidates): leftmost / middle / rightmost":
            " / ".join(_pct(col(k)) for k in ("pos_left", "pos_middle", "pos_right")),
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
        f"distance at {TICKS_M[-1]:g} m: mean [95% interval]": _num_ci(far),
        f"distance at {TICKS_M[-1]:g} m: median / 90th percentile":
            f"{np.median(far):.2f} m / {np.percentile(far, 90):.2f} m" if far else "-",
        "mean distance over 5 / 10 / 15 m (ADE) [95% interval]": _num_ci(col("ade_m")),
    }


def summarize_ticks(frames: list[dict], said: list[dict[int, str]]) -> dict[str, str]:
    """Tick answers vs labels. said[i]: {candidate index: tick answer} for frame i (missing = not answered)."""
    pairs = [(SAFE_ANSWERS.index(c["safe_until"]), SAFE_ANSWERS.index(s[k]), c["first_unsafe_m"] is not None)
             for f, s in zip(frames, said) for k, c in enumerate(f["candidates"]) if k in s]
    n_candidates = sum(len(f["candidates"]) for f in frames)
    if not pairs:
        return {"ticks answered": f"0/{n_candidates}"}
    truth, answer, unsafe = (np.array(x) for x in zip(*pairs))
    full = SAFE_ANSWERS.index("all")
    hazard = binary_scores(list(truth < full), list(answer < full))
    out = {
        "ticks answered": f"{len(pairs)}/{n_candidates}",
        "exact / within one tick": f"{_pct(list(answer == truth))} / {_pct(list(np.abs(answer - truth) <= 1))}",
        "mean tick error (ticks)": f"{np.mean(np.abs(answer - truth)):.3f}",
        "too optimistic / too cautious": f"{_pct(list(answer > truth))} / {_pct(list(answer < truth))}",
        "unsafe stretch called safe past it":
            f"{int((answer[unsafe] > truth[unsafe]).sum())}/{int(unsafe.sum())}" if unsafe.any() else "- (none)",
        'hazard flags ("not safe to the end"): labelled / flagged / both':
            f"{hazard['tp'] + hazard['fn']} / {hazard['tp'] + hazard['fp']} / {hazard['tp']}",
        "hazard flags: precision / recall / F1": " / ".join(_score(hazard[k]) for k in ("precision", "recall", "f1")),
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
    stop = binary_scores(label, said)
    return {
        "action correct": _pct([lab == s for lab, s in pairs]),
        "stop frames (label) / said stop / both": f"{sum(label)} / {sum(said)} / {hits}",
        "stop recall / precision": (f"{hits}/{sum(label)}" if sum(label) else "-") + " / " +
                                   (f"{hits}/{sum(said)}" if sum(said) else "-"),
        "stop F1": _score(stop["f1"]),
    }


def summarize_speed(answers: list[dict]) -> dict[str, str]:
    """Seconds per answer (mean / median / 95th percentile) and answer length, over asked frames."""
    seconds = [a["seconds"] for a in answers if a and a.get("asked") and a.get("seconds") is not None]
    tokens = [a["new_tokens"] for a in answers if a and a.get("asked") and a.get("new_tokens") is not None]
    if not seconds:
        return {"answers timed": "0"}
    return {
        "answers timed": str(len(seconds)),
        "seconds per answer: mean / median / 95th percentile":
            f"{np.mean(seconds):.2f} / {np.median(seconds):.2f} / {np.percentile(seconds, 95):.2f}",
        "answer length (tokens, mean)": f"{np.mean(tokens):.0f}" if tokens else "-",
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
