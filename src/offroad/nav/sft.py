"""Phase 5: turn labelled decision frames into supervised fine-tuning examples (pure, no file I/O).

One example = the picture (drawn by the caller with `draw_visual_prompt`), the prompt, and the
answer the model should give (the target). The target comes from the labels, which exist only
for training and scoring; the model's input is the same as at run time (docs/approach-2-plan.md,
section 8). Safeguards against overfitting, all from the plan's phase 5:

- **Random colors, listing order and wording** for every example, so nothing can be learned
  from "red" or "the first one listed" or one exact sentence.
- **Balancing:** each frame appears `frame_weight` times per epoch on average: easy straight
  frames less, turns, puddle frames, frames with an unsafe candidate, and stop frames more.
- **Mirroring:** half the examples are flipped left-right (the picture, the direction word and
  the history), which balances left and right turns.
- **History:** from the driver's own earlier decisions (the closest candidate's end bearing, as
  at run time the chosen candidate's), left out in HISTORY_NONE_SHARE of examples and made wrong
  (`prompts.wrong_history`) in HISTORY_WRONG_SHARE, so the model can't learn "repeat the last move".
- **Photometric changes** of the camera image (brightness, contrast, saturation, slight blur)
  before the lines are drawn, so the line colors stay exact.
- **Uncertain frames** (most of the candidates' ground without ground truth) are left out.
- **Frames where no candidate matches the driver** (user decision 2026-11-23: an experiment):
  "skip" leaves them out; "keep" trains the safety part on them, with the choice masked out of
  the loss in Mode A (the picture doesn't show the driver's intent) and the closest candidate
  as the choice in Mode B (the word carries the intent).

The target has no reasoning field (user decision 2026-11-23): the training data has no
human-written reasons, so the fine-tuned model is trained and asked without it.
"""

import json
from dataclasses import dataclass

import cv2
import numpy as np

from offroad.nav.prompts import WORDINGS, HistoryEntry, build_prompt, history_text, shuffled, wrong_history

NO_MATCH_POLICIES = ("skip", "keep")
UNCERTAIN_MAX_UNKNOWN = 0.5  # mean share of candidate points without ground truth (14 of 1,527 training frames)
FRAME_WEIGHTS = {"easy": 0.5, "turn": 1.5, "puddle": 2.0, "unsafe": 3.0, "stop": 3.0}
HISTORY_NONE_SHARE = 0.3
HISTORY_WRONG_SHARE = 0.15
MIRROR_SHARE = 0.5
PHOTOMETRIC_RANGE = 0.15  # brightness, contrast and saturation change by up to ±15%
BLUR_SHARE = 0.3  # share of examples with a slight blur
HISTORY_KINDS = ("driver", "none", "wrong")


# --- what kind of frame, and how often to use it ---


def is_puddle_frame(frame: dict) -> bool:
    """Some candidate crosses a puddle and some candidate is dry and safe to tick 1."""
    cands = frame["candidates"]
    return (any(c["first_caution_m"] is not None for c in cands)
            and any(c["first_caution_m"] is None and c["safe_until"] != "none" for c in cands))


def frame_class(frame: dict) -> str:
    """"stop", "unsafe", "puddle", "turn" or "easy" (the first that applies, rarest first)."""
    if frame["stop"]:
        return "stop"
    if any(c["first_unsafe_m"] is not None for c in frame["candidates"]):
        return "unsafe"
    if is_puddle_frame(frame):
        return "puddle"
    if frame["direction"] not in (None, "straight"):
        return "turn"
    return "easy"


def frame_weight(frame: dict) -> float:
    return FRAME_WEIGHTS[frame_class(frame)]


def copies_this_epoch(weight: float, rng: np.random.Generator) -> int:
    """How many examples of a frame in one epoch: the whole part of the weight, plus one with the rest as chance."""
    whole = int(np.floor(weight))
    return whole + int(rng.random() < weight - whole)


def is_uncertain(frame: dict, max_unknown: float = UNCERTAIN_MAX_UNKNOWN) -> bool:
    """Most of the candidates' ground has no ground truth, so their labels are guesses."""
    shares = [c["unknown_share"] for c in frame["candidates"]]
    return bool(shares) and float(np.mean(shares)) > max_unknown


# --- mirroring ---


def mirrored_word(word: str | None) -> str | None:
    """'slight left' <-> 'slight right', 'bear left' <-> 'bear right'; 'straight' and None stay."""
    if word is None:
        return None
    return word.replace("left", "@").replace("right", "left").replace("@", "right")


# --- the answer the model should give ---


def target_text(listed: list[str], safe_by_color: dict[str, str], choice: str, action: str) -> str:
    """The training answer, as JSON in the prompt's format without reasoning; safe_until in the listed order."""
    return json.dumps({"safe_until": {c: safe_by_color[c] for c in listed}, "choice": choice, "action": action})


def choice_span(target: str) -> tuple[int, int]:
    """(start, end) characters of the choice's value in a `target_text` string, quotes included."""
    key = '"choice": '
    start = target.index(key) + len(key)
    end = target.index('"', start + 1) + 1
    return start, end


# --- one example ---


@dataclass
class Example:
    """One training example (the picture is drawn by the caller from these settings).

    colors: (name, RGB) per candidate, left to right in the real (unmirrored) scene.
    mask_choice: leave the choice out of the loss (Mode A frames where no candidate matches the driver).
    """

    sequence: str
    frame: int
    mode: str
    prompt: str
    target: str
    mask_choice: bool
    colors: list[tuple[str, tuple[int, int, int]]]
    mirror: bool
    history_kind: str
    wording: int
    frame_class: str


def driver_history(previous: list[tuple[float, float]]) -> list[HistoryEntry]:
    """History entries from earlier decision frames: (timestamp, the driver's candidate's end bearing)."""
    return [HistoryEntry(t, "continue", b) for t, b in previous]


def driver_choice(frame: dict) -> int:
    """Index of the driver's candidate (the closest to the driven path)."""
    return int(frame["closest_to_driver"])


def make_example(
    frame: dict,
    timestamp: float,
    previous: list[tuple[float, float]],
    palette: list[tuple[str, tuple[int, int, int]]],
    list_seed: int,
    mode: str,
    mirror: bool,
    history_kind: str,
    wording: int,
    no_match: str,
    wrong_seed: int = 0,
    reasoning: bool = False,
) -> Example | None:
    """One example for a frame with candidates, or None if the no-match policy leaves the frame out.

    frame: a record from build_candidates.py (candidates left to right, with labels);
    timestamp: the frame's time; previous: (timestamp, bearing) of the earlier decision frames
    used for the history; palette: (name, RGB) per candidate (from `assign_colors`).
    """
    if no_match not in NO_MATCH_POLICIES:
        raise ValueError(f"no_match must be one of {NO_MATCH_POLICIES}, got {no_match!r}")
    if history_kind not in HISTORY_KINDS:
        raise ValueError(f"history_kind must be one of {HISTORY_KINDS}, got {history_kind!r}")
    cands = frame["candidates"]
    if not cands:
        raise ValueError("frames without candidates are not asked, so they make no example")
    matched = any(c["matches_driver"] for c in cands)
    if not matched and no_match == "skip":
        return None

    names = [name for name, _ in palette]
    listed = shuffled(names, list_seed)
    word = frame["direction"]
    entries = driver_history(previous)
    if mirror:
        word = mirrored_word(word)
        entries = [HistoryEntry(e.timestamp, e.action, None if e.bearing_deg is None else -e.bearing_deg)
                   for e in entries]
    if history_kind == "none":
        history = "none."
    elif history_kind == "wrong":
        history = history_text(wrong_history(entries, wrong_seed)[0], timestamp)
    else:
        history = history_text(entries, timestamp)

    prompt = build_prompt(listed, mode, word if mode == "B" else None, history, reasoning=reasoning, wording=wording)
    safe = {name: c["safe_until"] for name, c in zip(names, cands)}
    action = "stop" if frame["stop"] else "continue"
    target = target_text(listed, safe, names[driver_choice(frame)], action)
    return Example(frame["sequence"], frame["frame"], mode, prompt, target, mask_choice=not matched and mode == "A",
                   colors=palette, mirror=mirror, history_kind=history_kind, wording=wording,
                   frame_class=frame_class(frame))


def draw_settings(rng: np.random.Generator) -> dict:
    """Random settings for one training example (mode, mirror, history kind, wording, photometric change, seeds)."""
    u = rng.random()
    history_kind = "none" if u < HISTORY_NONE_SHARE else "wrong" if u < HISTORY_NONE_SHARE + HISTORY_WRONG_SHARE \
        else "driver"
    return {
        "mode": "A" if rng.random() < 0.5 else "B",
        "mirror": bool(rng.random() < MIRROR_SHARE),
        "history_kind": history_kind,
        "wording": int(rng.integers(WORDINGS)),
        "color_seed": int(rng.integers(2**31)),
        "list_seed": int(rng.integers(2**31)),
        "wrong_seed": int(rng.integers(2**31)),
        "photometric": {
            "brightness": float(rng.uniform(-PHOTOMETRIC_RANGE, PHOTOMETRIC_RANGE)),
            "contrast": float(rng.uniform(-PHOTOMETRIC_RANGE, PHOTOMETRIC_RANGE)),
            "saturation": float(rng.uniform(-PHOTOMETRIC_RANGE, PHOTOMETRIC_RANGE)),
            "blur": bool(rng.random() < BLUR_SHARE),
        },
    }


def photometric(image: np.ndarray, brightness: float = 0.0, contrast: float = 0.0, saturation: float = 0.0,
                blur: bool = False) -> np.ndarray:
    """A changed copy of an (H, W, 3) uint8 RGB image: brightness / contrast / saturation as fractions (±)."""
    x = image.astype(np.float32)
    x = x * (1.0 + brightness)
    mean = x.mean()
    x = (x - mean) * (1.0 + contrast) + mean
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * (1.0 + saturation)
    out = np.clip(x, 0, 255).astype(np.uint8)
    if blur:
        out = cv2.GaussianBlur(out, (5, 5), 1.0)
    return out
