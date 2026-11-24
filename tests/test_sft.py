"""Tests for the phase 5 training data (nav/sft.py), the label masking (vlm/finetune.py), and the
no-reasoning / wording / mirror options. No dataset, model or GPU needed."""

import json

import numpy as np
import pytest

from offroad.data.calibration import CameraCalibration
from offroad.mapping.elevation import GridSpec
from offroad.nav.candidates import resample_and_cut
from offroad.nav.prompts import WORDINGS, build_prompt, parse_answer
from offroad.nav.sft import (
    choice_span,
    copies_this_epoch,
    draw_settings,
    frame_class,
    is_uncertain,
    make_example,
    mirrored_word,
    photometric,
    target_text,
)
from offroad.nav.visual_prompt import draw_visual_prompt
from offroad.vlm.finetune import IGNORE, answer_labels, epoch_end_steps

PALETTE = [("red", (230, 40, 40)), ("blue", (40, 110, 255)), ("yellow", (250, 210, 0))]


def cand(bearing, match=False, safe="all", puddle=None, unsafe=None, unknown=0.25):
    return {"end_bearing_deg": bearing, "matches_driver": match, "safe_until": safe, "first_caution_m": puddle,
            "first_unsafe_m": unsafe, "unknown_share": unknown}


def frame(cands, closest, direction="straight", stop=False):
    return {"sequence": "00000", "frame": 10, "candidates": cands, "closest_to_driver": closest,
            "direction": direction, "stop": stop}


def three(direction="slight left", match=True, **kw):
    return frame([cand(15.0, match=match), cand(0.0, safe="tick 2", unsafe=11.0), cand(-15.0)], 0, direction, **kw)


# --- prompts ---


def test_prompt_without_reasoning_and_wordings():
    colors = ["red", "blue"]
    plain = build_prompt(colors, "A")
    short = build_prompt(colors, "A", reasoning=False)
    assert '"reasoning"' in plain and '"reasoning"' not in short
    assert len({build_prompt(colors, "B", "bear left", wording=w) for w in range(WORDINGS)}) == WORDINGS
    with pytest.raises(ValueError):
        build_prompt(colors, "A", wording=WORDINGS)
    answer = parse_answer('{"safe_until": {"red": "all", "blue": "none"}, "choice": "red", "action": "continue"}',
                          colors)
    assert answer.valid and answer.reasoning == ""  # an answer without reasoning is still valid


# --- frames ---


def test_frame_class_and_uncertain():
    assert frame_class(three()) == "unsafe"
    assert frame_class(frame([cand(0.0)], 0, stop=True)) == "stop"
    assert frame_class(frame([cand(0.0, puddle=5.0), cand(5.0)], 0)) == "puddle"
    assert frame_class(frame([cand(0.0)], 0, "bear right")) == "turn"
    assert frame_class(frame([cand(0.0)], 0)) == "easy"
    assert is_uncertain(frame([cand(0.0, unknown=0.7), cand(5.0, unknown=0.4)], 0))
    assert not is_uncertain(frame([cand(0.0, unknown=0.3)], 0))


def test_copies_average_to_the_weight():
    rng = np.random.default_rng(0)
    assert np.mean([copies_this_epoch(1.5, rng) for _ in range(4000)]) == pytest.approx(1.5, abs=0.05)
    assert {copies_this_epoch(3.0, rng) for _ in range(20)} == {3}


def test_mirrored_word():
    assert mirrored_word("bear left") == "bear right" and mirrored_word("slight right") == "slight left"
    assert mirrored_word("straight") == "straight" and mirrored_word(None) is None


# --- targets and examples ---


def test_target_and_choice_span():
    t = target_text(["blue", "red"], {"red": "all", "blue": "tick 1"}, "red", "continue")
    assert json.loads(t) == {"safe_until": {"blue": "tick 1", "red": "all"}, "choice": "red", "action": "continue"}
    assert list(json.loads(t)["safe_until"]) == ["blue", "red"]  # the listed order
    start, end = choice_span(t)
    assert t[start:end] == '"red"'


def test_make_example():
    f = three()
    ex = make_example(f, 100.0, [(99.5, 10.0)], PALETTE, list_seed=1, mode="B", mirror=False,
                      history_kind="driver", wording=0, no_match="skip")
    target = json.loads(ex.target)
    assert target["choice"] == "red" and target["action"] == "continue"
    assert target["safe_until"] == {"red": "all", "blue": "tick 2", "yellow": "all"}
    assert '"slight left"' in ex.prompt and "0.5 s ago: continued slight left" in ex.prompt
    assert '"reasoning"' not in ex.prompt and not ex.mask_choice and ex.frame_class == "unsafe"

    mirrored = make_example(f, 100.0, [(99.5, 10.0)], PALETTE, 1, "B", True, "driver", 0, "skip")
    assert '"slight right"' in mirrored.prompt and "continued slight right" in mirrored.prompt
    assert json.loads(mirrored.target) == target  # colors stay with their paths

    none = make_example(f, 100.0, [(99.5, 10.0)], PALETTE, 1, "A", False, "none", 0, "skip")
    assert "Recent decisions (oldest first): none." in none.prompt and '"slight left"' not in none.prompt
    wrong = make_example(f, 100.0, [(99.5, 10.0)], PALETTE, 1, "A", False, "wrong", 0, "skip")
    assert "continued bear right" in wrong.prompt  # opposite to the real left turn


def test_no_match_policies():
    f = three(match=False)
    assert make_example(f, 1.0, [], PALETTE, 1, "A", False, "none", 0, "skip") is None
    a = make_example(f, 1.0, [], PALETTE, 1, "A", False, "none", 0, "keep")
    b = make_example(f, 1.0, [], PALETTE, 1, "B", False, "none", 0, "keep")
    assert a.mask_choice and not b.mask_choice  # Mode A: safety only; Mode B: the closest candidate
    assert json.loads(b.target)["choice"] == "red"
    with pytest.raises(ValueError):
        make_example(f, 1.0, [], PALETTE, 1, "A", False, "none", 0, "maybe")


def test_draw_settings_shares():
    rng = np.random.default_rng(1)
    s = [draw_settings(rng) for _ in range(3000)]
    assert np.mean([x["mode"] == "A" for x in s]) == pytest.approx(0.5, abs=0.04)
    assert np.mean([x["mirror"] for x in s]) == pytest.approx(0.5, abs=0.04)
    assert np.mean([x["history_kind"] == "none" for x in s]) == pytest.approx(0.3, abs=0.04)
    assert np.mean([x["history_kind"] == "wrong" for x in s]) == pytest.approx(0.15, abs=0.03)


def test_photometric():
    image = np.full((20, 30, 3), 100, dtype=np.uint8)
    assert (photometric(image) == image).all()
    assert photometric(image, brightness=0.1).mean() == pytest.approx(110, abs=1)
    assert photometric(image, contrast=0.5, saturation=0.5, blur=True).shape == image.shape


# --- label masking ---


def test_answer_labels():
    ids, labels = answer_labels([5, 6, 7, 8], [(0, 3), (3, 6), (6, 9), (9, 12)], None, end_id=99)
    assert ids == [5, 6, 7, 8, 99] and labels == ids
    ids, labels = answer_labels([5, 6, 7, 8], [(0, 3), (3, 6), (6, 9), (9, 12)], (4, 8), end_id=99)
    assert labels == [5, IGNORE, IGNORE, 8, 99]  # the tokens touching characters 4-7


def test_epoch_end_steps():
    assert epoch_end_steps([16, 16, 10], grad_accum=8) == [2, 4, 6]  # 42 examples -> step 6 (ceil)


# --- mirrored drawing ---


def test_mirrored_picture_mirrors_the_lines():
    K = np.array([[2800.0, 0.0, 960.0], [0.0, 2800.0, 600.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]
    calib = CameraCalibration(K=K, dist_coeffs=np.zeros(5), T_cam_from_lidar=T)
    spec = GridSpec(x_min=-5.0, x_max=25.0, y_min=-12.0, y_max=12.0, resolution=0.25)
    image = np.zeros((1200, 1920, 3), dtype=np.uint8)
    image[:, :960] = 60  # left half lighter
    left = resample_and_cut(np.array([[0.0, 0.0], [15.0, 2.0]]), 0.25, 15.0)  # ends on the left (+y)
    ground = np.full(spec.shape, -1.2)
    plain, _ = draw_visual_prompt(image, calib, [left], PALETTE[:1], ground, spec)
    flipped, _ = draw_visual_prompt(image, calib, [left], PALETTE[:1], ground, spec, mirror=True)
    red = lambda p: np.nonzero((np.abs(p.astype(int) - PALETTE[0][1]).sum(axis=2) < 10).any(axis=0))[0]  # noqa: E731
    assert red(plain).mean() < 400 < red(flipped).mean()  # the line moved to the right half
    assert plain[10, 10, 0] == 60 and flipped[10, 790, 0] == 60  # and the image with it
