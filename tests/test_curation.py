"""Tests for evaluation-frame curation helpers. No dataset needed."""

import numpy as np

from offroad.eval.curation import Candidate, ahead_region_mask, class_pixel_fractions, pick_spaced


def test_ahead_region_mask_covers_expected_block():
    mask = ahead_region_mask((100, 200), rows=(0.5, 1.0), cols=(0.25, 0.75))
    assert mask.shape == (100, 200)
    assert mask[50:, 50:150].all()
    assert mask.sum() == 50 * 100


def test_class_pixel_fractions_whole_image():
    label = np.array([[3, 3], [3, 33]], dtype=np.uint8)
    fractions = class_pixel_fractions(label)
    assert fractions.shape == (256,)
    assert fractions[3] == 0.75
    assert fractions[33] == 0.25
    assert np.isclose(fractions.sum(), 1.0)


def test_class_pixel_fractions_respects_mask():
    label = np.array([[3, 3], [33, 33]], dtype=np.uint8)
    mask = np.array([[False, False], [True, True]])
    fractions = class_pixel_fractions(label, mask)
    assert fractions[33] == 1.0
    assert fractions[3] == 0.0


def test_pick_spaced_takes_best_first_and_skips_near_duplicates():
    candidates = [
        Candidate("00000", 100, 0.9),
        Candidate("00000", 110, 0.8),  # too close to frame 100
        Candidate("00001", 105, 0.7),  # different sequence, so allowed
        Candidate("00000", 300, 0.6),
    ]
    picked = pick_spaced(candidates, n=3, min_gap=50)
    assert [(c.sequence, c.frame) for c in picked] == [("00000", 100), ("00001", 105), ("00000", 300)]


def test_pick_spaced_avoids_frames_picked_for_other_categories():
    earlier = [Candidate("00002", 500, 0.5)]
    candidates = [Candidate("00002", 520, 0.9), Candidate("00002", 900, 0.1)]
    picked = pick_spaced(candidates, n=2, min_gap=100, already_picked=earlier)
    assert [c.frame for c in picked] == [900]


def test_pick_spaced_limits_picks_per_sequence():
    candidates = [Candidate("00002", f, 1.0) for f in (0, 200, 400, 600)] + [Candidate("00001", 0, 0.5)]
    picked = pick_spaced(candidates, n=3, min_gap=100, max_per_sequence=2)
    assert [(c.sequence, c.frame) for c in picked] == [("00002", 0), ("00002", 200), ("00001", 0)]


def test_pick_spaced_returns_fewer_when_not_enough_candidates():
    picked = pick_spaced([Candidate("00000", 0, 0.5)], n=5, min_gap=10)
    assert len(picked) == 1
