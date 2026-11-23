"""Tests for the candidate labels (driver match, safe distance, stop). No dataset needed."""

import numpy as np
import pytest

from offroad.mapping.elevation import GridSpec
from offroad.nav.candidates import Candidate, resample_and_cut
from offroad.nav.frame_candidates import unsafe_cells
from offroad.nav.labels import (
    CandidateLabels,
    cells_along,
    first_unsafe_index,
    has_real_choice,
    label_candidate,
    safe_until_tick,
    should_stop,
)

SPEC = GridSpec(x_min=-5.0, x_max=25.0, y_min=-12.0, y_max=12.0, resolution=0.25)


def straight(length=15.0, y=0.0) -> np.ndarray:
    return resample_and_cut(np.array([[0.0, y], [length, y]]), 0.25, length)


def candidate(xy) -> Candidate:
    return Candidate(xy=xy, target_bearing_deg=0.0, cost_per_m=1.0)


def labels_with(safe_until: str) -> CandidateLabels:
    return CandidateLabels(0.0, True, None, None, None, 15.0, safe_until, 0.0)


def test_cells_along_merges_repeats_and_skips_points_outside():
    xy = np.array([[0.05, 0.05], [0.1, 0.1], [0.3, 0.1], [100.0, 0.0]])
    cells, index = cells_along(xy, SPEC)
    assert cells == [SPEC.cell_index(0.05, 0.05), SPEC.cell_index(0.3, 0.1)]
    assert index.tolist() == [0, 0, 1, -1]


def test_first_unsafe_index_ignores_single_points():
    assert first_unsafe_index(np.array([False, True, False, False])) is None
    assert first_unsafe_index(np.array([False, True, False, True, True, True])) == 3
    assert first_unsafe_index(np.zeros(5, dtype=bool)) is None


@pytest.mark.parametrize("safe_m, full, answer", [
    (3.0, True, "none"), (5.0, True, "tick 1"), (9.9, True, "tick 1"), (12.0, True, "tick 2"),
    (14.2, True, "all"), (15.0, True, "all"), (14.2, False, "tick 2"),
])
def test_safe_until_tick(safe_m, full, answer):
    assert safe_until_tick(safe_m, full) == answer


def test_label_candidate_driver_match():
    driven = straight(20.0)
    assert label_candidate(candidate(straight(y=0.5)), np.ones(61, int), np.zeros(61, bool), driven).matches_driver
    far = label_candidate(candidate(straight(y=2.0)), np.ones(61, int), np.zeros(61, bool), driven)
    assert not far.matches_driver and far.driven_mean_m == pytest.approx(2.0)


def test_driver_match_ignores_the_shared_start():
    # Follows the driven path for 5 m, then veers off 45 deg: the far part decides.
    veer = resample_and_cut(np.array([[0.0, 0.0], [5.0, 0.0], [12.0, 7.0]]), 0.25, 15.0)
    lab = label_candidate(candidate(veer), np.ones(len(veer), int), np.zeros(len(veer), bool), straight(20.0))
    assert not lab.matches_driver and lab.driven_mean_m > 2.0


def test_label_candidate_safe_distance_and_class():
    xy = straight()
    classes = np.full(len(xy), 3)
    unsafe = np.zeros(len(xy), bool)
    classes[28:32], unsafe[28:32] = 6, True  # water from 7 m to 7.75 m
    lab = label_candidate(candidate(xy), classes, unsafe, straight(20.0))
    assert lab.first_unsafe_m == pytest.approx(7.0) and lab.first_unsafe_class == 6
    assert lab.safe_until == "tick 1" and lab.safe_m == pytest.approx(7.0)
    clean = label_candidate(candidate(xy), classes, np.zeros(len(xy), bool), straight(20.0))
    assert clean.safe_until == "all" and clean.first_unsafe_m is None


def test_puddle_is_caution_not_unsafe():
    xy = straight()
    classes = np.full(len(xy), 3)
    caution = np.zeros(len(xy), bool)
    classes[12:20], caution[12:20] = 31, True  # a puddle from 3 m to 4.75 m
    lab = label_candidate(candidate(xy), classes, np.zeros(len(xy), bool), straight(20.0), caution)
    assert lab.first_caution_m == pytest.approx(3.0)
    assert lab.safe_until == "all" and lab.first_unsafe_m is None  # still safe: no stop because of it
    assert label_candidate(candidate(xy), classes, np.zeros(len(xy), bool), straight(20.0)).first_caution_m is None


def test_short_candidate_is_safe_only_to_its_end():
    xy = straight(length=11.0)
    lab = label_candidate(candidate(xy), np.ones(len(xy), int), np.zeros(len(xy), bool), straight(20.0))
    assert lab.safe_m == pytest.approx(11.0) and lab.safe_until == "tick 2"


def test_unknown_share():
    xy = straight()
    classes = np.ones(len(xy), int)
    classes[:20] = 0
    lab = label_candidate(candidate(xy), classes, np.zeros(len(xy), bool), straight(20.0))
    assert lab.unknown_share == pytest.approx(20 / len(xy))


def test_stop_and_real_choice():
    assert should_stop([]) and not has_real_choice([])
    assert should_stop([labels_with("none"), labels_with("none")])
    assert not should_stop([labels_with("none"), labels_with("tick 1")])
    assert has_real_choice([labels_with("tick 1"), labels_with("all"), labels_with("none")])
    assert not has_real_choice([labels_with("all"), labels_with("none")])


def test_hard_obstacles_count_only_where_something_stands_up():
    cells = [(0, 0), (0, 1), (0, 2), (0, 3)]
    classes = np.array([31, 4, 4, 3])  # puddle, tree, tree, grass
    standing = np.zeros((1, 4), bool)
    standing[0, 2] = True  # only the second tree cell has something tall
    unsafe = unsafe_cells(classes, cells, standing, water_ids=[31], hard_ids=[4])
    assert unsafe.tolist() == [True, False, True, False]
