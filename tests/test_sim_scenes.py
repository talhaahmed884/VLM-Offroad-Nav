"""Tests for the step 6.4 scene designs (no MAVS needed)."""

import math

import numpy as np
import pytest

from offroad.sim.metrics import cumulative_length
from offroad.sim.scenes import BLOCKADE_LENGTH_M, across, arc, chain, designs, near_any, straight, vprp_text


def test_straight_and_arc_geometry():
    s = straight((0.0, 0.0), 0.0, 4.0)
    assert len(s) == 10 and s[-1] == pytest.approx([4.0, 0.0])
    a = arc((0.0, 0.0), 0.0, 10.0, math.pi / 2)  # a quarter circle to the left
    assert a[-1] == pytest.approx([10.0, 10.0], abs=1e-6)
    assert np.hypot(a[:, 0], a[:, 1] - 10.0) == pytest.approx(np.full(len(a), 10.0))
    b = arc((0.0, 0.0), 0.0, 10.0, -math.pi / 2)  # to the right
    assert b[-1] == pytest.approx([10.0, -10.0], abs=1e-6)


def test_chain_keeps_the_heading_through_arcs():
    t = chain((0.0, 0.0), 0.0, [("arc", 10.0, math.pi / 2), ("straight", 4.0)])
    assert t[-1] == pytest.approx([10.0, 14.0], abs=1e-6)  # after turning left 90 degrees it goes north
    assert cumulative_length(t)[-1] == pytest.approx(10 * math.pi / 2 + 4.0, rel=1e-3)


def test_designs_are_sensible():
    d = designs()
    assert set(d) == {"curves", "fork", "blocked", "halfblocked"}
    for design in d.values():
        for line in list(design.trails.values()) + list(design.routes.values()):
            assert np.all(np.abs(line) < 72.0)  # inside the 150 m terrain
            assert np.all(np.linalg.norm(np.diff(line, axis=0), axis=1) < 0.45)  # no gaps
    fork = d["fork"]
    assert set(fork.routes) == {"left", "right"} and fork.routes["left"][-1, 1] > 15 > -15 > fork.routes["right"][-1, 1]
    # Every bend is gentle enough for the RELLIS camera (usable +-17 degrees): standing anywhere on
    # a route and facing along it, the route point 15 m ahead is within 15 degrees of straight ahead.
    for design in d.values():
        for line in design.routes.values():
            for k in range(0, len(line) - 40, 5):
                d0 = line[k + 1] - line[k]
                heading = math.atan2(d0[1], d0[0])
                ahead = line[k + 37] - line[k]  # 37 x 0.4 m = 14.8 m along the route
                bearing = math.degrees(math.atan2(ahead[1], ahead[0]) - heading)
                assert abs((bearing + 180) % 360 - 180) < 15.0
    assert len(d["blocked"].blockades) == 4 and len(d["halfblocked"].blockades) == 1


def test_blockades_close_or_half_close_the_trail():
    d = designs()
    line = d["blocked"].routes["main"]
    walls = np.vstack([b.footprint_points() for b in d["blocked"].blockades])
    assert np.ptp(walls[:, 1]) == pytest.approx(4 * BLOCKADE_LENGTH_M, abs=0.25)  # across the trail and grass
    assert walls[:, 0] == pytest.approx(np.full(len(walls), line[100, 0]), abs=1e-6)  # 40 m in
    half = d["halfblocked"].blockades[0].footprint_points()
    assert 0.5 < half[:, 1].min() < 1.0  # the left third (+y) of the trail only
    route = d["halfblocked"].routes["main"]
    gap = np.min(np.linalg.norm(route[:, None, :] - half[None, :, :], axis=2))
    assert gap > 1.8  # the route swerves round the wall's end with room for the 1.6 m wide vehicle
    assert route[0] == pytest.approx(line[0]) and route[-1] == pytest.approx(line[-1])
    b = across(np.column_stack([np.arange(0, 20, 0.4), np.zeros(50)]), 8.0)
    assert b.yaw == pytest.approx(math.pi / 2)


def test_vprp_and_near_any():
    text = vprp_text(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert text.splitlines()[0] == "QS VANE REPLAY FILE" and text.splitlines()[-1] == "0.02:3.0000 4.0000 0.0:1 0 0 0"
    line = np.column_stack([np.arange(0, 10, 0.4), np.zeros(25)])
    assert near_any(np.array([[5.0, 1.0], [5.0, 3.0]]), [line], 2.0).tolist() == [True, False]
