"""Tests for the VLM's picture: colors, smoothed ground, projection, drawing. No dataset or model needed."""

import numpy as np
import pytest

from offroad.data.calibration import CameraCalibration
from offroad.mapping.elevation import GridSpec
from offroad.nav.candidates import CANDIDATE_COLORS, Candidate, resample_and_cut
from offroad.nav.selector import prepare_selector_input
from offroad.nav.visual_prompt import (
    IMAGE_WIDTH,
    assign_colors,
    draw_visual_prompt,
    frame_seed,
    project_flu,
    smooth_ground_heights,
    spread_overlapping,
)

SPEC = GridSpec(x_min=-5.0, x_max=25.0, y_min=-12.0, y_max=12.0, resolution=0.25)
IMAGE_SIZE = (1200, 1920)


def forward_camera() -> CameraCalibration:
    """A camera at the LiDAR looking along the vehicle's forward axis (FLU +x, which is LiDAR -x)."""
    K = np.array([[2800.0, 0.0, 960.0], [0.0, 2800.0, 600.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    # camera x = right = LiDAR +y; camera y = down = LiDAR -z; camera z = forward = LiDAR -x.
    T[:3, :3] = [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]
    return CameraCalibration(K=K, dist_coeffs=np.zeros(5), T_cam_from_lidar=T)


def straight(length=15.0, y=0.0) -> np.ndarray:
    return resample_and_cut(np.array([[0.0, y], [length, y]]), 0.25, length)


def flat_ground(z=-1.2) -> np.ndarray:
    return np.full(SPEC.shape, z)


# --- colors ---


def test_assign_colors_is_repeatable_and_distinct():
    a = assign_colors(4, seed=123)
    assert a == assign_colors(4, seed=123)
    assert len({name for name, _ in a}) == 4 and set(a) == set(CANDIDATE_COLORS)
    orders = {tuple(name for name, _ in assign_colors(4, frame_seed("00003", f))) for f in range(40)}
    assert len(orders) > 5  # the order really changes from frame to frame
    with pytest.raises(ValueError):
        assign_colors(5, seed=0)


# --- ground heights ---


def test_smooth_ground_follows_a_slope_and_ignores_outliers():
    xs = SPEC.x_min + (np.arange(SPEC.shape[0]) + 0.5) * SPEC.resolution
    ground = np.tile((-1.2 + 0.1 * xs)[:, None], (1, SPEC.shape[1]))  # rising 10% forward
    ground[SPEC.cell_index(8.0, 0.0)] = 3.0  # one bad cell
    xy = straight()
    z = smooth_ground_heights(xy, ground, SPEC)
    assert np.allclose(z, -1.2 + 0.1 * xy[:, 0], atol=0.06)


def test_smooth_ground_holds_the_ends_and_falls_back():
    ground = np.full(SPEC.shape, np.nan)
    xy = straight()
    assert np.allclose(smooth_ground_heights(xy, ground, SPEC), -1.2)  # nothing known: the default
    ground[:, :] = np.nan
    for x in np.arange(6.0, 12.0, 0.25):  # known only from 6 to 12 m
        ground[SPEC.cell_index(x, 0.0)] = -0.8
    z = smooth_ground_heights(xy, ground, SPEC)
    assert np.allclose(z, -0.8, atol=1e-6)  # held constant in the blind zone and beyond


# --- projection and drawing ---


def test_project_flu_marks_points_out_of_view():
    calib = forward_camera()
    uv = project_flu(np.array([[10.0, 0.0, 0.0], [-5.0, 0.0, -1.2], [10.0, 50.0, -1.2]]), calib, IMAGE_SIZE)
    assert np.allclose(uv[0], [960.0, 600.0])  # straight ahead at the camera height: the image center
    assert np.isnan(uv[1]).all() and np.isnan(uv[2]).all()  # behind / far to the side


def test_draw_visual_prompt_draws_each_path_in_its_color():
    image = np.full((*IMAGE_SIZE, 3), 128, dtype=np.uint8)
    colors = [("red", (230, 40, 40)), ("blue", (40, 110, 255))]
    picture, visible = draw_visual_prompt(image, forward_camera(), [straight(y=1.5), straight(y=-1.5)], colors,
                                          flat_ground(), SPEC)
    assert picture.shape == (500, IMAGE_WIDTH, 3) and visible == [True, True]
    for _, rgb in colors:
        assert (np.abs(picture.astype(int) - rgb).sum(axis=2) < 10).sum() > 100
    assert (picture == 255).all(axis=2).sum() > 20  # white tick dots
    assert (image == 128).all()  # the input is not changed


def test_spread_overlapping_puts_shared_lines_side_by_side():
    trunk = np.column_stack([np.full(20, 400.0), np.linspace(500.0, 300.0, 20)])  # going up the image
    left = np.vstack([trunk, [[380.0, 290.0], [360.0, 280.0]]])  # ends to the left
    right = np.vstack([trunk, [[420.0, 290.0], [440.0, 280.0]]])
    alone = trunk + [200.0, 0.0]  # far from both
    a, b, c = spread_overlapping([right, left, alone], spacing_px=7.0, overlap_px=8.0, smooth_points=1)
    assert np.allclose(b[:19, 0], 396.5) and np.allclose(a[:19, 0], 403.5)  # left one on the left, 7 px apart
    assert np.allclose(b[:19, 1], left[:19, 1]) and np.allclose(c, alone)  # moved only sideways; others untouched
    assert np.allclose(a[-1], right[-1]) and np.allclose(b[-1], left[-1])  # apart again: back in place
    with_gap = left.copy()
    with_gap[5] = np.nan  # a point out of view stays out of view
    assert np.isnan(spread_overlapping([with_gap, right], 7.0, 8.0)[0][5]).all()


def test_shared_stretch_shows_both_colors():
    branch_left = resample_and_cut(np.array([[0.0, 0.0], [10.0, 0.0], [15.0, 2.0]]), 0.25, 15.0)
    branch_right = resample_and_cut(np.array([[0.0, 0.0], [10.0, 0.0], [15.0, -2.0]]), 0.25, 15.0)
    colors = [("red", (230, 40, 40)), ("blue", (40, 110, 255))]
    image = np.full((*IMAGE_SIZE, 3), 128, dtype=np.uint8)
    picture, _ = draw_visual_prompt(image, forward_camera(), [branch_left, branch_right], colors, flat_ground(), SPEC)
    rows = picture[420:480]  # ~6-8.5 m ahead: the shared first 10 m, low in the picture
    for _, rgb in colors:  # both lines visible there, not one on top of the other
        assert (np.abs(rows.astype(int) - rgb).sum(axis=2) < 10).sum() > 30


def test_path_out_of_view_is_reported():
    image = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
    behind = resample_and_cut(np.array([[0.0, 0.0], [-4.0, 0.0]]), 0.25, 15.0)
    _, visible = draw_visual_prompt(image, forward_camera(), [behind], [("red", (230, 40, 40))], flat_ground(), SPEC)
    assert visible == [False]


def test_prepare_selector_input():
    image = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
    candidates = [Candidate(straight(y=y), 0.0, 1.0) for y in (2.0, 0.0, -2.0)]
    a = prepare_selector_input(image, forward_camera(), candidates, flat_ground(), SPEC, "00003", 42, "B", "straight")
    b = prepare_selector_input(image, forward_camera(), candidates, flat_ground(), SPEC, "00003", 42, "B", "straight")
    assert a.colors == b.colors and a.listed == b.listed and a.prompt == b.prompt  # repeatable
    assert sorted(a.listed) == sorted(a.colors) and len(set(a.colors)) == 3
    assert "Candidates: " + ", ".join(a.listed) in a.prompt and '"straight"' in a.prompt
