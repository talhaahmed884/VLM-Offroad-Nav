"""Tests for the goal cell and the runner helpers. No model, GPU, or dataset needed."""

import json

import numpy as np
import pytest

from offroad.mapping.elevation import GridSpec
from offroad.vlm.answer import parse_answer
from offroad.vlm.goal import goal_cell, goal_pixel, ground_height_at
from offroad.vlm.grid import ImageGrid
from offroad.vlm.runner import answer_to_dict, downscale, strip_thinking

SPEC = GridSpec()  # x -5..35, y -15..15, 0.25 m cells
GRID = ImageGrid(height=1200, width=1920)  # 150 x 160 px cells
K = np.array([[2800.0, 0.0, 960.0], [0.0, 2800.0, 600.0], [0.0, 0.0, 1.0]])

# A camera at the LiDAR origin looking forward (LiDAR -x), level with the ground:
# camera x (right) = LiDAR y, camera y (down) = -LiDAR z, camera z (forward) = -LiDAR x.
T_CAM_FROM_LIDAR = np.eye(4)
T_CAM_FROM_LIDAR[:3, :3] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]


def flat_ground(height: float) -> np.ndarray:
    return np.full(SPEC.shape, height)


# --- ground height ---


def test_ground_height_uses_the_cell_itself():
    ground = flat_ground(-1.5)
    i, j = SPEC.cell_index(20.0, 0.0)
    ground[i, j] = -1.2
    assert ground_height_at(ground, SPEC, 20.0, 0.0) == pytest.approx(-1.2)


def test_ground_height_falls_back_to_nearby_cells():
    ground = flat_ground(-1.5)
    i, j = SPEC.cell_index(20.0, 0.0)
    ground[i - 2:i + 3, j - 2:j + 3] = np.nan  # a hole around the target, e.g. a puddle
    assert ground_height_at(ground, SPEC, 20.0, 0.0) == pytest.approx(-1.5)


def test_ground_height_unknown_gives_none():
    assert ground_height_at(flat_ground(np.nan), SPEC, 20.0, 0.0) is None
    assert ground_height_at(flat_ground(-1.5), SPEC, 100.0, 0.0) is None  # far outside the map


# --- goal pixel and cell ---


def test_goal_pixel_straight_ahead():
    u, v = goal_pixel((20.0, 0.0, -1.5), T_CAM_FROM_LIDAR, K, (1200, 1920))
    assert u == pytest.approx(960.0)
    assert v == pytest.approx(600.0 + 2800.0 * 1.5 / 20.0)  # 1.5 m below the camera -> 810


def test_goal_pixel_left_is_left_in_the_image():
    u, _ = goal_pixel((20.0, 2.0, -1.5), T_CAM_FROM_LIDAR, K, (1200, 1920))
    assert u < 960.0


def test_goal_pixel_behind_or_outside_is_none():
    assert goal_pixel((-20.0, 0.0, -1.5), T_CAM_FROM_LIDAR, K, (1200, 1920)) is None
    assert goal_pixel((20.0, 0.0, 10.0), T_CAM_FROM_LIDAR, K, (1200, 1920)) is None  # above the image


def test_goal_cell_on_flat_ground():
    # v = 810 -> row 5 ("F"); u = 960 -> column index 6 ("7").
    assert goal_cell(flat_ground(-1.5), SPEC, T_CAM_FROM_LIDAR, K, GRID) == "F7"


def test_goal_cell_closer_goal_is_lower_in_the_image():
    assert goal_cell(flat_ground(-1.5), SPEC, T_CAM_FROM_LIDAR, K, GRID, goal_distance=8.0) == "H7"


def test_goal_cell_unknown_ground_gives_none():
    assert goal_cell(flat_ground(np.nan), SPEC, T_CAM_FROM_LIDAR, K, GRID) is None


# --- runner helpers ---


def test_downscale_keeps_aspect_ratio():
    image = np.zeros((1200, 1920, 3), dtype=np.uint8)
    assert downscale(image, 1280).shape == (800, 1280, 3)


def test_downscale_leaves_small_images_alone():
    image = np.zeros((600, 960, 3), dtype=np.uint8)
    assert downscale(image, 1280) is image


def test_strip_thinking():
    assert strip_thinking("<think>a { b }</think>\n{\"x\": 1}") == "{\"x\": 1}"
    assert strip_thinking("{\"x\": 1}") == "{\"x\": 1}"


def test_answer_to_dict_is_json_ready():
    text = json.dumps({
        "waypoints": ["H7", "G7"],
        "cost_hints": [{"cells": ["G5"], "terrain": "water", "traversability": "avoid"}],
        "reasoning": "r",
    })
    record = answer_to_dict(parse_answer(text, GRID))
    assert json.loads(json.dumps(record)) == {
        "valid": True,
        "waypoints": ["H7", "G7"],
        "hints": [{"cell": "G5", "terrain": "water", "traversability": "avoid"}],
        "reasoning": "r",
        "issues": [],
        "route_conflicts": [],
    }
