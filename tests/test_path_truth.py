"""Tests for two-source (LiDAR + camera) ground truth along a path. No dataset needed."""

import numpy as np
import pytest

from offroad.eval.path_truth import (
    SOURCE_CAMERA,
    SOURCE_LIDAR,
    SOURCE_OCCLUDED,
    SOURCE_UNSEEN,
    UNSEEN,
    lidar_depth_image,
    path_truth,
    project_cells,
)
from offroad.mapping.elevation import GridSpec

SPEC = GridSpec()  # x -5..35, y -15..15, 0.25 m cells
K = np.array([[2800.0, 0.0, 960.0], [0.0, 2800.0, 600.0], [0.0, 0.0, 1.0]])
T_CAM_FROM_LIDAR = np.eye(4)
T_CAM_FROM_LIDAR[:3, :3] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]  # level camera at the LiDAR, looking forward
GROUND = np.full(SPEC.shape, -1.5)  # flat ground 1.5 m below the camera
GRASS, PUDDLE, SKY, BUSH = 3, 31, 7, 19

CELL_15M = SPEC.cell_index(15.0, 0.0)  # its ground point is at about pixel (960, 880)
CELL_3M = SPEC.cell_index(3.0, 0.0)  # below the bottom of the image


def label_image(class_id: int = GRASS) -> np.ndarray:
    return np.full((1200, 1920), class_id, dtype=np.uint8)


def no_lidar() -> np.ndarray:
    return np.full(SPEC.shape, UNSEEN)


def truth(path, lidar_labels=None, image=None, points=None, ignore=(UNSEEN,)):
    return path_truth(path, no_lidar() if lidar_labels is None else lidar_labels, GROUND, SPEC,
                      T_CAM_FROM_LIDAR, K, image, points_lidar=points, ignore_ids=ignore)


def flu_to_lidar(points_flu):
    return np.asarray(points_flu, dtype=float) * np.array([-1.0, -1.0, 1.0])


def test_lidar_label_comes_first():
    lidar = no_lidar()
    lidar[CELL_15M] = BUSH
    result = truth([CELL_15M], lidar_labels=lidar, image=label_image(PUDDLE))
    assert result.labels.tolist() == [BUSH] and result.source.tolist() == [SOURCE_LIDAR]


def test_camera_fills_cells_without_lidar_points():
    result = truth([CELL_15M], image=label_image(PUDDLE))
    assert result.labels.tolist() == [PUDDLE] and result.source.tolist() == [SOURCE_CAMERA]


def test_cells_out_of_view_stay_unseen():
    result = truth([CELL_3M], image=label_image(PUDDLE))
    assert result.labels.tolist() == [UNSEEN] and result.source.tolist() == [SOURCE_UNSEEN]


def test_sky_label_at_a_ground_point_is_ignored():
    result = truth([CELL_15M], image=label_image(SKY), ignore=(UNSEEN, SKY))
    assert result.source.tolist() == [SOURCE_UNSEEN]


def test_no_label_image_means_lidar_only():
    result = truth([CELL_15M], image=None)
    assert result.source.tolist() == [SOURCE_UNSEEN]


def test_something_in_front_marks_the_cell_occluded():
    # A bush point two thirds of the way to the cell's ground point, on the same line of sight (same pixel).
    x, y = SPEC.cell_center(*CELL_15M)
    bush = flu_to_lidar([[x, y, -1.5]]) * (2 / 3)
    result = truth([CELL_15M], image=label_image(BUSH), points=bush)
    assert result.source.tolist() == [SOURCE_OCCLUDED] and result.labels.tolist() == [UNSEEN]


def test_ground_points_near_the_cell_are_not_occlusion():
    x, y = SPEC.cell_center(*CELL_15M)
    ground_point = flu_to_lidar([[x, y, -1.5]])
    result = truth([CELL_15M], image=label_image(PUDDLE), points=ground_point)
    assert result.source.tolist() == [SOURCE_CAMERA]


def test_project_cells_matches_the_pinhole_formula():
    uv, depth = project_cells([CELL_15M, CELL_3M], GROUND, SPEC, T_CAM_FROM_LIDAR, K, (1200, 1920))
    x, _ = SPEC.cell_center(*CELL_15M)
    assert uv[0, 1] == pytest.approx(600.0 + 2800.0 * 1.5 / x)
    assert depth[0] == pytest.approx(x)
    assert np.isnan(uv[1]).all() and np.isnan(depth[1])


def test_lidar_depth_image_keeps_the_nearest():
    points = flu_to_lidar([[10.0, 0.0, -1.0], [20.0, 0.0, -2.0]])  # both land at pixel (960, 880)
    nearest = lidar_depth_image(points, T_CAM_FROM_LIDAR, K, (1200, 1920), block=4)
    assert nearest[880 // 4, 960 // 4] == pytest.approx(10.0)
    assert np.isinf(nearest[0, 0])


def test_shares():
    lidar = no_lidar()
    lidar[CELL_15M] = GRASS
    other = SPEC.cell_index(20.0, 0.0)
    result = truth([CELL_15M, other, CELL_3M], lidar_labels=lidar, image=label_image(PUDDLE))
    assert result.source_shares() == pytest.approx({"unseen": 1 / 3, "lidar": 1 / 3, "camera": 1 / 3, "occluded": 0.0})
    assert result.class_shares() == pytest.approx({UNSEEN: 1 / 3, GRASS: 1 / 3, PUDDLE: 1 / 3})
