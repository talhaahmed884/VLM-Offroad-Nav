"""Unit tests for the colored candidate trajectory camera overlay."""

import numpy as np

from offroad.data.calibration import CameraCalibration
from offroad.mapping.elevation import GridSpec
from offroad.planning.candidates import CandidatePath
from offroad.viz.candidate_overlay import (
    draw_candidates_on_camera,
    project_candidate_to_image,
)


def _synthetic_calib() -> CameraCalibration:
    """Pinhole camera looking straight along FLU +x (LiDAR -x), mounted at z=0."""
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=float)
    # Camera frame: +z = forward (LiDAR -x), +x = right (LiDAR +y), +y = down (LiDAR -z).
    T_cam_from_lidar = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return CameraCalibration(K=K, dist_coeffs=np.zeros(5), T_cam_from_lidar=T_cam_from_lidar)


def _make_candidate(spec: GridSpec, color: str, y_m: float) -> CandidatePath:
    cells = [spec.cell_index(x, y_m) for x in np.linspace(4.0, 14.0, 11)]
    return CandidatePath(
        color=color,
        kind="geometric",
        lateral_offset_m=y_m,
        path=cells,
        total_cost=10.0,
        start=cells[0],
        goal=cells[-1],
        target_xy=(14.0, y_m),
        spec=spec,
    )


def test_project_candidate_to_image_projects_visible_ground_points() -> None:
    spec = GridSpec(x_min=0.0, x_max=20.0, y_min=-10.0, y_max=10.0, resolution=0.5)
    ground_z = np.full(spec.shape, -1.5, dtype=float)
    calib = _synthetic_calib()
    cand = _make_candidate(spec, "red", y_m=0.0)

    uv = project_candidate_to_image(cand, ground_z, calib, (480, 640))
    assert len(uv) == len(cand.path)
    # Straight ahead (y=0) should project near the horizontal principal point u=320.
    assert np.allclose(uv[:, 0], 320.0, atol=35.0)
    assert uv[-1, 0] > uv[0, 0]


def test_draw_candidates_on_camera_modifies_image_and_reports_visible_colors() -> None:
    spec = GridSpec(x_min=0.0, x_max=20.0, y_min=-10.0, y_max=10.0, resolution=0.5)
    ground_z = np.full(spec.shape, -1.5, dtype=float)
    calib = _synthetic_calib()
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    cands = [
        _make_candidate(spec, "red", y_m=0.0),
        _make_candidate(spec, "blue", y_m=1.5),
    ]
    overlay, visible = draw_candidates_on_camera(image, calib, ground_z, cands, alpha=1.0)
    assert visible == ["red", "blue"]
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert overlay.sum() > 0


def test_draw_candidates_skips_path_with_all_nan_ground() -> None:
    spec = GridSpec(x_min=0.0, x_max=20.0, y_min=-10.0, y_max=10.0, resolution=0.5)
    ground_z = np.full(spec.shape, np.nan, dtype=float)
    calib = _synthetic_calib()
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    cands = [_make_candidate(spec, "red", y_m=0.0)]
    overlay, visible = draw_candidates_on_camera(image, calib, ground_z, cands)
    assert visible == []
    assert np.array_equal(overlay, image)