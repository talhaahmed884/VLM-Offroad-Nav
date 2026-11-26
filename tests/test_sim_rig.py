"""Tests for the simulated sensor rig (no MAVS needed)."""

import numpy as np
import pytest

from offroad.geometry.projection import project_points_to_image
from offroad.mapping.elevation import lidar_to_flu
from offroad.sim.rig import (
    RELLIS_FX_PX,
    RigSpec,
    camera_calibration,
    camera_K,
    mavs_camera_args,
    sensor_offsets,
    to_lidar_axes,
    yaw_from_quaternion,
)


def test_camera_keeps_the_rellis_field_of_view():
    rig = RigSpec()
    assert rig.image_size == (960, 600)
    nx, ny, dw, dh, f = mavs_camera_args(rig)
    assert (nx, ny) == (960, 600)
    assert f / (dw / nx) == pytest.approx(RELLIS_FX_PX / 2)  # focal length in pixels
    half_fov = np.degrees(np.arctan(nx / 2 / (f / (dw / nx))))
    assert half_fov == pytest.approx(np.degrees(np.arctan(960 / RELLIS_FX_PX)))
    K = camera_K(rig)
    assert K[0, 2] == 480 and K[1, 2] == 300


def test_sensor_offsets_put_the_lidar_at_rellis_height():
    off = sensor_offsets(RigSpec(), cg_height_m=0.41)
    assert off["lidar"][2] == pytest.approx(1.2 - 0.41)
    assert np.subtract(off["camera"], off["lidar"]) == pytest.approx([0.1, 0.0, -0.17])


def test_mavs_points_go_to_our_lidar_axes_and_back_to_flu():
    mavs = np.array([[5.0, 1.0, -1.2, 0.3, 7.0]])  # x forward, y left, z up + intensity, label
    ours = to_lidar_axes(mavs)
    assert ours[0, :3] == pytest.approx([-5.0, -1.0, -1.2]) and ours[0, 3:] == pytest.approx([0.3, 7.0])
    assert lidar_to_flu(ours)[0] == pytest.approx([5.0, 1.0, -1.2])


def test_calibration_projects_forward_points_correctly():
    rig = RigSpec()
    cal = camera_calibration(rig)
    w, h = rig.image_size
    # A point straight ahead at the camera's own height lands on the image centre;
    # a point to the left lands left of it, a point on the ground below it.
    cam = np.array(rig.camera_from_lidar_m)
    flu = np.array([[20.0, 0.0, cam[2]], [20.0, 2.0, cam[2]], [20.0, 0.0, -1.2]])
    uv, depth, kept = project_points_to_image(to_lidar_axes(flu), cal.T_cam_from_lidar, cal.K, (h, w))
    assert kept.all()
    assert uv[0] == pytest.approx([w / 2, h / 2])
    assert uv[1, 0] < w / 2 and uv[2, 1] > h / 2
    assert depth[0] == pytest.approx(20.0 - cam[0])


def test_yaw_from_quaternion():
    for yaw in (0.0, 0.8, -2.5):
        q = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        assert yaw_from_quaternion(q) == pytest.approx(yaw)
