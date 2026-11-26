"""The simulated vehicle's sensor rig, set up like RELLIS-3D's (pure, no MAVS needed).

RELLIS-3D was recorded with a Clearpath Warthog, an Ouster OS1 LiDAR about 1.2 m above the
ground, and a camera about 0.1 m in front of and 0.17 m below the LiDAR, level, with a narrow
field of view (1920 x 1200 pixels, fx = 2813.6 px: about 38 x 24 degrees). The fine-tuned
selector was trained on that view, so the simulated sensors copy it. The camera image is rendered
at half resolution (960 x 600, same field of view): the selector's picture is 800 px wide anyway.

The vehicle is MAVS's MRZR4 (user decision 2026-11-24): MAVS's Warthog model reports a heading
about 15 degrees off the direction it moves, even on bare ground, so it can't follow a path and
its camera would look to the side. Only the body changes; the sensors stay RELLIS-like.

Frames:
- MAVS: the vehicle frame and the sensor frames are x forward, y left, z up (ENU world; heading
  0 = +x). A sensor's offset is given from the vehicle's centre of gravity (CG).
- Our code (phases 1-5) takes LiDAR points in the RELLIS Ouster's axes (forward = -x, left = -y,
  `mapping.elevation.T_FLU_FROM_LIDAR`) and a `CameraCalibration` whose T_cam_from_lidar starts
  from those axes. `to_lidar_axes` turns MAVS sensor points into those axes, and
  `camera_calibration` builds the matching calibration, so the phase 2-3 code runs unchanged.
- Camera optical frame (OpenCV): x right, y down, z forward.
"""

from dataclasses import dataclass

import numpy as np

from offroad.data.calibration import CameraCalibration
from offroad.geometry.transforms import invert_transform, make_transform
from offroad.mapping.elevation import T_FLU_FROM_LIDAR

RELLIS_FX_PX = 2813.64  # RELLIS-3D camera, 1920 x 1200 (camera_info.txt)
RELLIS_IMAGE_SIZE = (1920, 1200)

# Optical axes (x right, y down, z forward) expressed from FLU axes (x forward, y left, z up).
R_OPTICAL_FROM_FLU = np.array([[0.0, -1.0, 0.0],
                               [0.0, 0.0, -1.0],
                               [1.0, 0.0, 0.0]])


@dataclass(frozen=True)
class RigSpec:
    """Vehicle and sensors. Lengths in meters; offsets in the vehicle's FLU axes.

    lidar_height_m: LiDAR above the ground (RELLIS: about 1.2 m).
    camera_from_lidar_m: camera position relative to the LiDAR (RELLIS 00000 / 00004).
    image_size: (width, height) of the rendered camera image, pixels.
    scale: rendered size relative to RELLIS's 1920 x 1200 (the field of view stays the same).
    pixel_pitch_m: size of one detector pixel; only the ratio focal / pitch matters.
    """

    vehicle_file: str = "mrzr4_tires_low_gear.json"
    lidar_model: str = "OS1"
    lidar_height_m: float = 1.2
    lidar_forward_m: float = 0.0
    camera_from_lidar_m: tuple[float, float, float] = (0.1, 0.0, -0.17)
    scale: float = 0.5
    pixel_pitch_m: float = 3.0e-6

    @property
    def image_size(self) -> tuple[int, int]:
        return round(RELLIS_IMAGE_SIZE[0] * self.scale), round(RELLIS_IMAGE_SIZE[1] * self.scale)

    @property
    def fx_px(self) -> float:
        return RELLIS_FX_PX * self.scale


def camera_K(rig: RigSpec) -> np.ndarray:
    """(3, 3) pinhole intrinsics of the rendered image (square pixels, centred principal point)."""
    w, h = rig.image_size
    return np.array([[rig.fx_px, 0.0, w / 2.0], [0.0, rig.fx_px, h / 2.0], [0.0, 0.0, 1.0]])


def mavs_camera_args(rig: RigSpec) -> tuple[int, int, float, float, float]:
    """Arguments of MavsCamera.Initialize: (nx, ny, detector width, detector height, focal length), meters."""
    w, h = rig.image_size
    return w, h, w * rig.pixel_pitch_m, h * rig.pixel_pitch_m, rig.fx_px * rig.pixel_pitch_m


def sensor_offsets(rig: RigSpec, cg_height_m: float) -> dict[str, list[float]]:
    """MAVS SetOffset positions (from the CG, vehicle FLU axes) for "lidar" and "camera".

    cg_height_m: the vehicle's centre of gravity above the ground, measured once it has settled.
    Both sensors look straight ahead (identity orientation).
    """
    lidar = [rig.lidar_forward_m, 0.0, rig.lidar_height_m - cg_height_m]
    camera = [lidar[i] + rig.camera_from_lidar_m[i] for i in range(3)]
    return {"lidar": lidar, "camera": camera}


def camera_calibration(rig: RigSpec) -> CameraCalibration:
    """Calibration in our conventions: T_cam_from_lidar starts from the RELLIS Ouster axes at the LiDAR."""
    c = np.asarray(rig.camera_from_lidar_m, dtype=float)
    T_cam_from_flu = make_transform(R_OPTICAL_FROM_FLU, -R_OPTICAL_FROM_FLU @ c)
    return CameraCalibration(K=camera_K(rig), dist_coeffs=np.zeros(5),
                             T_cam_from_lidar=T_cam_from_flu @ T_FLU_FROM_LIDAR)


def to_lidar_axes(points_sensor: np.ndarray) -> np.ndarray:
    """MAVS sensor-frame points (N, 3+) in FLU axes -> the RELLIS Ouster axes our code expects.

    Extra columns (intensity, label) are kept.
    """
    out = np.array(points_sensor, dtype=float, copy=True)
    T = invert_transform(T_FLU_FROM_LIDAR)
    out[:, :3] = out[:, :3] @ T[:3, :3].T
    return out


def yaw_from_quaternion(q_wxyz) -> float:
    """Heading (radians, 0 = +x, counter-clockwise) of a [w, x, y, z] orientation."""
    w, x, y, z = q_wxyz
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
