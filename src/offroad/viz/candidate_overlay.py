"""Draw DriveVLN-style colored candidate trajectories onto a camera image (pure, no file I/O)."""

import cv2
import numpy as np

from offroad.data.calibration import CameraCalibration
from offroad.geometry.projection import project_points_to_image, transform_points
from offroad.geometry.transforms import invert_transform
from offroad.mapping.elevation import T_FLU_FROM_LIDAR
from offroad.planning.candidates import CandidatePath

# High-contrast RGB colors for candidate paths 
COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "red": (235, 45, 45),
    "blue": (45, 120, 255),
    "yellow": (250, 220, 30),
    "green": (40, 210, 80),
}


def project_candidate_to_image(
    candidate: CandidatePath,
    ground_z: np.ndarray,
    calib: CameraCalibration,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Project a CandidatePath onto the camera image plane.

    ground_z: (nx, ny) estimated ground height in FLU meters (NaN where unknown).
    image_shape: (H, W) of the RGB camera image.
    Returns uv: (M, 2) float pixel coordinates of the visible path points.
    """
    if not candidate.path:
        return np.zeros((0, 2), dtype=float)

    z = np.array([ground_z[c] for c in candidate.path], dtype=float)
    known = ~np.isnan(z)
    if not known.any():
        return np.zeros((0, 2), dtype=float)

    steps = np.arange(len(z))
    z = np.interp(steps, steps[known], z[known])
    points_flu = np.column_stack([candidate.path_xy(), z])
    points_lidar = transform_points(invert_transform(T_FLU_FROM_LIDAR), points_flu)
    uv, _, _ = project_points_to_image(points_lidar, calib.T_cam_from_lidar, calib.K, image_shape)
    return uv


def draw_candidates_on_camera(
    image: np.ndarray,
    calib: CameraCalibration,
    ground_z: np.ndarray,
    candidates: list[CandidatePath],
    thickness: int = 8,
    outline_thickness: int = 12,
    alpha: float = 0.85,
) -> tuple[np.ndarray, list[str]]:
    """Draw colored candidate trajectories onto a copy of an (H, W, 3) RGB image.

    Returns (overlay_rgb, visible_colors) where `visible_colors` lists the color names
    of candidates that had at least one point project inside the image.
    """
    canvas = image.copy()
    visible_colors: list[str] = []

    # Draw in reverse order so the first (nominal) candidate renders cleanly on top if paths overlap.
    projected: list[tuple[CandidatePath, np.ndarray]] = []
    for cand in candidates:
        uv = project_candidate_to_image(cand, ground_z, calib, image.shape[:2])
        if len(uv) > 0:
            visible_colors.append(cand.color)
            projected.append((cand, uv))

    for cand, uv in reversed(projected):
        pts = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
        rgb = COLOR_RGB.get(cand.color, (255, 255, 255))
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], isClosed=False, color=(15, 15, 15), thickness=outline_thickness, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [pts], isClosed=False, color=rgb, thickness=thickness, lineType=cv2.LINE_AA)
        else:
            center = (int(pts[0, 0, 0]), int(pts[0, 0, 1]))
            cv2.circle(canvas, center, max(2, outline_thickness // 2), (15, 15, 15), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, center, max(1, thickness // 2), rgb, -1, lineType=cv2.LINE_AA)

    if 0.0 <= alpha < 1.0:
        blended = cv2.addWeighted(canvas, alpha, image, 1.0 - alpha, 0.0)
        return blended, visible_colors
    return canvas, visible_colors