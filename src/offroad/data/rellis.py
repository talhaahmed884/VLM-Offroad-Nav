"""Load RELLIS-3D frames: RGB image, Ouster LiDAR scan, labels, and calibration.

Expected layout under the dataset root (see CLAUDE.md, "Dataset on disk"):

    Rellis-3D/
        Rellis_3D_ontology/ontology.yaml
        00000/
            camera_info.txt, transforms.yaml
            pylon_camera_node/frame000123-1581624652_750.jpg
            pylon_camera_node_label_id/frame000123-1581624652_750.png   (about every other frame)
            os1_cloud_node_kitti_bin/000123.bin
            os1_cloud_node_semantickitti_label_id/000123.label

Images and scans are matched by frame index (the number after "frame").
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from offroad.data.calibration import CameraCalibration, load_calibration

IMAGE_DIR = "pylon_camera_node"
IMAGE_LABEL_DIR = "pylon_camera_node_label_id"
LIDAR_DIR = "os1_cloud_node_kitti_bin"
LIDAR_LABEL_DIR = "os1_cloud_node_semantickitti_label_id"
ONTOLOGY_PATH = Path("Rellis_3D_ontology") / "ontology.yaml"

_IMAGE_NAME = re.compile(r"^frame(\d+)-.*\.jpg$")


def default_rellis_root() -> Path:
    """Return the dataset root: $RELLIS_ROOT if set, else <repo>/data/Rellis-3D."""
    env_root = os.environ.get("RELLIS_ROOT")
    if env_root:
        return Path(env_root)
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "Rellis-3D"


@dataclass
class RellisFrame:
    """Everything for one moment in time.

    image: (H, W, 3) uint8, RGB.
    points: (N, 4) float32, x, y, z (meters) and intensity, in the Ouster LiDAR frame.
        Empty returns at (0, 0, 0) are removed unless the loader was asked to keep them.
    point_labels: (N,) uint16 class IDs, one per row of `points`.
    label_image: (H, W) uint8 class IDs, or None if this frame has no image annotation.
    calib: camera intrinsics and T_cam_from_lidar for this sequence.
    """

    sequence: str
    index: int
    image: np.ndarray
    points: np.ndarray
    point_labels: np.ndarray
    label_image: np.ndarray | None
    calib: CameraCalibration


def load_lidar_bin(path: Path) -> np.ndarray:
    """Read a KITTI-style .bin scan into an (N, 4) float32 array (x, y, z, intensity)."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def load_lidar_labels(path: Path) -> np.ndarray:
    """Read a SemanticKITTI .label file into (N,) uint16 class IDs.

    Each entry is a uint32 whose lower 16 bits are the class ID and upper 16 bits
    an instance ID. RELLIS-3D does not use instance IDs, but we mask them off anyway.
    """
    raw = np.fromfile(path, dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.uint16)


def load_ontology(root: Path | None = None) -> tuple[dict[int, str], dict[int, tuple[int, int, int]]]:
    """Read ontology.yaml and return ({class_id: name}, {class_id: (r, g, b)})."""
    root = Path(root) if root is not None else default_rellis_root()
    with open(root / ONTOLOGY_PATH) as f:
        id_to_name, id_to_color = yaml.safe_load(f)
    colors = {class_id: tuple(rgb) for class_id, rgb in id_to_color.items()}
    return id_to_name, colors


class RellisSequence:
    """One RELLIS-3D sequence. `seq[i]` returns the RellisFrame with frame index i."""

    def __init__(self, sequence: str | int, root: Path | None = None, drop_empty_points: bool = True):
        self.root = Path(root) if root is not None else default_rellis_root()
        self.sequence = f"{int(sequence):05d}"
        self.sequence_dir = self.root / self.sequence
        self.drop_empty_points = drop_empty_points
        if not self.sequence_dir.is_dir():
            raise FileNotFoundError(
                f"Sequence folder not found: {self.sequence_dir}. Set RELLIS_ROOT to the Rellis-3D folder."
            )

        self.calib = load_calibration(self.sequence_dir)
        self._image_paths = self._index_images()

    def _index_images(self) -> dict[int, Path]:
        """Map frame index -> image path, parsed from names like frame000123-<timestamp>.jpg."""
        image_paths = {}
        for path in (self.sequence_dir / IMAGE_DIR).iterdir():
            match = _IMAGE_NAME.match(path.name)
            if match:
                image_paths[int(match.group(1))] = path
        return dict(sorted(image_paths.items()))

    @property
    def frame_indices(self) -> list[int]:
        """All frame indices in this sequence, sorted."""
        return list(self._image_paths)

    def __len__(self) -> int:
        return len(self._image_paths)

    def image_path(self, index: int) -> Path:
        """Path of this frame's RGB image."""
        return self._image_paths[index]

    def label_image_path(self, index: int) -> Path:
        """Path where this frame's semantic label PNG would be (it may not exist)."""
        image_name = self._image_paths[index].stem
        return self.sequence_dir / IMAGE_LABEL_DIR / f"{image_name}.png"

    def has_label_image(self, index: int) -> bool:
        """True if this frame has a semantic label image (about every other frame does)."""
        return self.label_image_path(index).exists()

    def __getitem__(self, index: int) -> RellisFrame:
        if index not in self._image_paths:
            raise IndexError(f"Frame {index} not in sequence {self.sequence}")

        image_bgr = cv2.imread(str(self._image_paths[index]), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        points = load_lidar_bin(self.sequence_dir / LIDAR_DIR / f"{index:06d}.bin")
        point_labels = load_lidar_labels(self.sequence_dir / LIDAR_LABEL_DIR / f"{index:06d}.label")
        if self.drop_empty_points:
            # About 40% of each scan are "no return" slots stored as (0, 0, 0).
            has_return = np.any(points[:, :3] != 0, axis=1)
            points = points[has_return]
            point_labels = point_labels[has_return]

        label_image = None
        if self.has_label_image(index):
            label_image = cv2.imread(str(self.label_image_path(index)), cv2.IMREAD_UNCHANGED)

        return RellisFrame(
            sequence=self.sequence,
            index=index,
            image=image,
            points=points,
            point_labels=point_labels,
            label_image=label_image,
            calib=self.calib,
        )
