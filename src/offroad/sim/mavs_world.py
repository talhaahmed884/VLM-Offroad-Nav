"""A thin wrapper around MAVS for phase 6: trail scenes, the Warthog, and its camera and LiDAR.

MAVS is loaded from its own checkout (not pip): set MAVS_ROOT to the MAVS folder, or keep the
default ~/source/repos/MAVS. Its Python API (src/mavs_python/mavs_interface.py) calls mavs.dll
through ctypes, so it runs in our conda environment. Everything that touches MAVS is here; the
rig maths is in sim/rig.py (pure, tested without MAVS).

Units are meters, seconds and radians; world coordinates are MAVS's local ENU (x east, y north,
z up); orientations are [w, x, y, z] quaternions.
"""

import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from offroad.sim.rig import RigSpec, mavs_camera_args, sensor_offsets, to_lidar_axes, yaw_from_quaternion

IDENTITY_Q = [1.0, 0.0, 0.0, 0.0]


def mavs_root() -> Path:
    return Path(os.environ.get("MAVS_ROOT", Path.home() / "source" / "repos" / "MAVS"))


def import_mavs():
    """The mavs_interface module (adds MAVS's Python folder to sys.path once)."""
    folder = mavs_root() / "src" / "mavs_python"
    if not (folder / "mavs_interface.py").exists():
        raise FileNotFoundError(f"No MAVS Python API in {folder}; set MAVS_ROOT to your MAVS checkout")
    if str(folder) not in sys.path:
        sys.path.append(str(folder))
    return importlib.import_module("mavs_interface")


def mavs_data_path() -> Path:
    import_mavs()
    return Path(importlib.import_module("mavs_python_paths").mavs_data_path)


@dataclass(frozen=True)
class TrailSceneSpec:
    """A generated forest trail (MavsRandomScene). Sizes in meters.

    The trail is saved as waypoints; MAVS's 'Ridges' and 'Valleys' trails cross the terrain,
    'Loop' goes around it.
    """

    terrain_size_m: float = 150.0
    hills_m: float = 2.0  # low-frequency roughness (rolling hills)
    bumps_m: float = 0.05  # high-frequency roughness
    mesh_resolution_m: float = 0.5
    trail_width_m: float = 4.0
    track_width_m: float = 0.3
    wheelbase_m: float = 1.6
    path_type: str = "Ridges"
    plant_density: float = 0.3
    # The only ecosystem whose plant meshes all exist in the user's MAVS data (2026-11-24): most
    # vegetation folders (USA_Conifers, Europe, Groundcover, ...) are not in the checkout.
    eco_file: str = "forest_simple.json"


def drop_missing_meshes(scene_json: Path) -> list[str]:
    """Remove objects whose mesh file doesn't exist from a MAVS scene file; returns the removed meshes.

    One missing mesh gives the scene an invalid bounding box, and then MAVS's ground-height query
    (used by the vehicle's tires) finds no ground at all (checked 2026-11-24).
    """
    data = json.loads(scene_json.read_text(encoding="utf-8"))
    root = Path(data.get("PathToMeshes", ""))
    kept, removed = [], []
    for obj in data.get("Objects", []):
        mesh = Path(obj["Mesh"])
        (kept if (mesh if mesh.is_absolute() else root / mesh).exists() else removed).append(obj)
    if removed:
        data["Objects"] = kept
        scene_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return [obj["Mesh"] for obj in removed]


def generate_trail_scene(out_dir: Path, name: str, spec: TrailSceneSpec) -> Path:
    """Generate a trail scene into out_dir (slow: MAVS grows the forest); returns <name>_scene.json.

    MAVS also writes <name>.obj (the ground) and <name>_path.vprp (the trail waypoints). Load the
    saved scene afterwards, so every run of every chooser drives the same trail.
    """
    mavs = import_mavs()
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = mavs.MavsRandomScene()
    scene.terrain_width = scene.terrain_length = spec.terrain_size_m
    scene.lo_mag, scene.hi_mag = spec.hills_m, spec.bumps_m
    scene.mesh_resolution = spec.mesh_resolution_m
    scene.trail_width, scene.track_width, scene.wheelbase = spec.trail_width_m, spec.track_width_m, spec.wheelbase_m
    scene.path_type = spec.path_type
    scene.plant_density = spec.plant_density
    scene.eco_file = spec.eco_file
    scene.basename = name
    scene.output_directory = str(out_dir) + os.sep
    cwd = os.getcwd()
    try:  # some MAVS versions write next to the working directory
        os.chdir(out_dir)
        scene.CreateScene()
    finally:
        os.chdir(cwd)
    path = out_dir / f"{name}_scene.json"
    if not path.exists():
        raise FileNotFoundError(f"MAVS did not write {path}")
    removed = drop_missing_meshes(path)
    if removed:
        print(f"Removed {len(removed)} object type(s) with missing meshes from {path.name}: {removed}")
    return path


def ground_only_scene(scene_json: Path) -> Path:
    """A copy of the scene without its objects (plants, obstacles), for the vehicle physics.

    Written next to the scene as <name>_ground.json (once).
    """
    out = Path(str(scene_json).replace("_scene.json", "_ground.json"))
    if not out.exists():
        data = json.loads(Path(scene_json).read_text(encoding="utf-8"))
        data.pop("Objects", None)
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


@dataclass
class Capture:
    """One sensor frame.

    image: (H, W, 3) uint8 RGB camera image.
    points_lidar: (N, 3) LiDAR returns in the RELLIS Ouster axes at the LiDAR (see sim/rig.py).
    labels: (N,) MAVS label of each return (-1 if labeling is off).
    position: (3,) vehicle CG in world ENU; yaw: heading in radians (0 = +x).
    """

    image: np.ndarray
    points_lidar: np.ndarray
    labels: np.ndarray
    position: np.ndarray
    yaw: float


class MavsWorld:
    """A loaded scene with the Warthog, its camera and LiDAR (RigSpec), and the trail waypoints."""

    def __init__(self, scene_json: Path, rig: RigSpec = RigSpec(), labeling: bool = True, hour: int = 12):
        mavs = import_mavs()
        self.mavs, self.rig = mavs, rig
        # The sensors see the full scene (self.env); the vehicle drives on the bare ground
        # (self.ground_env). With the plants in it, MAVS's vehicle physics ran 30x slower
        # (7.5 vs 0.26 s per simulated second, 2026-11-24); in MAVS plants only add rolling
        # resistance, they are not solid, so leaving them out of the physics changes little.
        self.scene = mavs.MavsEmbreeScene()
        self.scene.Load(str(scene_json))
        if labeling:
            self.scene.TurnOnLabeling()
        self.labeling = labeling
        self.env = mavs.MavsEnvironment()
        self.env.SetScene(self.scene.scene)
        self.env.SetTime(hour)
        self.env.SetTurbidity(3.0)
        self.env.SetCloudCover(0.3)
        self.ground_scene = mavs.MavsEmbreeScene()
        self.ground_scene.Load(str(ground_only_scene(scene_json)))
        self.ground_env = mavs.MavsEnvironment()
        self.ground_env.SetScene(self.ground_scene.scene)

        self.vehicle = mavs.MavsRp3d()
        self.vehicle.Load(str(mavs_data_path() / "vehicles" / "rp3d_vehicles" / rig.vehicle_file))

        self.camera = mavs.MavsCamera()
        self.camera.Initialize(*mavs_camera_args(rig))
        self.camera.SetGammaAndGain(0.6, 1.0)
        self.camera.RenderShadows(True)
        self.camera.SetEnvironmentProperties(self.env.obj)
        self.lidar = mavs.MavsLidar(rig.lidar_model)

        vprp = Path(str(scene_json).replace("_scene.json", "_path.vprp"))
        self.trail = np.zeros((0, 2))
        if vprp.exists():
            wp = mavs.MavsWaypoints()
            wp.Load(str(vprp))
            self.trail = np.array([p[:2] for p in wp.waypoints], dtype=float)
        self.cg_height_m = None

    def ground_height(self, x: float, y: float) -> float:
        return float(self.ground_scene.GetSurfaceHeight(x, y))

    def place(self, x: float, y: float, yaw: float, settle_s: float = 2.0, dt: float = 0.02) -> None:
        """Put the vehicle at (x, y) facing yaw, let it settle with no pedals, then mount the sensors.

        The sensors are mounted from the settled CG height, so the LiDAR sits rig.lidar_height_m
        above the ground as on RELLIS.
        """
        self.vehicle.SetInitialPosition(x, y, self.ground_height(x, y) + 0.5)
        self.vehicle.SetInitialHeading(yaw)
        # No pedals while settling: in this MAVS build "brake" from standstill drives backwards
        # (checked 2026-11-24, Warthog and MRZR4), so the brake is only used while moving forward.
        self.vehicle.Update(self.ground_env, 0.0, 0.0, 0.0, 1e-6)
        for _ in range(int(settle_s / dt)):
            self.vehicle.Update(self.ground_env, 0.0, 0.0, 0.0, dt)
        p = self.vehicle.GetPosition()
        self.cg_height_m = float(p[2]) - self.ground_height(p[0], p[1])
        offsets = sensor_offsets(self.rig, self.cg_height_m)
        self.lidar.SetOffset(offsets["lidar"], IDENTITY_Q)
        self.camera.SetOffset(offsets["camera"], IDENTITY_Q)

    def step(self, throttle: float, steering: float, braking: float, dt: float) -> None:
        self.vehicle.Update(self.ground_env, throttle, steering, braking, dt)
        self.env.AdvanceTime(dt)

    def pose(self) -> tuple[np.ndarray, list[float]]:
        return np.array(self.vehicle.GetPosition(), dtype=float), list(self.vehicle.GetOrientation())

    def heading_and_speed(self) -> tuple[float, float]:
        """(heading in radians, forward speed in m/s) of the vehicle."""
        look = self.vehicle.GetLookTo()
        heading = float(np.arctan2(look[1], look[0]))
        vel = self.vehicle.GetVelocity()
        return heading, float(vel[0] * np.cos(heading) + vel[1] * np.sin(heading))

    def capture(self, dt: float = 0.1) -> Capture:
        """Render the camera and scan the LiDAR at the vehicle's current pose."""
        position, orientation = self.pose()
        self.camera.SetPose(list(position), orientation)
        self.camera.Update(self.env, dt)
        image = np.ascontiguousarray(self.camera.GetNumpyArray(), dtype=np.uint8)
        self.lidar.SetPose(list(position), orientation)
        self.lidar.Update(self.env, dt)
        if self.labeling:
            self.lidar.AnnotateFrame(self.env)
        xyzil = np.array(self.lidar.GetUnRegisteredPointsXYZIL(), dtype=float).reshape(-1, 5)
        return Capture(image=image, points_lidar=to_lidar_axes(xyzil[:, :3]),
                       labels=xyzil[:, 4].astype(int) if self.labeling else np.full(len(xyzil), -1),
                       position=position, yaw=yaw_from_quaternion(orientation))
