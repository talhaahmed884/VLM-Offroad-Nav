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
from offroad.sim.scenes import (BLOCKADE_HEIGHT_SCALE, BLOCKADE_MESH, BLOCKADE_THICKNESS_M, Blockade, SceneDesign,
                                near_any, vprp_text)

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


# Solid things in a scene, for the collision check (MAVS's vehicle drives through them, because
# the physics runs on the bare ground): tree trunks by mesh (approximate trunk radii, m) and the
# step 6.4 blockades (sampled along their length, radius = half their thickness).
TRUNK_RADIUS_M = {"vegetation/oak/": 0.4, "vegetation/pine_tree/": 0.3, "vegetation/honey_locust/": 0.3,
                  "vegetation/weeping_cherry/": 0.3}


def scene_obstacles(scene_json: Path) -> tuple[np.ndarray, np.ndarray]:
    """(points (N, 2) world xy, radii (N,)) of the trunks and blockades in a scene file."""
    data = json.loads(Path(scene_json).read_text(encoding="utf-8"))
    points, radii = [], []
    for obj in data.get("Objects", []):
        mesh = obj["Mesh"].replace("\\", "/")
        radius = next((r for key, r in TRUNK_RADIUS_M.items() if key in mesh), None)
        if radius is not None:
            for inst in obj["Instances"]:
                points.append(inst["Position"][:2])
                radii.append(radius * float(inst.get("Scale", [1.0])[0]))
        elif mesh == BLOCKADE_MESH:
            for inst in obj["Instances"]:
                b = Blockade(inst["Position"][0], inst["Position"][1], inst["YawPitchRoll"][0])
                for p in b.footprint_points():
                    points.append(p)
                    radii.append(BLOCKADE_THICKNESS_M / 2)
    return np.array(points, dtype=float).reshape(-1, 2), np.array(radii, dtype=float)


def build_designed_scene(design: SceneDesign, base_scene_json: Path, out_dir: Path,
                         trail_width_m: float = 4.0) -> Path:
    """Draw a step 6.4 design into a copy of a generated forest; returns <name>_scene.json.

    The trails become the scene's dirt texture (one MAVS path file each), plants are removed
    within 2.5 m of the trail centre (trees within 3.5 m, so no trunk stands on the trail edge),
    and the blockades are added as 1.8 m walls. Also written: <name>_design.json (routes and
    blockades: the answer key) and <name>_path.vprp (the first route).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(base_scene_json).read_text(encoding="utf-8"))
    paths = []
    for trail_name, points in design.trails.items():
        path = out_dir / f"{design.name}_trail_{trail_name}.vprp"
        path.write_text(vprp_text(points), encoding="utf-8")
        paths.append(str(path))
    data["Layered Surface"]["Trail"]["Path"] = paths
    data["Layered Surface"]["Trail"]["Trail Width"] = trail_width_m
    lines = list(design.trails.values())
    for obj in data.get("Objects", []):
        is_tree = any(key in obj["Mesh"].replace("\\", "/") for key in TRUNK_RADIUS_M)
        positions = np.array([inst["Position"][:2] for inst in obj["Instances"]]).reshape(-1, 2)
        if len(positions):
            near = near_any(positions, lines, 3.5 if is_tree else 2.5)
            obj["Instances"] = [inst for inst, n in zip(obj["Instances"], near) if not n]
    if design.blockades:
        data.setdefault("Objects", []).append({
            "Mesh": BLOCKADE_MESH, "Rotate Y to Z": True,
            "Instances": [{"YawPitchRoll": [b.yaw, 0.0, 0.0], "Position": [b.x, b.y, 0.0],
                           # MAVS scales after turning the mesh's y (up) to z: height is the 3rd value
                           "Scale": [1.0, 1.0, BLOCKADE_HEIGHT_SCALE]} for b in design.blockades]})
    scene = out_dir / f"{design.name}_scene.json"
    scene.write_text(json.dumps(data), encoding="utf-8")
    (out_dir / f"{design.name}_path.vprp").write_text(vprp_text(design.routes[design.default_route]), encoding="utf-8")
    (out_dir / f"{design.name}_design.json").write_text(json.dumps({
        "name": design.name, "note": design.note, "default_route": design.default_route,
        "routes": {k: np.round(v, 3).tolist() for k, v in design.routes.items()},
        "blockades": [{"x": b.x, "y": b.y, "yaw": b.yaw} for b in design.blockades]}, indent=1), encoding="utf-8")
    ground = Path(str(scene).replace("_scene.json", "_ground.json"))
    if ground.exists():  # rebuilt from the new scene on the next load
        ground.unlink()
    return scene


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
    """A loaded scene with the vehicle, its camera and LiDAR (RigSpec), the trail (answer key) and
    the solid obstacles for the collision check (`obstacles`: points (N, 2) and radii (N,))."""

    def __init__(self, scene_json: Path, rig: RigSpec = RigSpec(), labeling: bool = True, hour: int = 12,
                 date: tuple[int, int, int] = (2026, 6, 21), cloud_cover: float = 0.0, route: str | None = None):
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
        # A fixed date, time (to the second) and no clouds make every run render the same pixels
        # (checked 2026-11-26). With MAVS's defaults the sun follows the computer's clock and the
        # clouds are random: two runs seconds apart differed by 5.8 grey levels on average, and
        # 3 of 24 VLM answers changed.
        self.env.SetDate(*date)
        self.env.SetTimeSeconds(hour, 0, 0)
        self.env.SetTurbidity(3.0)
        self.env.SetCloudCover(cloud_cover)
        self.ground_scene = mavs.MavsEmbreeScene()
        self.ground_scene.Load(str(ground_only_scene(scene_json)))
        self.ground_env = mavs.MavsEnvironment()
        self.ground_env.SetScene(self.ground_scene.scene)

        self.vehicle = None  # made by place()

        self.camera = mavs.MavsCamera()
        self.camera.Initialize(*mavs_camera_args(rig))
        self.camera.SetGammaAndGain(0.6, 1.0)
        self.camera.RenderShadows(True)
        self.camera.SetEnvironmentProperties(self.env.obj)
        self.lidar = mavs.MavsLidar(rig.lidar_model)

        # The trail that scores the run (the answer key): a designed scene's route (step 6.4), else
        # the trail MAVS generated.
        design = Path(str(scene_json).replace("_scene.json", "_design.json"))
        vprp = Path(str(scene_json).replace("_scene.json", "_path.vprp"))
        self.trail = np.zeros((0, 2))
        self.route = None
        if design.exists():
            routes = json.loads(design.read_text(encoding="utf-8"))["routes"]
            self.route = route or next(iter(routes))
            if self.route not in routes:
                raise ValueError(f"{design.name} has routes {sorted(routes)}, not {self.route!r}")
            self.trail = np.array(routes[self.route], dtype=float)
        elif vprp.exists():
            wp = mavs.MavsWaypoints()
            wp.Load(str(vprp))
            self.trail = np.array([p[:2] for p in wp.waypoints], dtype=float)
        self.obstacles = scene_obstacles(scene_json)
        self.cg_height_m = None

    def ground_height(self, x: float, y: float) -> float:
        return float(self.ground_scene.GetSurfaceHeight(x, y))

    def place(self, x: float, y: float, yaw: float, settle_s: float = 2.0, dt: float = 0.02) -> None:
        """Put the vehicle at (x, y) facing yaw, let it settle with no pedals, then mount the sensors.

        The sensors are mounted from the settled CG height, so the LiDAR sits rig.lidar_height_m
        above the ground as on RELLIS. A new vehicle is loaded every time: MAVS only takes the
        initial position before the vehicle's first update, so placing the same vehicle again
        left it where it was (found 2026-11-26).
        """
        self.vehicle = self.mavs.MavsRp3d()
        self.vehicle.Load(str(mavs_data_path() / "vehicles" / "rp3d_vehicles" / self.rig.vehicle_file))
        self.vehicle.SetInitialPosition(x, y, self.ground_height(x, y) + 0.5)
        self.vehicle.SetInitialHeading(yaw)
        # No pedals while settling: in this MAVS build "brake" from standstill drives backwards
        # (checked 2026-11-24, Warthog and MRZR4), so the brake is only used while moving forward.
        self.vehicle.Update(self.ground_env, 0.0, 0.0, 0.0, 1e-6)
        for _ in range(int(settle_s / dt)):
            self.vehicle.Update(self.ground_env, 0.0, 0.0, 0.0, dt)
        p = self.vehicle.GetPosition()
        if np.hypot(p[0] - x, p[1] - y) > 1.0:
            raise RuntimeError(f"the vehicle settled at ({p[0]:.1f}, {p[1]:.1f}), not at ({x:.1f}, {y:.1f})")
        self.cg_height_m = float(p[2]) - self.ground_height(p[0], p[1])
        offsets = sensor_offsets(self.rig, self.cg_height_m)
        self.lidar.SetOffset(offsets["lidar"], IDENTITY_Q)
        self.camera.SetOffset(offsets["camera"], IDENTITY_Q)

    def show_vehicle(self, params=None) -> None:
        """Add the vehicle's body and four wheels to the rendered scene as actors, and a chase camera.

        For videos only (sim/video.py). The vehicle normally lives in ground_env (its physics),
        so the sensors never see it, as on RELLIS. The actors are kept 500 m under the ground
        except while chase_view renders, so capture() sees exactly what it sees without them
        (checked 2026-11-28: identical image and LiDAR points). params: video.ChaseParams.
        """
        import tempfile

        from offroad.sim.video import ChaseParams, chase_K

        params = params or ChaseParams()
        vehicle = json.loads((mavs_data_path() / "vehicles" / "rp3d_vehicles" / self.rig.vehicle_file)
                             .read_text(encoding="utf-8"))
        folder = Path(tempfile.mkdtemp(prefix="mavs_actors_"))

        def add(mesh: dict, name: str) -> int:
            actor = {"Actors": [{"Mesh": mesh["File"], "Rotate Y to Z": mesh.get("Rotate Y to Z", False),
                                 "Rotate X to Y": mesh.get("Rotate X to Y", False),
                                 "Rotate Y to X": mesh.get("Rotate Y to X", False), "Locked to Ground": False,
                                 "Instances": [{"Initial Position": [0.0, 0.0, -500.0], "Offset": [0.0, 0.0, 0.0],
                                                "Scale": mesh.get("Scale", [1.0, 1.0, 1.0]), "Speed": 0.0}]}]}
            path = folder / f"{name}.json"
            path.write_text(json.dumps(actor), encoding="utf-8")
            return self.env.AddActor(str(path), False)

        self.body_actor = add(vehicle["Mesh"], "body")
        # Wheel centres from the CG: each axle's offset, half its track to each side, tyre radius up from the ground.
        self.wheels = [(add(vehicle["Tire Mesh"], f"wheel{2 * k + s}"), axle["Longitudinal Offset"],
                        side * axle["Track Width"] / 2, axle["Tire"]["Radius"])
                       for k, axle in enumerate(vehicle["Axles"]) for s, side in enumerate((1.0, -1.0))]
        w, h = params.size
        self.chase_args = (w, h, w * self.rig.pixel_pitch_m, h * self.rig.pixel_pitch_m,
                           chase_K(params)[0, 0] * self.rig.pixel_pitch_m)
        self.chase_camera, self.chase_renders = None, 0
        self.wheel_turn, self.last_xy = 0.0, None

    # MAVS's RGB camera exposes its first frames with a fixed factor, then scales every image to
    # the same mean brightness (auto-exposure over the last second, rgb_camera.cpp
    # ApplyElectronics), which washed out chase views without sky. A fresh chase camera every
    # few frames keeps the fixed exposure: shade looks like shade, sun like sun (2026-11-28).
    # The sensor camera (the VLM's input) is left as it is in every run.
    CHASE_FIXED_EXPOSURE_FRAMES = 5

    def _chase_camera(self):
        if self.chase_camera is None or self.chase_renders % self.CHASE_FIXED_EXPOSURE_FRAMES == 0:
            self.chase_camera = self.mavs.MavsCamera()
            self.chase_camera.Initialize(*self.chase_args)
            self.chase_camera.SetGammaAndGain(0.6, 1.0)
            self.chase_camera.RenderShadows(True)
            self.chase_camera.SetEnvironmentProperties(self.env.obj)
        self.chase_renders += 1
        return self.chase_camera

    def chase_view(self, camera_position: list[float], camera_quat: list[float]) -> np.ndarray:
        """(H, W, 3) RGB render of the full scene with the vehicle in it, from a camera pose (world)."""
        from offroad.sim.video import quat_about_y, quat_matrix, quat_mul

        position, q = self.pose()
        if self.last_xy is not None:  # wheels roll with the distance driven
            self.wheel_turn += float(np.hypot(*(position[:2] - self.last_xy))) / self.wheels[0][3]
        self.last_xy = position[:2].copy()
        self.env.SetActorPosition(self.body_actor, list(position), q)
        R = quat_matrix(q)
        wheel_q = quat_mul(q, quat_about_y(self.wheel_turn))
        for actor, forward, left, radius in self.wheels:
            centre = position + R @ np.array([forward, left, radius - self.cg_height_m])
            self.env.SetActorPosition(actor, list(centre), wheel_q)
        camera = self._chase_camera()
        camera.SetPose(list(camera_position), list(camera_quat))
        camera.Update(self.env, 0.1)
        image = np.ascontiguousarray(camera.GetNumpyArray(), dtype=np.uint8).copy()
        hidden = [float(position[0]), float(position[1]), float(position[2]) - 500.0]
        for actor in [self.body_actor, *(w[0] for w in self.wheels)]:
            self.env.SetActorPosition(actor, hidden, q)
        return image

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
