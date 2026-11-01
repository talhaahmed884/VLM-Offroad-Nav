# VLM-Offroad-Nav

CS5230 (Connected and Autonomous Vehicles) course project at Cal Poly Pomona.

Can an onboard Vision-Language Model act as a high-level guide for a geometric path planner in off-road driving? We test this on the [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) dataset.

**Current phase:** load RELLIS-3D frames and project Ouster LiDAR points into the camera image.

## Setup

```bash
conda create -n vlm-offroad-nav python=3.13   # or python -m venv .venv
conda activate vlm-offroad-nav
pip install -r requirements.txt
```

## Data

Download RELLIS-3D (images, Ouster point clouds, labels) plus the "Camera Intrinsic" and "Basler Camera to Ouster LiDAR" calibration files. Keep the data outside the repo and point to it with an environment variable:

```bash
export RELLIS_ROOT=/path/to/Rellis-3D
```

If `RELLIS_ROOT` is unset, code falls back to `./data/Rellis-3D/` (gitignored), with sequence folders `00000/`–`00004/` directly inside it.

## Project LiDAR into an image

```bash
python scripts/project_lidar.py --sequence 0 --frame 0
```

This saves `outputs/overlay_00000_000000.png`, with each LiDAR point drawn as a dot colored by distance (blue = near, red = far).

## Layout

- `src/offroad/` — library code (data loading, geometry, visualization)
- `scripts/` — runnable entry points
- `tests/` — pytest tests (run `pytest` from the repo root; no dataset needed)
- `outputs/` — generated images (gitignored)
