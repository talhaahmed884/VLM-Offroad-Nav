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

If `RELLIS_ROOT` is unset, code falls back to `./data/` (gitignored).

## Layout

- `src/offroad/` — library code (data loading, geometry, visualization)
- `scripts/` — runnable entry points
- `tests/` — pytest tests (`pytest tests/`)
- `outputs/` — generated images (gitignored)
