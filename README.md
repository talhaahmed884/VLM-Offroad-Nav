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

We use the file-based release of [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) (no ROS bags). The data is never committed to git.

### 1. Download

All links are from the [RELLIS-3D README](https://github.com/unmannedlab/RELLIS-3D) (checked 2026-10-31). You need these seven files:

| File | Size | What it gives you |
|---|---|---|
| [Full Images](https://drive.google.com/file/d/1F3Leu0H_m6aPVpZITragfreO_SGtL2yV/view?usp=sharing) | 11 GB | `pylon_camera_node/`, RGB camera images |
| [Full Image Annotations ID Format](https://drive.google.com/file/d/16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5/view?usp=sharing) | 94 MB | `pylon_camera_node_label_id/`, per-pixel class IDs |
| [Ouster LiDAR SemanticKITTI Format](https://drive.google.com/file/d/1lDSVRf_kZrD0zHHMsKJ0V1GN9QATR4wH/view?usp=sharing) | 14 GB | `os1_cloud_node_kitti_bin/`, LiDAR scans |
| [Ouster LiDAR Annotation SemanticKITTI Format](https://drive.google.com/file/d/12bsblHXtob60KrjV7lGXUQTdC5PhV8Er/view?usp=sharing) | 174 MB | `os1_cloud_node_semantickitti_label_id/`, per-point class IDs |
| [Camera Intrinsic](https://drive.google.com/file/d/1NAigZTJYocRSOTfgFBddZYnDsI_CSpwK/view?usp=sharing) | 2 KB | `camera_info.txt` in each sequence |
| [Basler Camera to Ouster LiDAR](https://drive.google.com/file/d/19EOqWS9fDUFp4nsBrMCa69xs9LgIlS2e/view?usp=sharing) | 3 KB | `transforms.yaml` in each sequence |
| [Ontology Definition](https://drive.google.com/file/d/1K8Zf0ju_xI5lnx3NTDLJpVTs59wmGPI6/view?usp=sharing) | 18 KB | `Rellis_3D_ontology/`, the class ID to name mapping |

You can skip everything else on that page: the PLY point clouds, anything Velodyne or stereo, the raw calibration data, and the ROS bags.

**If Google Drive blocks the download:** large files are often rate-limited. Wait 24 hours and try again, use the [Baidu Pan mirror](https://pan.baidu.com/s/1akqSm7mpIMyUJhn_qwg3-w?pwd=4gk3) (code `4gk3`), or email maskjp@tamu.edu with the subject "RELLIS-3D Access Request".

### 2. Unzip into `data/`

Create a `data/` folder at the repo root, next to `src/` (not inside it), and unzip every file into it. Most zips contain a top-level `Rellis-3D/` folder, so their contents merge into one tree. If anything lands somewhere else (for example the ontology folder or the calibration files), move it so the result matches the tree in step 3.

On Windows, "Extract All" adds an extra folder named after the zip. Either set the destination to exactly `...\VLM-Offroad-Nav\data`, or use 7-Zip's "Extract Here" from inside `data/`.

You need about 45 GB of free disk space after unzipping.

### 3. Check the folder structure

When you're done, it should look like this:

```
VLM-Offroad-Nav/
└── data/
    └── Rellis-3D/
        ├── Rellis_3D_ontology/
        │   ├── ontology.yaml
        │   ├── ontology.csv
        │   └── ontology.png
        ├── 00000/
        │   ├── camera_info.txt                          # intrinsics: fx fy cx cy
        │   ├── transforms.yaml                          # camera <-> Ouster extrinsics
        │   ├── pylon_camera_node/
        │   │   └── frame000000-1581624652_750.jpg       # one image per frame
        │   ├── pylon_camera_node_label_id/
        │   │   └── frame000000-1581624652_750.png       # only about every other frame
        │   ├── os1_cloud_node_kitti_bin/
        │   │   └── 000000.bin                           # one scan per frame
        │   └── os1_cloud_node_semantickitti_label_id/
        │       └── 000000.label                         # one label file per scan
        ├── 00001/   (same layout)
        ├── 00002/   (same layout)
        ├── 00003/   (same layout)
        └── 00004/   (same layout)
```

Images and scans are matched by frame number: `frame000123-<timestamp>.jpg` goes with `000123.bin`. Each sequence should contain this many images, scans, and scan label files:

| Sequence | 00000 | 00001 | 00002 | 00003 | 00004 |
|---|---|---|---|---|---|
| Frames | 2847 | 2319 | 4147 | 2184 | 2059 |

To confirm everything is in place, run `python scripts/project_lidar.py --sequence 0 --frame 0` (see below). If it saves an image, the data is set up correctly.

### Storing the data somewhere else

If you keep the dataset outside the repo (for example, on another drive or on the GPU cluster), point to the `Rellis-3D` folder with an environment variable:

```bash
export RELLIS_ROOT=/path/to/Rellis-3D          # macOS / Linux
```

```powershell
$env:RELLIS_ROOT = "D:\datasets\Rellis-3D"    # Windows PowerShell
```

If `RELLIS_ROOT` is not set, the code looks in `./data/Rellis-3D/`.

### License

RELLIS-3D is published under [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) (non-commercial use). If you use it, cite Jiang et al., "RELLIS-3D Dataset: Data, Benchmarks and Analysis" ([arXiv 2011.12954](https://arxiv.org/abs/2011.12954)).

## Project LiDAR into an image

```bash
python scripts/project_lidar.py --sequence 0 --frame 0
```

This saves `outputs/overlay_00000_000000.png`, with each LiDAR point drawn as a dot colored by distance (blue = near, red = far).

## Propose evaluation frames

```bash
python scripts/compute_label_stats.py   # once, a few minutes: class shares per labelled frame
python scripts/select_eval_frames.py    # candidates per category + contact sheets
```

This writes `outputs/eval_candidates/candidates.csv` and one contact sheet per category (mud, water, bush, rubble, easy). In each sheet, the category's pixels are tinted magenta and the "ahead" region is outlined in white.

## Layout

- `src/offroad/` — library code (data loading, geometry, visualization)
- `scripts/` — runnable entry points
- `tests/` — pytest tests (run `pytest` from the repo root; no dataset needed)
- `outputs/` — generated images (gitignored)
