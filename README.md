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

## Run everything at once

```bash
python scripts/run_all.py              # unit tests + every script on one sample frame (about a minute)
python scripts/run_all.py --full       # also the slow steps: all 36 evaluation frames, label statistics
python scripts/run_all.py --tests-only # just the unit tests
python scripts/run_all.py --list       # show the steps and their commands
```

It prints each step's output as it runs, then a pass/fail summary. Use `--only <step> ...` to run specific steps. The individual scripts are described below.

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

The chosen evaluation frames are in [`configs/eval_frames.csv`](configs/eval_frames.csv): 36 frames, the top 8 of mud, water, bush, and easy plus the top 4 rubble.

## Elevation map (top-down terrain)

```bash
python scripts/build_elevation_map.py --sequence 0 --frame 0
```

This saves `outputs/elevation_00000_000000.png`: the camera image plus three top-down maps of the ground ahead (ground height, step height, and the most common ground-truth class per 25 cm cell). The vehicle is at the bottom looking up, and the cyan dashed lines show what the camera sees.

## Geometric cost map (baseline)

```bash
python scripts/build_cost_map.py --sequence 0 --frame 0
```

This saves `outputs/costmap_00000_000000.png`, showing which 25 cm cells the geometry-only baseline treats as smooth, rough, unknown, or blocked (anything taller than 0.3 m above the local ground; change this with `--max-step`). It also prints how much of each ground-truth class (bush, mud, puddle, ...) the geometry marks as an obstacle.

## Baseline A* path

```bash
python scripts/plan_path.py --sequence 0 --frame 0
```

This saves `outputs/path_00000_000000.png`: the geometry-only A* path from the vehicle toward a target 20 m ahead (`--goal-distance`), drawn on the cost map, on the ground-truth classes, and projected into the camera image. If obstacles cut the target off, the path goes to the closest cell it can reach, and the script reports how far short it stops. It also prints the path length, its cost, and the ground-truth classes it crosses.

## Baseline on all evaluation frames

```bash
python scripts/run_baseline.py
```

This runs the geometry-only baseline on every frame in `configs/eval_frames.csv`. It saves one picture per frame and `summary.csv` (target reached or not, shortfall, length, detour ratio, cost, and the share of the path crossing each ground-truth class) to `outputs/baseline/`, then prints a summary per category.

## VLM grid overlay

```bash
python scripts/draw_grid.py --sequence 0 --frame 120   # one frame
python scripts/draw_grid.py --eval                     # all evaluation frames -> outputs/grid/
```

This draws the labelled grid the VLM will answer in: 8 rows (A–H, from the top) by 12 columns (1–12, from the left), so the top-left cell is A1 and the bottom-right is H12. Use `--rows` and `--cols` to change it.

## VLM prompt and answer checker

```bash
python scripts/vlm_prompt.py --sequence 0 --frame 120                         # print the prompt, save the gridded image
python scripts/vlm_prompt.py --sequence 0 --frame 120 --answer my_answer.txt  # also check a saved VLM answer
```

The prompt asks the VLM for JSON with `waypoints` (grid cells from near to far), `cost_hints` (cells with a `terrain` type and a `traversability` level from `preferred` to `impassable`), and a short `reasoning`. The checker repairs common format mistakes, drops invalid cells and unknown words, and lists every problem it found. Files are saved to `outputs/vlm/`.

## Run a VLM locally

The VLM runs on your own GPU with PyTorch and Hugging Face Transformers. The default model is [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) in 4-bit, which fits an 8 GB GPU. Install the extra packages once (the steps are also in [`requirements-vlm.txt`](requirements-vlm.txt)):

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-vlm.txt
python scripts/check_gpu.py     # checks PyTorch sees the GPU and 4-bit layers work
```

Then:

```bash
python scripts/run_vlm.py --sequence 0 --frame 120   # one frame
python scripts/run_vlm.py                            # all evaluation frames
```

The first run downloads the model (about 9 GB) into the Hugging Face cache, outside the repo. For each frame the script names the grid cell of the 20 m target (the same target the baseline uses), sends the gridded image (shrunk to 1280 px wide) and the prompt, and checks the answer. It saves the prompt, the raw answer, the checked answer, and the response time to `outputs/vlm_runs/<model>_<settings>/`, plus `summary.csv`. The printed table shows, per category, how many answers were usable, the mean number of format problems, how many routes pass through cells the model itself marked "avoid" or "impassable" (`conflicted`), and the mean response time. Frames already answered are skipped, so an interrupted run can continue (`--overwrite` redoes them). Use `--model` and `--quantize` to try other models (e.g. `--model Qwen/Qwen3-VL-2B-Instruct --quantize none`), and `--no-briefing` to leave out the vehicle sentence.

## Score and draw the VLM answers

```bash
python scripts/score_vlm.py                                   # every run in outputs/vlm_runs
python scripts/score_vlm.py --run Qwen3-VL-8B-Instruct_4bit   # one run
```

This compares each saved answer with the RELLIS-3D camera label image, cell by cell. Is a "water" cell really water? Is a "preferred" cell really good ground? Were the puddle cells near the vehicle flagged? Was a person mentioned when one is in view? How much water does the route cross, compared with driving straight ahead? It prints one column per run and saves `outputs/vlm_runs/comparison.csv`, a `scores.csv` per run, and a picture per answer in `outputs/vlm_runs/<run>/answers/`: the camera image with the hints tinted by traversability and the route drawn on it, next to the label image. The thresholds (for example "a cell is water if ≥ 30% of its pixels are puddle or water") are constants at the top of `src/offroad/vlm/scoring.py`.

## Layout

- `src/offroad/` — library code (data loading, geometry, visualization)
- `scripts/` — runnable entry points
- `tests/` — pytest tests (run `pytest` from the repo root; no dataset needed)
- `outputs/` — generated images (gitignored)
