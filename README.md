# VLM-Offroad-Nav

CS5230 (Connected and Autonomous Vehicles) course project at Cal Poly Pomona.

Can a small vision-language model (VLM) guide an off-road vehicle without a map? We test this
on the [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) dataset.

## Approaches

This repository holds two approaches on separate branches:

| Branch | Approach | Status |
|---|---|---|
| [`approach-1-vlm-grid-costmap`](../../tree/approach-1-vlm-grid-costmap) | The VLM reads a labelled grid on the camera image and re-weights a LiDAR cost map; A\* plans on it. Includes the full evaluation, results, and report figures. | done |
| `approach-2-mapless-vln` (this branch) | **Mapless "keep going" navigation, after DriveVLN** (Guo et al., CVPR 2026): a geometric planner proposes a few drivable candidate paths, which are drawn in colors on the camera image; the VLM (Qwen3-VL-4B, zero-shot and fine-tuned) picks one, says how far each is safe, and whether to stop. Optional coarse directions ("bear left"). | planning done; implementation starting |

The plan, decisions, and settings for approach 2 are in [`docs/approach-2-plan.md`](docs/approach-2-plan.md).
This branch keeps only the parts of approach 1 that approach 2 reuses: the data loader, the
LiDAR-camera projection, the terrain and cost map, A\*, the geometric baseline, the VLM runner,
and the ground-truth and safety metrics.

## Setup

```bash
conda create -n vlm-offroad-nav python=3.13   # or python -m venv .venv
conda activate vlm-offroad-nav
pip install -r requirements.txt
```

For the VLM and fine-tuning (needs an NVIDIA GPU):

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-vlm.txt
python scripts/check_gpu.py     # checks PyTorch sees the GPU and 4-bit layers work
```

`requirements-vlm.txt` pins Transformers, Accelerate, bitsandbytes (4-bit models), PEFT (LoRA
adapters), and TRL (fine-tuning trainers). Models are downloaded from Hugging Face on first use
into `~/.cache/huggingface` (set `HF_HOME` to put them elsewhere).

## Data

We use the file-based release of [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) (no ROS bags). The data is never committed to git.

### 1. Download

All links are from the [RELLIS-3D README](https://github.com/unmannedlab/RELLIS-3D) (checked 2026-11-22). You need these eight files:

| File | Size | What it gives you |
|---|---|---|
| [Full Images](https://drive.google.com/file/d/1F3Leu0H_m6aPVpZITragfreO_SGtL2yV/view?usp=sharing) | 11 GB | `pylon_camera_node/`, RGB camera images |
| [Full Image Annotations ID Format](https://drive.google.com/file/d/16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5/view?usp=sharing) | 94 MB | `pylon_camera_node_label_id/`, per-pixel class IDs |
| [Ouster LiDAR SemanticKITTI Format](https://drive.google.com/file/d/1lDSVRf_kZrD0zHHMsKJ0V1GN9QATR4wH/view?usp=sharing) | 14 GB | `os1_cloud_node_kitti_bin/`, LiDAR scans |
| [Ouster LiDAR Annotation SemanticKITTI Format](https://drive.google.com/file/d/12bsblHXtob60KrjV7lGXUQTdC5PhV8Er/view?usp=sharing) | 174 MB | `os1_cloud_node_semantickitti_label_id/`, per-point class IDs |
| [Ouster LiDAR Scan Poses files](https://drive.google.com/file/d/1V3PT_NJhA41N7TBLp5AbW31d0ztQDQOX/view?usp=sharing) | 174 MB | `poses.txt` in each sequence: the pose of every scan, i.e. the path the vehicle actually drove. **New for approach 2.** |
| [Camera Intrinsic](https://drive.google.com/file/d/1NAigZTJYocRSOTfgFBddZYnDsI_CSpwK/view?usp=sharing) | 2 KB | `camera_info.txt` in each sequence |
| [Basler Camera to Ouster LiDAR](https://drive.google.com/file/d/19EOqWS9fDUFp4nsBrMCa69xs9LgIlS2e/view?usp=sharing) | 3 KB | `transforms.yaml` in each sequence |
| [Ontology Definition](https://drive.google.com/file/d/1K8Zf0ju_xI5lnx3NTDLJpVTs59wmGPI6/view?usp=sharing) | 18 KB | `Rellis_3D_ontology/`, the class ID to name mapping |

You can skip everything else on that page: the PLY point clouds, anything Velodyne or stereo, the raw calibration data, the split files, and the ROS bags. (We make our own train / validation / test split by sequence.)

**If Google Drive blocks the download:** large files are often rate-limited. Wait 24 hours and try again, use the [Baidu Pan mirror](https://pan.baidu.com/s/1akqSm7mpIMyUJhn_qwg3-w?pwd=4gk3) (code `4gk3`), or email maskjp@tamu.edu with the subject "RELLIS-3D Access Request".

### 2. Unzip into `data/`

Create a `data/` folder at the repo root, next to `src/` (not inside it), and unzip every file into it. Most zips contain a top-level `Rellis-3D/` folder, so their contents merge into one tree. If anything lands somewhere else (for example the ontology folder, the calibration files, or `poses.txt`), move it so the result matches the tree in step 3.

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
        │   ├── poses.txt                                # one pose per Ouster scan (approach 2)
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

Images and scans are matched by frame number: `frame000123-<timestamp>.jpg` goes with `000123.bin`. Each sequence should contain this many images, scans, and scan label files (and `poses.txt` should have one line per scan; we check this when it is first loaded):

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
python scripts/run_all.py              # unit tests + every script on one sample frame
python scripts/run_all.py --full       # also the slow steps (e.g. the GPU check)
python scripts/run_all.py --tests-only # just the unit tests
python scripts/run_all.py --list       # show the steps and their commands
```

It prints each step's output as it runs, then a pass/fail summary. Use `--only <step> ...` to run specific steps.

## Building blocks (kept from approach 1)

```bash
python scripts/project_lidar.py --sequence 0 --frame 0        # LiDAR points drawn on the camera image
python scripts/build_elevation_map.py --sequence 0 --frame 0  # top-down terrain heights
python scripts/build_cost_map.py --sequence 0 --frame 0       # geometry-only cost map (smooth / rough / unknown / blocked)
python scripts/plan_path.py --sequence 0 --frame 0            # A* path toward a target 20 m ahead
```

Each saves a picture to `outputs/`. The approach 2 scripts below are added phase by phase;
see [`docs/approach-2-plan.md`](docs/approach-2-plan.md).

## Approach 2, phase 1: driven paths and decision frames

```bash
python scripts/build_decision_frames.py                     # -> configs/decision_frames.csv
python scripts/show_driven_path.py --sequence 0 --frame 600 # the path the driver took from one frame
```

The scan poses give the LiDAR's position in a fixed world frame for every scan. Seen from one
frame, the positions of the following frames trace the path the driver actually took next.
This is the reference for "which candidate path matches the driver".

`build_decision_frames.py` picks the frames where the VLM will be asked to decide, about 2 per
second. It skips frames where the vehicle stands still or the recording ends within 20 m, and
labels each frame with its split (by sequence: train 00000–00002, validation 00003, test 00004)
and with Mode B's coarse direction ("bear left", "slight left", "straight", "slight right",
"bear right"), taken from where the driver was about 50 m later. Only about every other frame
has a camera label image, so a sampled frame without one is moved to its labelled neighbour
(0.1 s away). The CSV is small and is committed, so everyone uses the same frames.

`show_driven_path.py` saves `outputs/driven_<seq>_<frame>.png`. It has three panels: the
camera image with the driven path drawn on the ground, the cost map with the driven path, and
the whole drive seen from above. It prints how high the LiDAR rides above the mapped ground
along the path (about 1.2 m), as a check that the poses and the LiDAR agree.

## Approach 2, phase 2: candidate paths and their labels

```bash
python scripts/build_candidates.py                           # all decision frames, strict vs relaxed (~2 min)
python scripts/build_candidates.py --every 10 --split val    # a quick look at a subset
python scripts/show_candidates.py --sequence 0 --frame 601   # one frame's candidates, both variants
```

For each decision frame, a planner proposes up to 4 drivable paths about 15 m long that the VLM
will choose between (as DriveVLN's planner does). It fans out toward 13 directions within the
camera's view, finds the cheapest path toward each on a cost grid, smooths it, and keeps paths
that are clearly different (at least 2 m apart somewhere). Two cost grids are compared:
**strict** (approach 1's cost map: anything taller than 0.3 m, such as tall grass at a trail's
edge, blocks) and **relaxed** (only hard obstacles taller than 1.5 m block; lower vegetation is
passable but expensive).

Each candidate gets labels automatically: does it match the driver (mean distance to the driven
path ≤ 1 m, measured from 5 m on), and how far is it safe (the first water or hard obstacle
along it, from the LiDAR and camera labels, rounded down to a tick at 5 / 10 / 15 m). Puddles
are "caution", not unsafe, because the RELLIS driver drives through them; each candidate records
where its first puddle is. A frame's correct action is "stop" when no candidate is safe to the
first tick. These labels are only the answer key for training and scoring: the candidates come
from LiDAR geometry alone, and the VLM sees only the camera image, so nothing needs labels at
run time (e.g. in a closed-loop simulator).

`build_candidates.py` writes `outputs/candidates/{strict,relaxed}.jsonl` (one line per frame)
and a comparison table, `outputs/candidates/comparison.md`, per split. `show_candidates.py`
saves `outputs/candidates_<seq>_<frame>.png`: the candidates on the camera image (white dots at
the ticks, the driven path in cyan) and on both cost grids from above.

## Layout

- `src/offroad/`: library code
  - `data/`: RELLIS-3D loading and calibration
  - `geometry/`: transforms and LiDAR-to-camera projection
  - `mapping/`: elevation map and geometric cost map
  - `planning/`: A\* and the geometric baseline
  - `nav/`: approach 2: driven paths from the poses, direction words, decision frames, candidate paths and their labels
  - `vlm/`: running a local VLM (Transformers)
  - `eval/`: ground truth along a path (LiDAR + camera labels) and safety metrics
  - `viz/`: pictures
- `scripts/`: runnable entry points
- `tests/`: pytest tests (run `pytest` from the repo root; no dataset needed)
- `configs/`: the decision frames (`decision_frames.csv`)
- `docs/`: the approach 2 plan
- `outputs/`: generated images and tables (not meant to be committed)
