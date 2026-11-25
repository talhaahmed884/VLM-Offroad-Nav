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

## Approach 2, phase 3: the VLM's picture and the zero-shot selector

```bash
python scripts/show_prompt.py --sequence 0 --frame 601 --mode B   # exactly what the VLM sees, plus the answer key
python scripts/run_selector.py --limit 5                          # zero-shot Qwen3-VL-4B on 5 validation frames
python scripts/run_selector.py --split val --every 4              # ~84 validation frames, both modes (~20-30 min)
python scripts/run_selector.py --split val --every 4 --history none   # the same without the history
python scripts/run_selector.py --split val --every 4 --model Qwen/Qwen3-VL-8B-Instruct   # the 8B reference
```

The relaxed candidates are drawn on the camera image (shrunk to 800 × 500) as colored lines with
white dots at 5, 10 and 15 m and their color name at the far end. Where candidates share a
stretch (they often follow the same track at first), the lines are drawn side by side, so every
color can be followed from the bottom of the picture. Colors are assigned at random
per frame and listed in the prompt in random order, so the model can't learn "red is usually
right". The prompt asks, for each candidate, how far it is safe ("all", "tick 2", "tick 1",
"none"), then which one to drive on and whether to continue or stop. Mode B adds a direction
word. The history is the model's own last 3 decisions, without colors ("2.0 s ago: continued
slight left"). The picture uses only the image, the calibration, the candidates and the LiDAR
ground, so it works the same in a closed-loop simulator.

`run_selector.py` checks each answer, scores it against the labels (does the choice match the
driver, is it safe, does it go through a puddle when a dry path existed, are the ticks right),
and writes `outputs/selector/<tag>_mode{A,B}.jsonl` and `<tag>_summary.md`. A stopped run can be
continued with the same command. Each answer records the picture version; when the drawing
changes, the script won't add to old answers (use a new `--tag`, or `--fresh` to overwrite).
Finished runs are sorted into one subfolder per collection (`outputs/selector/phase3a_...` to
`phase5b_...`; the index is `outputs/selector/README.md`). `evaluate.py --runs <tag>` finds a tag
in any subfolder; `run_selector.py` refuses a tag that is already in a subfolder unless
`--out-dir` points there, so a run is never split in two.

## Approach 2, phase 4: evaluation against baselines

```bash
python scripts/run_all.py --only candidates eval        # candidates with terrain numbers, then baselines on test
python scripts/run_selector.py --split test --every 1   # zero-shot 4B on all 331 test frames, both modes (~1 h)
python scripts/run_selector.py --split test --every 1 --history none
python scripts/run_selector.py --split test --every 1 --history wrong
python scripts/run_selector.py --split test --every 1 --modes B --direction "bear left"
python scripts/run_selector.py --split test --every 1 --modes B --direction "bear right"
python scripts/run_selector.py --split test --every 1 --model Qwen/Qwen3-VL-8B-Instruct
python scripts/evaluate.py --split test \
    --runs test_Qwen3-VL-4B-Instruct test_Qwen3-VL-4B-Instruct_nohist test_Qwen3-VL-4B-Instruct_wronghist test_Qwen3-VL-8B-Instruct \
    --direction-test test_Qwen3-VL-4B-Instruct_dir-bear-left test_Qwen3-VL-4B-Instruct_dir-bear-right \
    --history-test test_Qwen3-VL-4B-Instruct test_Qwen3-VL-4B-Instruct_wronghist
```

`evaluate.py` puts each run next to simple baselines on the same frames: a random pick, the
straightest candidate, the cheapest one (the geometric planner's own pick), "toward the word"
in Mode B (no image needed), the driver's candidate (the best possible), and the driven path
itself. It reports how often the choice matches the driver, what terrain the chosen path
crosses, how close it passes hard obstacles, how far it is from the driven path at 5 / 10 / 15 m,
how good the safe-distance ticks are (against "always all"), and continue / stop. Two behaviour
tests check whether the choice follows the direction word (same frames asked with "bear left"
and "bear right") and whether a wrong history pulls it. Output: `outputs/eval/<split>_evaluation.md`.

Error measures for the final report (added in phase 5): shares come with a 95% confidence
interval; the distance from the driven path at 15 m with interval, median and 90th percentile,
and the mean over 5 / 10 / 15 m (ADE, as in trajectory-prediction papers); straight vs turn
frames; on which side wrong choices fall and where the chosen candidate sits (left / middle /
right); the ticks as hazard flags ("not safe to the end": precision / recall / F1) and the mean
tick error; stop F1; seconds per answer; and **paired comparisons**: every run is compared frame
by frame with the straightest and the cheapest candidate (only the frames where the two disagree
count; exact McNemar p-value), plus any `--paired FIRST SECOND` you add (run tags or baseline names).

## Approach 2, phase 5: fine-tuning the selector (QLoRA)

```bash
# On the laptop (needs the RELLIS sequences 00000-00003): render the training data
python scripts/build_sft_data.py --no-match skip      # -> outputs/sft/nomatch-skip
python scripts/build_sft_data.py --no-match keep      # -> outputs/sft/nomatch-keep
# On a GPU machine (Polaris): train, saving an adapter per epoch
python scripts/train_selector.py --data outputs/sft/nomatch-skip   # -> outputs/adapters/nomatch-skip/epoch1..5
# Pick the epoch on validation, then ask the test drive with the chosen adapter
python scripts/run_selector.py --split val --adapter outputs/adapters/nomatch-skip/epoch3
python scripts/evaluate.py --split val --runs val_ft-nomatch-skip-epoch1 val_ft-nomatch-skip-epoch2 ...
python scripts/run_selector.py --split test --every 1 --adapter outputs/adapters/nomatch-skip/epoch3
```

The training examples are the selector's own pictures and prompts, with the answer built from
the labels (the driver's candidate as the choice, the true safe distances, continue / stop),
without the reasoning sentence. Against overfitting, every example gets its own random colors,
listing order and wording, half are mirrored left-right, the history is sometimes left out or
made wrong, the camera image gets small brightness / contrast / color changes, and rare frames
(turns, puddles, unsafe candidates) are used more often than easy straight ones. The base model
stays frozen in 4-bit; only small LoRA adapters are trained. `--no-match skip|keep` builds the
two datasets of the experiment on frames where no candidate matches the driver.

### Phase 5 results (done 2026-11-24)

Both datasets were trained for 5 epochs on Polaris (H200, 33–40 min each). The epoch was chosen
on the validation drive (00003): **"skip" (no-match frames left out), epoch 4**, the reported
model. The test drive (00004, 328 frames with candidates) was used only once, for the final
numbers (`outputs/eval/test_finetuned.md`; Mode A):

| chooser | matches the driver [95% interval] | turn frames | mean distance from the driven path (ADE) |
|---|---|---|---|
| random | 43% [38, 48] | 33% | 1.57 m |
| zero-shot 4B (asked without reasoning) | 44% [38, 49] | 18% | 1.53 m |
| cheapest candidate | 48% [43, 54] | 42% | 1.26 m |
| straightest candidate | 67% [62, 72] | 43% | 1.07 m |
| **fine-tuned 4B, skip epoch 4** | **66% [61, 71]** | **57%** | **0.98 m** |
| fine-tuned 4B, keep epoch 5 | 59% [53, 64] | 49% | 1.06 m |
| driver's candidate (best possible) | 89% [86, 92] | 72% | 0.75 m |

- Fine-tuning helps: right where the zero-shot model is wrong 110 times, the reverse 35
  (p < 0.001). Leaving out the no-match frames helps too (skip vs keep: 31 vs 6, p < 0.001).
- The test drive is mostly straight, so "straightest" ties the model there (50 vs 51 frames);
  on the twistier validation drive the model beats it (56 vs 44%; 66 vs 25 frames, p < 0.001).
  The model is better on turns and its paths end closer to the driver's.
- Limitations: after fine-tuning the model **ignores the direction word** (a forced "bear left"
  vs "bear right" changes 5% of its choices; zero-shot 89%) and the history; it says "safe all"
  for every path. RELLIS has 7 unsafe stretches and no stop situations on the test drive, so
  safety and stop can't be measured offline (closed loop, phase 6).
- Not done (decided at closing): GRPO, fine-tuning the 8B; the check on non-RELLIS scenes moves
  to phase 6.

## Layout

- `src/offroad/`: library code
  - `data/`: RELLIS-3D loading and calibration
  - `geometry/`: transforms and LiDAR-to-camera projection
  - `mapping/`: elevation map and geometric cost map
  - `planning/`: A\* and the geometric baseline
  - `nav/`: approach 2: driven paths from the poses, direction words, decision frames, candidate paths and their labels, the VLM's picture and prompts, answer checking and scoring, terrain numbers per path, and the evaluation with baselines
  - `vlm/`: running a local VLM (Transformers), with an optional fine-tuned LoRA adapter; fine-tuning helpers
  - `eval/`: ground truth along a path (LiDAR + camera labels) and safety metrics
  - `viz/`: pictures
- `scripts/`: runnable entry points
- `tests/`: pytest tests (run `pytest` from the repo root; no dataset needed)
- `configs/`: the decision frames (`decision_frames.csv`)
- `docs/`: the approach 2 plan
- `outputs/`: generated images and tables (not meant to be committed)
