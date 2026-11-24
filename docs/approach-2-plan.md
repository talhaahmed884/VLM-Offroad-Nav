# Approach 2: mapless, language-guided off-road driving (plan)

Adapting the planner + VLM-selector idea of **DriveVLN** (Guo et al., CVPR 2026, "Towards
Mapless Vision-and-Language Navigation in Autonomous Driving") from on-road parking lots to
off-road driving on RELLIS-3D. This is a plan only; nothing here is implemented yet.
Approach 1 is preserved on the branch `approach-1-vlm-grid-costmap`.

## 1. What DriveVLN does (summary)

- **Task.** The vehicle gets only a short, destination-level instruction ("Go to the exit"), no
  map and no step-by-step route. At every decision it must (a) say whether it has arrived and
  (b) pick a direction that leads toward the destination, using what it sees (signs, landmarks)
  and what it decided before.
- **Model: two parts.**
  1. A **planning module** (DiffusionDrive, from camera + LiDAR) proposes several *feasible but
     destination-agnostic* candidate trajectories (8 waypoints over 4 s). It only knows what is
     drivable, not where to go.
  2. A **navigation selector** (Qwen2.5-VL-3B, fine-tuned) sees the front camera image with the
     top-4 candidates **drawn in different colors**, plus the instruction and a rolling history
     of its earlier questions and answers. It answers in a fixed format: "not arrived; yellow".
- **Training.** Supervised fine-tuning (LoRA, rank 64, 5 epochs, 800×600 images) on single
  frames, then reinforcement learning (GRPO) in simulation. The reward combines a *local* term
  (how close the chosen trajectory is to the ground-truth one, zeroed on any collision) and a
  *global* term (how much of the shortest route it followed, plus a bonus for arriving).
- **Benchmark.** 200 CARLA scenes rebuilt from real scans; closed-loop evaluation. Metrics:
  success rate, navigation error, L2 error of the chosen trajectory at 1/2/3 s, collision rate,
  and a Driving Score = route completion × infraction penalty.
- **Key results.** Fine-tuned 3B selector: Driving Score 0.67, success 0.44. Zero-shot GPT-5: 0.48;
  Qwen2.5-VL-72B: 0.45. RL added a lot (success 0.21 → 0.44).

## 2. Our task: "keep going" off-road, with optional coarse directions

**Decision (2026-11-22): no destinations.** Off-road, destinations are arbitrary and usually
unnamed. A realistic command is "keep driving (this way) until I stop you". The stop comes from
the operator or from odometry, not from the VLM. So DriveVLN's "arrived?" question is replaced,
and the instruction becomes optional and coarse.

At each decision frame, the planner proposes K drivable candidate paths, drawn in colors with
numbered distance ticks. The VLM answers, in one call:

```json
{
  "choice": "<color of the path to drive>",
  "safe_until": {"<color>": "all" | "tick 1" | "tick 2" | "none", "...": "..."},
  "action": "continue" | "stop",
  "reasoning": "<one or two sentences>"
}
```

- **choice:** the path to drive next.
- **safe_until:** for every candidate, how far along it is safe, read off numbered ticks at 5, 10,
  15 m (VLMs are poor at meters but can point at a tick). This is the user's idea of letting the
  VLM check how far the planner's path is really drivable.
- **action:** "stop" only when no candidate is safe; this replaces DriveVLN's arrival flag.

Two prompt modes share this answer format, the frames, the candidates, and the labels:

| | Mode A: "keep going" | Mode B: "keep going + direction" |
|---|---|---|
| Instruction | none: "drive on safely; prefer trails; avoid water, mud and obstacles" | plus a coarse direction, e.g. "bear left", "keep straight", "bear right" |
| Main question | can a small VLM keep the vehicle safe? | can it also follow a coarse human direction? |
| Scored on | safety of the chosen path; safe-distance answers; stop decisions | the same, plus "did it take the path the driver took" |

Mode B's direction is generated automatically from where the driver actually went over the next
~50 m, and is kept coarse (a few discrete options) so it doesn't give the answer away. It plays
the role of DriveVLN's instruction (and of a coarse GPS heading) and gives forks a right answer.

**Both modes are evaluated.** One fine-tuned model is trained on a mix of both (some examples
with a direction, some without).

### Mapping from DriveVLN

| DriveVLN (on-road) | Off-road adaptation |
|---|---|
| Destination instruction ("Go to the exit") | None (Mode A) or a coarse direction (Mode B); the stop is external |
| "Arrived or not" | "Continue or stop", where stop = no safe candidate |
| Choose the candidate toward the destination | Choose a safe candidate (A), or the one matching the direction (B) |
| — | New: how far each candidate is safe (distance ticks) |
| DiffusionDrive planner (learned) | Our cost map + A\* toward a fan of directions (from approach 1) |
| CARLA closed loop | Offline, open-loop on recorded RELLIS drives; the driver's path is the reference |
| Collision in simulation | Path crosses LiDAR points labelled as hard obstacles |
| — | Water / mud / vegetation on the path, from approach 1's two-source ground truth |

This also fixes approach 1's main weakness: there the VLM had to name grid cells and re-weight
costs, and it described only about 12% of the ground. Choosing among a few drawn paths and
pointing at a tick is a much easier question for a VLM.

## 3. What we reuse from approach 1

- data loading, calibration, LiDAR-to-camera projection (`data/`, `geometry/`);
- the terrain map and cost map (`mapping/`), A\* and Dijkstra (`planning/astar.py`);
- drawing paths on the camera image (`viz/baseline_figure.draw_path_on_camera`);
- the local VLM runner and the GPU setup (`vlm/backend.py`, `vlm/runner.py`);
- two-source ground truth along a path, hazard and clearance metrics (`eval/path_truth.py`, `eval/metrics.py`);
- `run_all.py`, the test conventions, and the oracle idea (an upper bound with perfect choices).

These are the only parts of approach 1 kept on this branch (the branch was cleaned on
2026-11-22). Ideas from deleted approach-1 code, such as the forgiving JSON answer checker, the
paired wins/losses comparison, and the report tables and charts, will be rewritten for approach 2
as needed; the originals stay on the `approach-1-vlm-grid-costmap` branch.

## 4. New data and labels

1. **Ouster scan poses** (RELLIS-3D download "Ouster LiDAR Scan Poses", about 174 MB, skipped so
   far). One pose per scan gives the vehicle's actual driven path. Everything below depends on
   it. On download, check the format (expected: one 3×4 matrix per line, SemanticKITTI style),
   which frame it is in, and that it lines up with our forward-left-up map frame.
2. **Decision frames.** Every RELLIS frame that has a scan and a pose, sampled at about 2 Hz (as
   DriveVLN recorded), so neighbouring frames aren't near-duplicates.
3. **Labels, all generated automatically:**
   - **driven path:** the next ~20 m of the recorded trajectory, in the vehicle frame;
   - **driver's candidate:** the candidate closest to the driven path; candidates within a
     tolerance (e.g. ≤ 1 m mean distance) also count, so frames with several equivalent paths
     aren't scored as wrong;
   - **safe distance per candidate:** the first water or hard-obstacle cell along it (puddles
     are "caution" since 2026-11-22, see setting 6),
     from the two-source ground truth, rounded to a tick;
   - **stop:** true when every candidate is unsafe within the first tick;
   - **Mode B direction:** the bearing of the driven path ~50 m ahead, binned into a few words.
4. **Splits by sequence, not by frame** (neighbouring frames are correlated): for example
   train 00000–00002, validate 00003, test 00004.

## 5. Implementation phases

Each phase ends in something we can run and check, as in approach 1.

### Phase 1: poses and decision frames
- Download the poses; write a loader and a check script that draws the driven trajectory of a
  sequence top-down and projects the next 20 m of it into the camera image. The projected path
  should lie on the trail the vehicle actually drove.
- Frame builder (pure + tests): sample decision frames at ~2 Hz, compute each frame's driven path
  in the vehicle frame and its Mode B direction word; write the split files.

**Status (2026-11-22): done; the user ran it and reviewed the picture.**
- **Poses checked:** `poses.txt` has one 3×4 matrix per scan (two files lack a final newline, so
  `wc -l` shows one fewer). They are `T_world_from_lidar`: two scans 2–3 s apart overlap 59–65%
  after moving them to the world frame this way, versus 1–11% with the inverse. The vehicle moves
  along LiDAR −x at about 1.25 m/s, and the drives are 226–483 m long. Along the driven path the
  LiDAR stays about 1.2 m above the mapped ground (00000/600: 1.16–1.28 m), so the poses' heights
  agree with the LiDAR ground.
- **Decision frames:** 2,220 (train 1,553 / val 336 / test 331), after skipping frames where
  the vehicle stands still (91) or the recording ends within 20 m (308). Mode B directions are
  mostly "straight" (train 862 of 1,553; val 256 of 336; test 250 of 331). The test drive has
  only 3 "bear left" and 0 "bear right" frames, so Mode B results on test will mainly show the
  slight turns; report per word.
- **Early warning for phase 2:** in 00000/2042 the driven trail beyond ~15 m runs through cells
  the cost map marks as blocked (grass and brush along the trail edges). Candidates from the cost
  map may not follow real trails there; phase 2's candidate-quality check will tell.
- **Added in phase 2:** only about every other frame has a camera label image, and the safe-distance
  labels need it, so each sampled frame now moves to a labelled neighbour (0.1 s away) when it has
  none (`prefer_labelled`). 2,207 of the 2,220 decision frames have one (13 have no labelled
  neighbour); the CSV gained a `labelled` column. The frame count and directions are unchanged.

### Phase 2: candidate paths and labels (the planning module)
- **K diverse, feasible candidates** per frame from the LiDAR cost map: end points on an arc
  ~15 m ahead at bearings from −60° to +60°, A\* to each, near-duplicates dropped (paths less than
  ~1.5 m apart), keep up to 4 (as DriveVLN). Smooth the A\* paths so they look drivable.
- Labels: driver's candidate(s), safe distance per candidate, stop.
- **Candidate-set quality:** how close the best candidate gets to the driven path (the ceiling
  for any selector), and how often frames have a real choice (≥ 2 safe, clearly different
  candidates). Tune the generator here before involving a VLM.
- Later comparison if there's time: **curved anchors**, a fixed library of smooth, drivable path
  shapes (straight, gentle/sharp left/right) built by clustering the recorded drives, placed in
  front of the vehicle and kept only if they don't hit obstacles. This is closer to DriveVLN's planner.

**Status (2026-11-22): done and verified by the user's run (see the end of this section).** The user asked
to build two cost grids and compare them ("strict" = approach 1's cost map, where anything over
0.3 m blocks; "relaxed" = only hard obstacles over 1.5 m block, lower vegetation costs 10).
- **How candidates are made** (`nav/candidates.py`): one Dijkstra shortest-path tree per frame;
  a fan of 13 bearings (−30° … +30°, every 5°) at 15 m; for each, the goal that trades closeness
  to the target against cost (the baseline's rule); smoothed, resampled every 0.25 m, cut at 15 m.
  Changes from the first sketch above, found on real frames:
  - **Camera view:** the camera sees only about ±19°, so ±60° candidates were invisible to the
    VLM. A candidate must now keep ≥ 80% of its points from 5 m on inside the view (2° margin).
  - **Duplicates:** "mean distance < 1.5 m" merged paths that share the trail for 7 m and then
    split 3–4 m apart. Two paths are now different if they get ≥ 2 m apart somewhere (compared
    point by point at the same distance along them). On a sample this raised the strict / relaxed
    mean from 1.3 / 1.5 to 2.2 / 3.1 candidates.
- **Labels** (`nav/labels.py`, `nav/frame_candidates.py`):
  - The driver match is measured from the first tick (5 m) on; with the shared, invisible start
    included, nearly every candidate "matched". The driver's choice is the closest candidate.
  - A hard-obstacle label counts only where the LiDAR also sees something taller than 0.3 m in or
    next to the cell: RELLIS marks some flat ground as "person" (people walking near the vehicle
    smear onto the ground in the annotation). An unsafe stretch must be ≥ 0.5 m (2 points) long.
  - The driven path is labelled like a candidate, as a check on the labels.
- **Trial run, all 2,220 frames (~2 min with 4 workers):**

  | | strict | relaxed |
  |---|---|---|
  | candidates per frame (mean) | 2.20 | 3.13 |
  | frames with 0 / 1 / 2 / 3 / 4 | 4 / 22 / 36 / 29 / 10% | 1 / 4 / 17 / 36 / 42% |
  | best candidate to driven path, median / 90th pct | 0.61 / 1.14 m | 0.68 / 1.34 m |
  | driver match (a candidate within 1 m) | 83% | 83% |
  | matching candidates per frame | 1.24 | 1.34 |
  | real choice (2+ safe to tick 1) | 75% | 95% |
  | stop | 6% | 2% |
  | full length | 86% | 98% |
  | candidates through a puddle (caution) | 9% | 8% |
  | driver's candidate through a puddle / of those, with a dry alternative | 9% / 21% | 9% / 42% |

  (Numbers with puddles as caution. With puddles unsafe, the first run gave real choice 72 / 92%
  and stop 8 / 4%.)

  - The ~17% of frames with no match are mostly sharp turns: the driver's path leaves the
    camera's view (53–63 of ~95 misses in a sample); the rest miss by 1.0–1.3 m.
  - Relaxed candidates almost never cross vegetation (cost per m ≤ 3.4); they gain options
    because vegetation isn't inflated, e.g. the left rut, the trail, and a branch into tall grass.
  - **Puddles:** the driver drives through puddles. With puddles unsafe, the labels called the
    driver's own path unsafe in 6% of frames (137 puddle, 4 person), and in most "stop" frames
    the driver kept going (150 of 183 strict, 50 of 79 relaxed). **User decision 2026-11-22:
    puddles are "caution"**: they don't shorten the safe distance or cause "stop", but each
    candidate records where its first puddle is (`first_caution_m`). Now the driver's own path
    is safe to the end in 100% of frames; the driver goes through a puddle in 6% of them.
  - **"Stop" on RELLIS is only planner failure.** All remaining stop frames (129 strict, 35
    relaxed) have no candidates or only candidates that end before 5 m, and the driver kept
    going in every one. RELLIS has no situation where the driver really had to stop, so stop
    can't be evaluated meaningfully here; it needs the simulator (phase 6). Phase 3–4 should
    decide whether to leave these frames out of the stop metrics.
  - **Safe distance is nearly always "all"** (96% of relaxed candidates); the unsafe ones are
    trees (83), "person" (19) and rubble (11). Safety on RELLIS will mostly show in the caution
    metric: in relaxed, 42% of the frames where the driver's candidate crosses a puddle (~85
    frames) also have a dry candidate. This makes balancing in phase 5 important.
- **User decision 2026-11-22: the relaxed cost grid** is used from phase 3 on
  (`CHOSEN_VARIANT`): more and more distinct candidates (3.1 vs 2.2 per frame, real choice 95%
  vs 75%) with the same driver match (83%). `build_candidates.py` still builds both, for the report.
- **User decision 2026-11-22: keep the 1 m match tolerance** (setting 5, measured from 5 m on;
  ~1.3 matching candidates per frame). Training uses the single closest candidate as the
  driver's choice; the tolerance only makes scoring forgiving of near-identical choices.
- **Verified 2026-11-22: phase 2 is done.** The user ran `run_all.py`; `configs/decision_frames.csv`,
  both JSONL files and `comparison.md` are byte-identical to my trial run.
- **Notes for phase 3's picture** (from `outputs/candidates_00000_000601.png`): at 10–15 m the
  candidates are squeezed into a thin band of image rows and cross each other near the bottom;
  their far ends show small hooks and wiggles because each dot sits on its own map cell's
  ground height. Draw them as lines on a smoothed ground (e.g. a plane fitted under the path, or
  the pose height minus the sensor height as `show_driven_path.py` does), with ticks that stand out.

### Phase 3: visual prompt and zero-shot selector
- Draw the candidates on the front camera image in distinct colors that stand out from grass and
  sky, each with a name at its end and numbered ticks at 5 / 10 / 15 m. Paths start below the
  image (the ~4.4 m blind zone) and are drawn from their first visible point.
- Prompts for Mode A and Mode B, the JSON answer above, and a forgiving checker (as in approach 1).
- **History:** the last few decisions as a short text summary (cheap on memory), with earlier
  images as an option if memory allows.
- Run zero-shot with **Qwen3-VL-4B** on the laptop; the existing 8B as a reference point.
- **Against shortcuts** (user decision 2026-11-22, see section 8): assign the colors to the
  candidates at random (and list them in the prompt in random order, not left to right); draw all
  candidates in the same style (same line width and dot size); no costs or other planner numbers
  in the prompt. The model should tell the candidates apart only by color and what it sees.
- **History:** prompts are built so the history can be left out, and evaluated both with and
  without it (see phase 4).

**Status (2026-11-22): written; my trial on 12 validation frames is below; not yet run by the user.**
- **Picture** (`nav/visual_prompt.py`): the camera image shrunk to 800 × 500 (the camera's
  shape; setting 10 said 800 × 600, which would stretch it). Each candidate is a colored line
  with a dark outline (drawn side by side where candidates share a stretch, see below), white dots at 5 / 10 / 15 m (the 5 m dot is at or just below the bottom
  edge; the camera starts seeing at ~4.4 m), and its color name at the far end. The lines sit on
  a smoothed ground (a quadratic along each path fitted to the cost map's ground, one round of
  outlier removal, held constant in the blind zone) and get extra smoothing for drawing only
  (3 m window: the 0.25 m grid zigzag showed as ~20 px wiggles at 15 m). Only run-time data is
  used (image, calibration, candidates, LiDAR ground), never poses or labels.
- **Random colors and order** per frame, seeded from sequence and frame so runs repeat
  (`assign_colors`, `frame_seed`, `prompts.shuffled`).
- **Prompts** (`nav/prompts.py`): Mode A ("keep driving … no destination"; prefer the trail) and
  Mode B (plus the direction word and its meaning). The tick rules are spelled out, unsafe =
  deep water or something the vehicle would hit, and grass, low bushes, mud and puddles are
  drivable ("prefer a dry path when one is as good"). Answer: `{"safe_until": {...}, "choice",
  "action", "reasoning"}`. The per-candidate judgement comes **first**: with the choice first,
  the 4B justified its pick by calling every other candidate "none" (first try, 6 frames).
- **History** (`history_text`): the last 3 decisions without colors ("2.0 s ago: continued
  slight left"), from the model's own earlier answers in the run, as in closed loop; or none.
- **Checker** (`parse_answer`): first JSON object, case-insensitive colors ("Red path" → red;
  two colors named → no choice), tick synonyms ("tick2", "10 m", 2), action synonyms.
- **Scoring** (`nav/scoring.py`, labels only as the answer key): choice matches the driver /
  is the closest candidate, chosen candidate safe, chose a puddle when a dry path existed, ticks
  exact / within one / too optimistic / too cautious, action correct.
- **Scripts:** `show_prompt.py` (the exact picture and prompt, plus the answer key) and
  `run_selector.py` (`--split`, `--every`, `--limit`, `--modes`, `--history own|none`, `--model`,
  `--tag`; resumable; → `outputs/selector/<tag>_mode{A,B}.jsonl` + `<tag>_summary.md`).
  `run_all.py` steps `prompt` and `selector` (slow: 3 frames).
- **Trial, Qwen3-VL-4B 4-bit, first 12 validation frames (00003, every 4th):** 100% valid
  answers, ~5 s per answer, 3.4 GB GPU memory. The chosen path was always safe, but it matched
  the driver in only 1–2 of 12 (from frame 168 on, no candidate matches: the driver turns), and
  it often picked the rightmost candidate. Ticks were never too optimistic but always too
  cautious: it never answers "all", stopping at "tick 2" even on clear paths. It mentions real
  objects (the orange barriers in 00003/42) beyond the 15 m the labels cover. This is the
  zero-shot starting point; further prompt tuning on validation frames would overfit the prompt,
  so it stops here, and fine-tuning (phase 5) is where this should improve.
- **The user's run (2026-11-23), 4B and 8B, val every 4th (83 frames asked, both modes):**

  | | 4B A | 4B B | 8B A | 8B B |
  |---|---|---|---|---|
  | valid answers | 100% | 100% | 100% | 100% |
  | choice matches the driver | 25% | 28% | 19% | 27% |
  | ticks exact | 2% | 2% | 37% | 45% |
  | ticks too optimistic / too cautious | 0 / 98% | 0 / 98% | 0 / 63% | 0 / 55% |
  | seconds per answer | 5.8 | 5.5 | 4.5 | 4.2 |

  - **Choice no better than chance:** a random pick would match the driver in 32% of these
    frames, the straightest candidate in 36%, the middle one in 40%; the best possible is 82%
    (in the other frames no candidate matches). The 4B and 8B agree on only ~55% of choices.
  - **Mode B ignores the word:** in the 15 "slight left" frames, the leftmost candidate would
    match 60%, the models 27%.
  - **Safety numbers are trivial on this drive:** 280 of 284 val candidates are labelled "all"
    (train: 95%), so "chosen safe", "action correct" and "always all" (99% of ticks exact) need
    no vision. Phase 4 must report each number next to these baselines; phase 5 needs balancing.
  - **Picture problem:** the candidates branch off one shortest-path tree, so about half share
    their first 10 m with another (within 0.1 m), and the line drawn last hid the other (in 223 of
    321 val frames a candidate was hidden up to 10 m, e.g. 00003/22, where the two hidden ones
    were the driver's).
- **User decision 2026-11-23: shared stretches are drawn side by side** (picked from mock-ups:
  now / 50% transparent / side by side / both). Transparency alone doesn't help: exactly
  overlapping lines blend into a new color (red over yellow → orange) that names no candidate,
  and without the outline the lines fade on grass. `spread_overlapping` shifts lines that would
  overlap (within 8 px at the same distance along) 7 px apart, in the left-to-right order of
  where they end, smoothed over 9 points; tick dots stay in the middle of a shared stretch.
  `PICTURE_VERSION = 2` is stored with each answer, and `run_selector.py` refuses to add to a file
  with answers to another version.
- **The user's rerun with picture version 2 (2026-11-23, tags `val_4B_v2`, `val_8B_v2`):** choice
  matches the driver 4B 25% / 27%, 8B 25% / 27% (Mode A / B); ticks exact 4B 1–2%, 8B 37% / 46%;
  4.2–5.2 s per answer. Choices changed in only 12–18% of frames, and the match rate stayed the
  same (8B Mode A 19% → 25%, 5 frames). So the hidden lines were not the main reason for the weak
  choices; e.g. in 00003/22 both models now pick red (the driver's), but overall the zero-shot
  choice is still at chance (random 32%). No listing-order bias (first listed chosen as often as
  chance); a color preference (4B: yellow 33, blue 28, red 13, magenta 9 of 83), which the random
  colors turn into noise instead of a fake score. One likely reason the driver match is low: on a
  wide trail, both ruts (or the trail and its edge) are equally reasonable, and only one is "the
  driver's". **Phase 3 is done** (pending the user's review); next is phase 4 when the user says.

### Phase 4: evaluation
- **Baselines:** random choice; the geometric choice (lowest-cost candidate); the straightest
  candidate; "always all safe, continue"; "follow the direction" without vision (the candidate
  closest to the Mode B bearing); and the **oracle** (always a correct candidate, always correct
  safe distances). On trails, the straightest or cheapest candidate is often right, so the VLM
  must beat these to show it reads the image rather than a shortcut.
- **Per-frame metrics:**
  - path choice: water / mud / vegetation on the chosen path, collision proxy (crosses
    hard-obstacle points), clearance; and in Mode B, agreement with the driver and L2 error
    against the driven path at 5 / 10 / 15 m;
  - safe distance: accuracy of the tick answers (exact, and within one tick), and how often it
    calls an unsafe stretch safe (the dangerous error);
  - stop: precision and recall of "stop".
- Report results separately for **frames with a real choice**, since many off-road frames have
  only one sensible path, and **per answer type** (e.g. how many true "stop" frames it catches,
  how many unsafe stretches it calls unsafe), not just overall accuracy: "always all safe,
  continue" is right most of the time.
- **Does it follow the direction?** (Mode B) Ask the same frame with different direction words
  ("bear left", then "bear right"); where a matching candidate exists, the choice should change.
- **Does it lean on the history?** Evaluate with and without the history, and with a wrong
  earlier decision in it. In closed loop the history holds the model's own (possibly wrong)
  decisions, not the driver's.
- ~~**Comparison with approach 1:** score approach 1's baseline and hybrid paths with the same
  safety metrics (and driven-path L2) on the same test frames.~~ **User decision 2026-11-23:**
  not in phase 4; the two approaches are compared at the end, using approach 1's existing report
  (its numbers come from 36 frames with destinations, so the comparison is side by side, not on
  the same frames). The "cheapest" baseline is roughly what approach 1's geometric planner picks.
- **User decision 2026-11-23: all 331 test frames** (sequence 00004, `--every 1`).

**Status (2026-11-23): written; checked on the validation runs; the test runs are the user's.**
- **Terrain numbers per path** (`nav/candidate_metrics.py`, stored by `build_candidates.py` as
  `metrics` on every candidate and on the driven path; the labels are unchanged, checked
  byte-for-byte): shares of water / puddle / mud / grass-bush / tree-log / no ground truth,
  closest hard obstacle (`clearance_m`), near miss (< 0.75 m, half the vehicle's width), slope
  and bump height. Hard obstacles count only if they stand more than 0.3 m above the ground
  (new `min_height` in `eval.metrics.hard_obstacle_xy`; approach 1's default is unchanged):
  without it, ~1/3 of all paths, the driver's included, had a "near miss", all from "person"
  labels on flat ground (median 6 cm high). After the fix: 6–9% for candidates and 8% for the
  driven path (val), mostly branches 1.65–1.8 m up that the vehicle drove under.
- **Evaluation** (`nav/evaluation.py`, `scripts/evaluate.py`): a choice is a weight per candidate
  (one-hot; uniform for random, so random is the exact average). Baselines: random, straightest,
  cheapest, Mode B "toward the word" (end bearing closest to the word: straight 0°, slight 17.5°,
  bear 35°), driver (oracle), and the **driven path itself as a reference column**. Rows: matches
  the driver (all / real choice / turn frames), closest candidate, safe to tick 1 / end, puddle
  when a dry path existed, terrain shares, closest hard obstacle, near misses, slope, distance
  from the driven path at 5 / 10 / 15 m. Ticks: "always all" vs runs, with exact / within one /
  too optimistic / too cautious, "unsafe stretch called safe past it", and per true answer type.
  Continue / stop: "always continue" vs runs, stop recall and precision. All columns use the
  same frames (those every run answered).
- **Behaviour tests** (`run_selector.py`): `--direction WORD` gives Mode B the same word in every
  frame (`evaluate.py --direction-test LEFT RIGHT`: followed / ignored / reversed, and how often
  the leftmost / rightmost is picked); `--history wrong` replaces the latest history entry with a
  sharp turn not taken, opposite to the real one (or a seeded random side) (`--history-test OWN
  WRONG`: moved toward / away / same). Default tags get `_nohist`, `_wronghist`, `_dir-<word>`.
- **First look, validation (val every 4th, picture v2):** the cheapest candidate matches the
  driver in 47% (turn frames 58%), straightest 36%, random 32%, "toward the word" 42%, oracle 82%;
  the 4B and 8B 25–27%. So the geometric planner's own pick is far ahead of the zero-shot VLM. The
  driver's center line runs over much less "mud" than the candidates (1.7% vs ~10%): the wheels
  are in the ruts, the center on the grass between them, while candidates follow the ruts.
- **Zero-shot results on the test drive (2026-11-23; the user ran everything on Polaris; all 331
  frames, 328 with candidates; `outputs/eval/test_evaluation.md`):**
  - **Choice matches the driver:** Mode A: random 43%, straightest 67%, cheapest 48%, oracle 89%;
    4B 43% (no history 45%, wrong history 45%), 8B 42%. Mode B: toward the word 67%; 4B 48%,
    8B 47%. In turn frames (83 of 328, mostly "slight left"): Mode A 4B 18%, 8B 19% (random 33%,
    straightest 43%); Mode B 4B 41%, 8B 35% (toward the word 45%). So zero-shot the VLM chooses
    at chance level, far below "straightest"; the test drive is mostly straight (245 of 328
    frames), which is why straightest and random score higher than on validation.
  - **Position bias:** the 8B picks the rightmost candidate in 222 of 326 Mode A frames (68%);
    the 4B prefers middle / right.
  - **The word is understood:** with "bear left" vs "bear right" the choice ends further left
    under "bear left" in 89% of 326 frames, never the reverse; rightmost under "bear right" 97%,
    leftmost under "bear left" only 48% (the right lean again). In normal Mode B, where the word
    is mostly "straight", it doesn't pick the straightest.
  - **History is ignored:** a made-up sharp turn changes the choice in only 5% (Mode A) / 8%
    (Mode B) of frames, as often away from it as toward it; no history ≈ own history.
  - **Safety and stop can't be measured on this drive:** only 6 of 1,123 candidates have an unsafe
    stretch (no model called them safe past it; "always all" got all 6 wrong), 7 are labelled
    "tick 2", and there are no stop frames. Ticks: the 4B says "all" almost never (8–17 of
    1,116), the 8B is right 40% / 50% (A / B), both never too optimistic.
  - **Puddles** (13 frames only): the 8B took the puddle when a dry path existed in 31% (4 of
    13), random 46%, the 4B 54%. Terrain, near misses and slope are about the same for every
    chooser and similar to the driven path.
  - **Target for phase 5:** beat "straightest" / "toward the word" (67%) on driver match, and
    show the ticks distinguish paths; the oracle (89%) is the ceiling.
- **Phase 4 is done** (closed by the user, 2026-11-23).

### Phase 5: fine-tuning
- **QLoRA supervised fine-tuning of Qwen3-VL-4B on the laptop** (RTX 5060, 8 GB): the base model
  stays frozen in 4-bit, only small LoRA adapters are trained. Batch size 1 with gradient
  accumulation, gradient checkpointing, 800×600 images (as DriveVLN), a mix of Mode A and Mode B
  examples, shuffled color assignment (so the model can't learn "always blue"), and paraphrased
  instructions.
- **Against overfitting** (user decision 2026-11-22, see section 8):
  - **History dropout:** in some training examples leave the history out, and in some put a
    wrong earlier decision in it, so the model doesn't learn "repeat the last answer" (in the
    training data the history comes from the driver, who is almost always right).
  - **Balance:** most frames are "straight", ~90% of candidates are "safe all", and "stop" is
    rare (~4%). Use the rare frames (turns, unsafe candidates, stop) more often and the easy
    straight ones less, so the model can't score well by always answering "all safe, continue".
  - **Mirroring:** flip the image left-right, mirror the candidates, and swap left / right in
    the Mode B word. This doubles the turn data and balances left and right (training has 382
    right-turn and 309 left-turn frames; the test drive has 20 "slight right" and no "bear right").
  - **Frames where no candidate matches the driver** (user decision 2026-11-23: an experiment).
    238 of the 1,527 training frames with candidates (15%; 44–49% of the "bear" frames, 3% of
    "straight"): the driver turned off a trail that is still open ahead (235 of them have a
    candidate safe to tick 1), usually out of the camera's view. Not dead ends: those are the 26
    frames with no candidate at all, where the vehicle stops without asking the VLM. Train with
    and without these frames and compare. Options for "with": the safety part only (no choice
    target) in Mode A, and the candidate closest to the driver in Mode B, where the direction
    word carries the intent.
  - **Photometric augmentation:** small random changes of brightness, contrast, color, and blur,
    so the model doesn't fit RELLIS's look (simulator images look different).
  - **Few epochs, early stopping:** frames 0.5 s apart on the same trail are near-duplicates, so
    the ~1,550 training frames hold far fewer distinct examples than it seems. Validate on 00003,
    stop when it stops improving, and use the test drive (00004) only for the final numbers.
  - **Skip uncertain frames:** leave out frames where most of the candidates' ground has no
    ground truth.
- Compare zero-shot vs fine-tuned on the test sequence, in both modes. Zero-shot can't have
  overfit RELLIS, so it's the reference for how much fine-tuning helps outside it.
- **Check general ability:** after fine-tuning, run a few non-RELLIS images (e.g. from the
  simulator) to make sure the model still judges drivable ground sensibly.
- **Optional afterwards:** GRPO with a per-frame reward (choice safety + agreement with the
  driver + tick accuracy), which needs no simulator. DriveVLN's route-level reward needs closed
  loop and is out of scope.
- **8B:** decide after the 4B results. Fine-tuning an 8B doesn't fit in 8 GB; it would run on the
  Polaris H200.

### Phase 6 (stretch): closed loop
- An off-road simulator (e.g. Isaac Sim, AirSim/Colosseum, BeamNG, or an off-road CARLA map)
  would allow real closed-loop driving. Revisit only if the offline results call for it.
- **What it needs:** an RGB camera (the VLM's input) and a LiDAR or other depth sensor mounted
  roughly like RELLIS's (~1.2 m up), because the candidates come from the LiDAR cost map. The
  simulator's own ground truth (collisions, water, distance covered) is used only to score runs.
- **No labels at run time:** the RELLIS labels (and ours made from them) are only training
  targets and the answer key for scoring. Candidates come from LiDAR geometry only, and the VLM
  sees only the camera image with the candidates drawn on it, so the same pipeline runs in
  closed loop without labels.
- **Closed loop tests what open loop can't:** the training frames all lie on the driver's own
  good path, so the model never sees a state where it has already drifted. In closed loop small
  errors add up and lead to such states; this is the main thing a simulator run would measure.

## 6. Decisions

**Made (2026-11-22):**
1. **Evaluation mode:** offline, open-loop on RELLIS-3D; a simulator only if needed later.
2. **Task:** no destinations. "Keep going until stopped" (Mode A) and "keep going + coarse
   direction" (Mode B), both evaluated; every answer includes a path choice, a safe distance per
   candidate (ticks), and continue/stop.
3. **Candidates:** cost-map fan + A\* first; curved anchors as an optional later comparison.
4. **Model and fine-tuning:** Qwen3-VL-4B, QLoRA on the laptop. DriveVLN used Qwen2.5-VL-3B; the
   4B is the closest current model of that size, framed as a starting point for future work.
   8B decided after the 4B results (fine-tuning it would need the H200).
5. **Cost grid for the candidates:** relaxed (only hard obstacles over 1.5 m block; lower
   vegetation costs 10), chosen after the phase 2 comparison with the strict grid.

**Settings (the recommended defaults, accepted 2026-11-22):**

| # | Setting | Value |
|---|---|---|
| 1 | Split by sequence | train 00000–00002, validate 00003, test 00004 |
| 2 | Decision frames | about 2 Hz (≈ 2,700 frames in total) |
| 3 | Branch | `approach-2-mapless-vln`, from `main` |
| 4 | Candidates | up to 4 per frame, about 15 m long, ticks at 5 / 10 / 15 m |
| 5 | "Driver's path" | candidates within 1 m mean distance of the driven path (several may count) |
| 6 | Unsafe | water or a hard obstacle (approach 1's classes; a hard obstacle only where the LiDAR sees something standing); mud and grass count as safe. **Puddles are "caution"** (changed 2026-11-22: the driver drives through them): recorded and scored as "puddle on the chosen path when a dry candidate existed", but not unsafe |
| 7 | Stop is correct | every candidate is unsafe before its first tick (5 m) |
| 8 | Mode B directions | "bear left", "slight left", "straight", "slight right", "bear right", from the driven path ~50 m ahead |
| 9 | History | the last 3 decisions as a short text summary |
| 10 | Image size | 800×600 |
| 11 | Fine-tuning | LoRA rank 64, learning rate 1e-4, 5 epochs (DriveVLN's); GRPO decided after the SFT results |
| 12 | Models in the report | Qwen3-VL-4B zero-shot and fine-tuned (main); the 8B zero-shot as a reference |
| 13 | Approach-1 comparison | approach 1's geometric baseline on the same test frames, same safety metrics; its VLM hybrid only if time allows (needs new 8B runs, ~1.5 h) |

## 7. Risks and limitations to expect

- **RELLIS isn't a navigation dataset.** The driver mostly follows trails, so real forks are
  rare; many frames have one obviously right choice. Report "real choice" frames separately.
- **Open-loop only:** a wrong choice is scored but never driven, so errors don't compound the way
  they would on a real vehicle.
- **The driver's path isn't the only good path.** That's why Mode A is scored on safety, and
  Mode B uses a tolerance and a direction.
- **The Mode B direction comes from the driver's own future path**, so it leaks a little of the
  answer; it's kept coarse, and Mode A is the leak-free comparison.
- **Small and correlated data** for fine-tuning: a few thousand frames from one campus. The test
  sequence must stay unseen.
- **Labels are only as good as the ground truth:** coarse RELLIS labels, ruts labelled as mud,
  and ~20% of path cells without ground truth (mostly the blind zone).
- **Latency:** about 10–14 s per answer on the laptop. Fine offline; a real vehicle would need a
  faster model or fewer decisions.
- **The blind zone and cost-map limits** carry over, since the candidates come from the same cost map.

## 8. Safeguards against overfitting and shortcuts (agreed 2026-11-22)

The user's concern: the model must not learn things that won't hold in a closed-loop simulator,
where there are no labels and its own mistakes shape what it sees next. Where each one goes:

| Safeguard | Why | Phase |
|---|---|---|
| Labels only as targets / answer key, never as input | They don't exist in closed loop | 2 (done: candidates use LiDAR geometry only) |
| Random color assignment and prompt order, same drawing style, no costs in the prompt | So it can't learn "red is usually right" or "pick the middle one" | 3 and 5 |
| Simple baselines: random, straightest, cheapest, always "all safe, continue" | The VLM must beat them to show it reads the image | 4 |
| Results per answer type (stop recall, unsafe calls) | Overall accuracy hides "always continue" | 4 |
| Same frame with different direction words | Checks it follows the instruction | 4 |
| History dropout and wrong history; evaluate with and without | In closed loop the history holds its own mistakes | 3, 4, 5 |
| Balanced training data (more turns, unsafe, stop) | Most frames are easy and straight | 5 |
| Mirroring with left / right swapped | Doubles turns; test has no "bear right" | 5 |
| Brightness / color / blur augmentation | Simulator images look different from RELLIS | 5 |
| Few epochs, early stopping on 00003, test 00004 only at the end | Neighbouring frames are near-duplicates | 5 |
| Skip frames with mostly unknown ground | Don't train on guesses | 5 |
| Zero-shot reported next to fine-tuned; a few non-RELLIS images after fine-tuning | Shows what fine-tuning changes outside RELLIS | 5 |
| Puddles as caution, not unsafe | Labels that contradict the driver teach RELLIS habits, not driving | 2 (done 2026-11-22) |
