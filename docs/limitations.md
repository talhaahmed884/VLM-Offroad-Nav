# Limitations

Known limits of the evaluation, for the report. The numbers refer to the final evaluation
(`scripts/evaluate.py`, 36 frames, conditions: baseline, hybrid with Qwen3-VL-4B / 8B in 4-bit,
and hybrid with oracle hints from the labels).

## Setup and scope

1. **Offline and per frame, not a drive.** Each frame is planned on its own, from a recorded scan.
   There is no closed-loop driving (the vehicle follows its recorded path, not ours) and no
   replay over time. So the *asynchronous* part of the research question is only argued, not
   measured: the VLM takes about 13 s per answer, during which a vehicle at 2–3 m/s moves
   30–40 m, more than the 25 m range in which hints are used. A time replay would need the
   scan poses (not downloaded), hints stored in world coordinates, and VLM answers on many
   more frames.
2. **A fixed target.** The goal is always 20 m straight ahead, not a mission waypoint or the path
   the vehicle actually took.
3. **Small, hand-picked test set.** 36 frames chosen from the label statistics (8 mud, 8 water,
   8 bush, 4 rubble, 8 easy). Differences of one or two frames are within noise. RELLIS-3D has
   no close-range rocks or steep slopes, so "rocky slopes" from the proposal is not covered;
   the rubble frames only have rubble far away. One "easy" frame (00002/102) is mostly tall
   grass and bush.
4. **Tuned on the test frames.** The cost-map settings (phase 3) and the fusion settings
   (multipliers, unblocking rule, 25 m limit; phase 5) were chosen while looking at these same
   frames. There is no held-out set, so the numbers are likely optimistic for both planners.
5. **One run per model, greedy decoding.** No repeated runs or confidence intervals; the report
   gives frame-by-frame wins/losses instead.

## The VLM

6. **Small models, zero-shot, compressed.** Qwen3-VL-4B and 8B, 4-bit, on an 8 GB laptop GPU, with
   no fine-tuning. Larger models (e.g. 30B-class on the H200 cluster) were not evaluated.
7. **Low coverage.** The 8B hints cover only about 12% of the visible ground within 25 m, partly
   because the prompt caps the hints (6 entries × 8 cells) to stop runaway lists. The oracle,
   which hints every cell, shows how much is lost: success 32 vs 31, mud on the path in mud
   frames 11% vs 31%, bush frames successful 7/8 vs 5/8.
8. **Weak routes.** The waypoints are often a straight column to the goal, and 6 of 36 routes pass
   through cells the model itself marked "avoid". They are used only as a gentle cost discount.
9. **The prompt includes the vehicle's rules** (grass and soft bushes drivable; trees, rocks,
   people, deep water not; mud and puddles risky). This is fixed vehicle knowledge, not per-frame
   information, but it tells the model part of the answer.

## The fusion design

10. **Blind zone.** The first ~4.4 m in front of the vehicle are below the camera's view, so the VLM
    cannot change them. In the tall-grass frame 00001/2132 the grass the VLM unblocks is an
    unreachable island, and the hybrid takes the same 1.43× detour as the baseline, even with
    oracle hints.
11. **Coarse far cells.** Near the horizon one grid cell covers a strip of ground up to ~24 m deep.
    An "impassable" hint on such a cell blocks far too much ground: the oracle fails easy
    00003/1060 (stops 1.2 m short) and water 00000/120 (stops 2.5 m short, blocked by a person
    hint on a far cell; the person is actually 13–14 m ahead, not at the target).
12. **A small safety cost.** The hybrids pass slightly closer to hard obstacles (closest approach
    1.55 m baseline vs 1.47 m for the 8B hybrid; 4 frames better, 7 worse), for example by
    hugging the trail edge to avoid a puddle. Clearance never drops much below about 1 m because
    of the 0.75 m obstacle inflation.

## Ground truth and metrics

13. **Coarse labels.** RELLIS-3D labels whole wet areas as puddle (including ridges and grass edges)
    and labels plain trail ruts as mud, so "mud on the path" is not a failure in itself.
14. **20% of path cells have no ground truth**, mostly the first ~5 m (no LiDAR points near the
    vehicle and outside the camera's view). Camera labels show the nearest surface, so some
    "bush" cells may be bush in front of the ground; only 0–5% were caught as occluded.
15. **Threshold choices.** "Goal blocked" = a hard obstacle within 1.5 m of the target; success
    then requires stopping within 2.5 m. The tie tolerances in the wins/losses table (0.01 share,
    0.25 m shortfall, 0.1 m clearance, 0.02 detour) are judgment calls. "Slope" mostly measures
    the uphill incline of the trails (about 6–7 cm per m), not bumpiness.
16. **Timing.** Geometry times (about 0.06 s baseline, 0.08 s hybrid per frame) are single-threaded
    Python on a laptop CPU. VLM times are on an RTX 5060 laptop GPU and include image preprocessing.
