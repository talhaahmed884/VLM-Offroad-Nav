### Overall results (means over all frames)

| Condition | Success | Reached | Short (m) | Detour | Water | Mud | Veg. ok | Veg. bad | Hard min (m) | Slope (cm/m) | Plan (s) | VLM (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 30/36 | 25/36 | 1.30 | 1.10 | 4% | 11% | 62% | 0% | 1.55 | 6.39 | 0.06 | – |
| Hybrid 4B | 30/36 | 26/36 | 1.26 | 1.09 | 4% | 12% | 60% | 0% | 1.44 | 6.57 | 0.08 | 12.46 |
| Hybrid 8B | 31/36 | 26/36 | 1.29 | 1.10 | 3% | 11% | 63% | 0% | 1.47 | 6.45 | 0.08 | 13.77 |
| Hybrid oracle | 32/36 | 28/36 | 0.96 | 1.13 | 2% | 5% | 70% | 1% | 1.43 | 6.66 | 0.09 | – |

### Share of the path through water or puddles (lower is better)

| Category | Baseline | Hybrid 4B | Hybrid 8B | Hybrid oracle |
|:---|---:|---:|---:|---:|
| mud | 3% | 2% | 2% | 3% |
| water | 17% | 16% | 11% | 8% |
| bush | 0% | 0% | 0% | 0% |
| rubble | 0% | 0% | 0% | 0% |
| easy | 0% | 0% | 0% | 0% |
| All | 4% | 4% | 3% | 2% |

### Share of the path through mud (RELLIS-3D labels trail ruts as mud)

| Category | Baseline | Hybrid 4B | Hybrid 8B | Hybrid oracle |
|:---|---:|---:|---:|---:|
| mud | 32% | 34% | 31% | 11% |
| water | 14% | 15% | 14% | 7% |
| bush | 3% | 3% | 3% | 3% |
| rubble | 1% | 0% | 1% | 1% |
| easy | 0% | 0% | 0% | 0% |
| All | 11% | 12% | 11% | 5% |

### Frames where the path reached the target, or stopped within 2.5 m of a blocked target

| Category | Baseline | Hybrid 4B | Hybrid 8B | Hybrid oracle |
|:---|---:|---:|---:|---:|
| mud | 7/8 | 7/8 | 7/8 | 8/8 |
| water | 8/8 | 8/8 | 8/8 | 7/8 |
| bush | 4/8 | 4/8 | 5/8 | 7/8 |
| rubble | 4/4 | 4/4 | 4/4 | 4/4 |
| easy | 7/8 | 7/8 | 7/8 | 6/8 |
| All | 30/36 | 30/36 | 31/36 | 32/36 |

### Closest approach to a hard obstacle along the path, m (mean over frames)

| Category | Baseline | Hybrid 4B | Hybrid 8B | Hybrid oracle |
|:---|---:|---:|---:|---:|
| mud | 0.95 | 0.94 | 1.00 | 1.05 |
| water | 1.19 | 1.00 | 1.21 | 1.12 |
| bush | 1.18 | 1.18 | 1.26 | 1.16 |
| rubble | 2.11 | 1.92 | 1.95 | 1.84 |
| easy | 2.60 | 2.41 | 2.18 | 2.20 |
| All | 1.55 | 1.44 | 1.47 | 1.43 |

### Frame by frame against the baseline: success frames gained/lost, and wins/losses/ties per metric (ties within 0.01 share, 0.25 m shortfall, 0.1 m clearance, 0.02 detour)

| Condition | Success +/− | Water W/L/T | Mud W/L/T | Shortfall W/L/T | Hard min W/L/T | Detour W/L/T |
|:---|---:|---:|---:|---:|---:|---:|
| Hybrid 4B | +0/−0 | 5/1/30 | 6/5/25 | 3/1/32 | 3/6/27 | 7/4/25 |
| Hybrid 8B | +1/−0 | 7/3/26 | 6/4/26 | 1/0/35 | 4/7/25 | 7/8/21 |
| Hybrid oracle | +5/−3 | 5/1/30 | 14/1/21 | 9/3/24 | 6/11/19 | 5/16/15 |
