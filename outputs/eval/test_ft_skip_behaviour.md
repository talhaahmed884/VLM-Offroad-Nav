## Evaluation: test, 328 frames with candidates

Frames without any candidate (the vehicle stops without asking): 3. Baselines and runs are scored on the same frames.

[a, b] = 95% confidence interval (Wilson for shares, normal approximation for means): with this many frames the true value is probably in that range. ADE = mean distance from the driven path over 5 / 10 / 15 m. For the random baseline the distances are averages over all candidates of a frame.

### Path choice, Mode A (keep going)

| metric | random | straightest | cheapest | driver (oracle) | driven path (reference) | test_ft-nomatch-skip-epoch4 |
|---|---|---|---|---|---|---|
| frames with a choice | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 |
| choice matches the driver [95% interval] | 43% [38, 48] | 67% [62, 72] | 48% [43, 54] | 89% [86, 92] | 100% [99, 100] | 66% [61, 71] |
| ... in frames with a real choice | 43% | 67% | 48% | 90% | 100% | 67% |
| ... in straight frames / turn frames (driver's word) | 46% [40, 53] / 33% [24, 44] | 75% [69, 80] / 43% [33, 54] | 51% [44, 57] / 42% [32, 53] | 95% [92, 97] / 72% [62, 81] | 100% [98, 100] / 100% [96, 100] | 70% [64, 75] / 57% [46, 67] |
| choice is the driver's closest candidate | 31% | 58% | 43% | 100% | 100% | 61% |
| wrong choices: left / right of the driver's candidate | 57% / 43% | 61% / 39% | 74% / 26% | - / - | - / - | 63% / 37% |
| chosen position (2+ candidates): leftmost / middle / rightmost | 31% / 39% / 31% | 10% / 83% / 7% | 35% / 52% / 13% | 21% / 63% / 16% | - / - / - | 29% / 54% / 18% |
| chosen path safe to tick 1 / to the end | 100% / 99% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| chose a puddle when a dry path existed | 46% | 62% | 77% | 54% | - | 62% |
| water / puddle / mud on the path (share of length) | 0.0% / 1.3% / 22.5% | 0.0% / 1.3% / 25.4% | 0.0% / 1.8% / 25.4% | 0.0% / 1.3% / 25.6% | 0.0% / 0.9% / 18.1% | 0.0% / 1.5% / 25.5% |
| grass-bush / tree-log / no ground truth | 49.0% / 0.1% / 26.7% | 46.4% / 0.1% / 26.4% | 46.1% / 0.0% / 26.3% | 46.0% / 0.1% / 26.5% | 52.8% / 0.0% / 27.9% | 46.2% / 0.1% / 26.4% |
| closest hard obstacle (mean) | 2.72 m | 2.82 m | 2.85 m | 2.91 m | 2.93 m | 2.90 m |
| near misses (hard obstacle < 0.75 m) | 8.1% | 5.2% | 3.7% | 4.3% | 10.1% | 4.9% |
| ground slope (cm per m) | 5.2 | 5.0 | 4.8 | 5.0 | 5.3 | 4.9 |
| distance from the driven path at 5 / 10 / 15 m | 0.66 m / 1.21 m / 2.85 m | 0.56 m / 0.95 m / 1.70 m | 0.73 m / 1.16 m / 1.89 m | 0.47 m / 0.63 m / 1.14 m | 0.00 m / 0.00 m / 0.02 m | 0.60 m / 0.90 m / 1.43 m |
| distance at 15 m: mean [95% interval] | 2.85 m [2.72, 2.97] | 1.70 m [1.53, 1.88] | 1.89 m [1.72, 2.07] | 1.14 m [1.02, 1.25] | 0.02 m [0.02, 0.02] | 1.43 m [1.28, 1.58] |
| distance at 15 m: median / 90th percentile | 2.62 m / 4.00 m | 1.30 m / 3.74 m | 1.46 m / 3.88 m | 0.87 m / 2.16 m | 0.02 m / 0.03 m | 1.17 m / 2.66 m |
| mean distance over 5 / 10 / 15 m (ADE) [95% interval] | 1.57 m [1.50, 1.64] | 1.07 m [0.98, 1.16] | 1.26 m [1.17, 1.35] | 0.75 m [0.69, 0.81] | 0.01 m [0.01, 0.01] | 0.98 m [0.90, 1.06] |

### Path choice, Mode B (keep going + the driver's direction word)

| metric | random | straightest | cheapest | toward the word | driver (oracle) | driven path (reference) | test_ft-nomatch-skip-epoch4 |
|---|---|---|---|---|---|---|---|
| frames with a choice | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 |
| choice matches the driver [95% interval] | 43% [38, 48] | 67% [62, 72] | 48% [43, 54] | 67% [62, 72] | 89% [86, 92] | 100% [99, 100] | 66% [61, 71] |
| ... in frames with a real choice | 43% | 67% | 48% | 67% | 90% | 100% | 66% |
| ... in straight frames / turn frames (driver's word) | 46% [40, 53] / 33% [24, 44] | 75% [69, 80] / 43% [33, 54] | 51% [44, 57] / 42% [32, 53] | 75% [69, 80] / 45% [34, 55] | 95% [92, 97] / 72% [62, 81] | 100% [98, 100] / 100% [96, 100] | 70% [64, 76] / 54% [44, 65] |
| choice is the driver's closest candidate | 31% | 58% | 43% | 62% | 100% | 100% | 61% |
| wrong choices: left / right of the driver's candidate | 57% / 43% | 61% / 39% | 74% / 26% | 73% / 27% | - / - | - / - | 63% / 37% |
| chosen position (2+ candidates): leftmost / middle / rightmost | 31% / 39% / 31% | 10% / 83% / 7% | 35% / 52% / 13% | 26% / 64% / 10% | 21% / 63% / 16% | - / - / - | 29% / 53% / 18% |
| chosen path safe to tick 1 / to the end | 100% / 99% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| chose a puddle when a dry path existed | 46% | 62% | 77% | 62% | 54% | - | 54% |
| water / puddle / mud on the path (share of length) | 0.0% / 1.3% / 22.5% | 0.0% / 1.3% / 25.4% | 0.0% / 1.8% / 25.4% | 0.0% / 1.3% / 25.5% | 0.0% / 1.3% / 25.6% | 0.0% / 0.9% / 18.1% | 0.0% / 1.4% / 25.3% |
| grass-bush / tree-log / no ground truth | 49.0% / 0.1% / 26.7% | 46.4% / 0.1% / 26.4% | 46.1% / 0.0% / 26.3% | 46.0% / 0.1% / 26.7% | 46.0% / 0.1% / 26.5% | 52.8% / 0.0% / 27.9% | 46.6% / 0.0% / 26.4% |
| closest hard obstacle (mean) | 2.72 m | 2.82 m | 2.85 m | 2.80 m | 2.91 m | 2.93 m | 2.90 m |
| near misses (hard obstacle < 0.75 m) | 8.1% | 5.2% | 3.7% | 6.1% | 4.3% | 10.1% | 5.2% |
| ground slope (cm per m) | 5.2 | 5.0 | 4.8 | 4.9 | 5.0 | 5.3 | 4.9 |
| distance from the driven path at 5 / 10 / 15 m | 0.66 m / 1.21 m / 2.85 m | 0.56 m / 0.95 m / 1.70 m | 0.73 m / 1.16 m / 1.89 m | 0.61 m / 0.92 m / 1.49 m | 0.47 m / 0.63 m / 1.14 m | 0.00 m / 0.00 m / 0.02 m | 0.61 m / 0.90 m / 1.44 m |
| distance at 15 m: mean [95% interval] | 2.85 m [2.72, 2.97] | 1.70 m [1.53, 1.88] | 1.89 m [1.72, 2.07] | 1.49 m [1.36, 1.63] | 1.14 m [1.02, 1.25] | 0.02 m [0.02, 0.02] | 1.44 m [1.29, 1.59] |
| distance at 15 m: median / 90th percentile | 2.62 m / 4.00 m | 1.30 m / 3.74 m | 1.46 m / 3.88 m | 1.31 m / 2.93 m | 0.87 m / 2.16 m | 0.02 m / 0.03 m | 1.16 m / 2.66 m |
| mean distance over 5 / 10 / 15 m (ADE) [95% interval] | 1.57 m [1.50, 1.64] | 1.07 m [0.98, 1.16] | 1.26 m [1.17, 1.35] | 1.01 m [0.93, 1.08] | 0.75 m [0.69, 0.81] | 0.01 m [0.01, 0.01] | 0.98 m [0.90, 1.07] |

### Safe distance (ticks)

| metric | always "all" | test_ft-nomatch-skip-epoch4 A | test_ft-nomatch-skip-epoch4 B |
|---|---|---|---|
| ticks answered | 1123/1123 | 1123/1123 | 1123/1123 |
| exact / within one tick | 99% / 100% | 99% / 100% | 99% / 100% |
| mean tick error (ticks) | 0.006 | 0.006 | 0.006 |
| too optimistic / too cautious | 1% / 0% | 1% / 0% | 1% / 0% |
| unsafe stretch called safe past it | 6/6 | 6/6 | 6/6 |
| hazard flags ("not safe to the end"): labelled / flagged / both | 7 / 0 / 0 | 7 / 0 / 0 | 7 / 0 / 0 |
| hazard flags: precision / recall / F1 | - / 0.00 / - | - / 0.00 / - | - / 0.00 / - |
| label "none": answered exactly | - (none) | - (none) | - (none) |
| label "tick 1": answered exactly | - (none) | - (none) | - (none) |
| label "tick 2": answered exactly | 0/7 | 0/7 | 0/7 |
| label "all": answered exactly | 1116/1116 | 1116/1116 | 1116/1116 |

### Continue / stop

| metric | always continue | test_ft-nomatch-skip-epoch4 A | test_ft-nomatch-skip-epoch4 B |
|---|---|---|---|
| action correct | 100% | 100% | 100% |
| stop frames (label) / said stop / both | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| stop recall / precision | - / - | - / - | - / - |
| stop F1 | - | - | - |

### Paired comparison, Mode A (driver match on the same frames)

Only the frames where the two disagree decide; a p-value below 0.05 means the difference is unlikely to be luck.

| metric | test_ft-nomatch-skip-epoch4 vs straightest | test_ft-nomatch-skip-epoch4 vs cheapest |
|---|---|---|
| frames | 328 | 328 |
| both right / both wrong | 168 / 59 | 137 / 88 |
| only the first right | 50 | 81 |
| only the second right | 51 | 22 |
| p-value (exact McNemar) | 1.000 | 0.000 |

### Paired comparison, Mode B (driver match on the same frames)

Only the frames where the two disagree decide; a p-value below 0.05 means the difference is unlikely to be luck.

| metric | test_ft-nomatch-skip-epoch4 vs straightest | test_ft-nomatch-skip-epoch4 vs cheapest |
|---|---|---|
| frames | 328 | 328 |
| both right / both wrong | 167 / 59 | 137 / 89 |
| only the first right | 50 | 80 |
| only the second right | 52 | 22 |
| p-value (exact McNemar) | 0.921 | 0.000 |

### Speed (as measured on the machine that ran the selector)

| metric | test_ft-nomatch-skip-epoch4 A | test_ft-nomatch-skip-epoch4 B |
|---|---|---|
| answers timed | 328 | 328 |
| seconds per answer: mean / median / 95th percentile | 1.15 / 1.18 / 1.24 | 1.17 / 1.18 / 1.24 |
| answer length (tokens, mean) | 39 | 39 |

### Does the choice follow the word? ("bear left" vs "bear right", Mode B)

| metric |  |
|---|---|
| frames (2+ candidates, both answered) | 326 |
| choice further left with "bear left" (followed) | 2% |
| same candidate both times (ignored) | 95% |
| choice further left with "bear right" (reversed) | 3% |
| leftmost with "bear left" / rightmost with "bear right" | 26% / 16% |

### Does a wrong history pull the choice? (Mode A: test_ft-nomatch-skip-epoch4 vs test_ft-nomatch-skip-epoch4_wronghist)

| metric |  |
|---|---|
| frames (2+ candidates, both answered) | 325 |
| moved toward the made-up turn | 3% |
| moved away from it | 2% |
| same candidate | 95% |

### Does a wrong history pull the choice? (Mode B: test_ft-nomatch-skip-epoch4 vs test_ft-nomatch-skip-epoch4_wronghist)

| metric |  |
|---|---|
| frames (2+ candidates, both answered) | 325 |
| moved toward the made-up turn | 2% |
| moved away from it | 2% |
| same candidate | 96% |

