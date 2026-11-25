## Evaluation: test, 328 frames with candidates

Frames without any candidate (the vehicle stops without asking): 3. Baselines and runs are scored on the same frames.

[a, b] = 95% confidence interval (Wilson for shares, normal approximation for means): with this many frames the true value is probably in that range. ADE = mean distance from the driven path over 5 / 10 / 15 m. For the random baseline the distances are averages over all candidates of a frame.

### Path choice, Mode A (keep going)

| metric | random | straightest | cheapest | driver (oracle) | driven path (reference) | test_Qwen3-VL-4B-Instruct | test_Qwen3-VL-4B-Instruct_nohist | test_Qwen3-VL-4B-Instruct_wronghist | test_Qwen3-VL-8B-Instruct |
|---|---|---|---|---|---|---|---|---|---|
| frames with a choice | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 |
| choice matches the driver [95% interval] | 43% [38, 48] | 67% [62, 72] | 48% [43, 54] | 89% [86, 92] | 100% [99, 100] | 43% [38, 49] | 45% [39, 50] | 45% [40, 50] | 42% [37, 47] |
| ... in frames with a real choice | 43% | 67% | 48% | 90% | 100% | 43% | 44% | 45% | 42% |
| ... in straight frames / turn frames (driver's word) | 46% [40, 53] / 33% [24, 44] | 75% [69, 80] / 43% [33, 54] | 51% [44, 57] / 42% [32, 53] | 95% [92, 97] / 72% [62, 81] | 100% [98, 100] / 100% [96, 100] | 52% [46, 58] / 18% [11, 28] | 52% [46, 58] / 22% [14, 32] | 54% [48, 60] / 18% [11, 28] | 49% [43, 56] / 19% [12, 29] |
| choice is the driver's closest candidate | 31% | 58% | 43% | 100% | 100% | 31% | 30% | 33% | 23% |
| wrong choices: left / right of the driver's candidate | 57% / 43% | 61% / 39% | 74% / 26% | - / - | - / - | 38% / 62% | 42% / 58% | 35% / 65% | 19% / 81% |
| chosen position (2+ candidates): leftmost / middle / rightmost | 31% / 39% / 31% | 10% / 83% / 7% | 35% / 52% / 13% | 21% / 63% / 16% | - / - / - | 17% / 44% / 39% | 21% / 44% / 35% | 16% / 45% / 40% | 10% / 22% / 68% |
| chosen path safe to tick 1 / to the end | 100% / 99% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 99% | 100% / 100% | 100% / 99% | 100% / 100% |
| chose a puddle when a dry path existed | 46% | 62% | 77% | 54% | - | 54% | 46% | 54% | 31% |
| water / puddle / mud on the path (share of length) | 0.0% / 1.3% / 22.5% | 0.0% / 1.3% / 25.4% | 0.0% / 1.8% / 25.4% | 0.0% / 1.3% / 25.6% | 0.0% / 0.9% / 18.1% | 0.0% / 1.3% / 21.3% | 0.0% / 1.2% / 21.3% | 0.0% / 1.3% / 21.6% | 0.0% / 1.0% / 20.0% |
| grass-bush / tree-log / no ground truth | 49.0% / 0.1% / 26.7% | 46.4% / 0.1% / 26.4% | 46.1% / 0.0% / 26.3% | 46.0% / 0.1% / 26.5% | 52.8% / 0.0% / 27.9% | 50.2% / 0.1% / 26.8% | 50.3% / 0.1% / 26.8% | 50.0% / 0.1% / 26.7% | 51.8% / 0.1% / 26.8% |
| closest hard obstacle (mean) | 2.72 m | 2.82 m | 2.85 m | 2.91 m | 2.93 m | 2.78 m | 2.80 m | 2.79 m | 2.83 m |
| near misses (hard obstacle < 0.75 m) | 8.1% | 5.2% | 3.7% | 4.3% | 10.1% | 6.7% | 6.4% | 6.4% | 9.1% |
| ground slope (cm per m) | 5.2 | 5.0 | 4.8 | 5.0 | 5.3 | 5.5 | 5.5 | 5.5 | 5.8 |
| distance from the driven path at 5 / 10 / 15 m | 0.66 m / 1.21 m / 2.85 m | 0.56 m / 0.95 m / 1.70 m | 0.73 m / 1.16 m / 1.89 m | 0.47 m / 0.63 m / 1.14 m | 0.00 m / 0.00 m / 0.02 m | 0.58 m / 1.10 m / 2.94 m | 0.59 m / 1.07 m / 2.85 m | 0.59 m / 1.08 m / 2.88 m | 0.55 m / 1.12 m / 3.50 m |
| distance at 15 m: mean [95% interval] | 2.85 m [2.72, 2.97] | 1.70 m [1.53, 1.88] | 1.89 m [1.72, 2.07] | 1.14 m [1.02, 1.25] | 0.02 m [0.02, 0.02] | 2.94 m [2.71, 3.17] | 2.85 m [2.63, 3.08] | 2.88 m [2.65, 3.11] | 3.50 m [3.26, 3.74] |
| distance at 15 m: median / 90th percentile | 2.62 m / 4.00 m | 1.30 m / 3.74 m | 1.46 m / 3.88 m | 0.87 m / 2.16 m | 0.02 m / 0.03 m | 2.55 m / 5.73 m | 2.48 m / 5.33 m | 2.47 m / 5.79 m | 3.24 m / 6.60 m |
| mean distance over 5 / 10 / 15 m (ADE) [95% interval] | 1.57 m [1.50, 1.64] | 1.07 m [0.98, 1.16] | 1.26 m [1.17, 1.35] | 0.75 m [0.69, 0.81] | 0.01 m [0.01, 0.01] | 1.54 m [1.44, 1.64] | 1.50 m [1.40, 1.61] | 1.52 m [1.41, 1.62] | 1.73 m [1.61, 1.84] |

### Path choice, Mode B (keep going + the driver's direction word)

| metric | random | straightest | cheapest | toward the word | driver (oracle) | driven path (reference) | test_Qwen3-VL-4B-Instruct | test_Qwen3-VL-4B-Instruct_nohist | test_Qwen3-VL-4B-Instruct_wronghist | test_Qwen3-VL-8B-Instruct |
|---|---|---|---|---|---|---|---|---|---|---|
| frames with a choice | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 |
| choice matches the driver [95% interval] | 43% [38, 48] | 67% [62, 72] | 48% [43, 54] | 67% [62, 72] | 89% [86, 92] | 100% [99, 100] | 48% [43, 54] | 47% [42, 53] | 48% [43, 53] | 47% [42, 53] |
| ... in frames with a real choice | 43% | 67% | 48% | 67% | 90% | 100% | 48% | 47% | 48% | 47% |
| ... in straight frames / turn frames (driver's word) | 46% [40, 53] / 33% [24, 44] | 75% [69, 80] / 43% [33, 54] | 51% [44, 57] / 42% [32, 53] | 75% [69, 80] / 45% [34, 55] | 95% [92, 97] / 72% [62, 81] | 100% [98, 100] / 100% [96, 100] | 51% [44, 57] / 41% [31, 52] | 50% [44, 56] / 40% [30, 51] | 51% [45, 57] / 39% [29, 49] | 51% [45, 58] / 35% [26, 46] |
| choice is the driver's closest candidate | 31% | 58% | 43% | 62% | 100% | 100% | 34% | 35% | 34% | 30% |
| wrong choices: left / right of the driver's candidate | 57% / 43% | 61% / 39% | 74% / 26% | 73% / 27% | - / - | - / - | 33% / 67% | 37% / 63% | 31% / 69% | 24% / 76% |
| chosen position (2+ candidates): leftmost / middle / rightmost | 31% / 39% / 31% | 10% / 83% / 7% | 35% / 52% / 13% | 26% / 64% / 10% | 21% / 63% / 16% | - / - / - | 15% / 44% / 40% | 21% / 39% / 40% | 14% / 43% / 42% | 13% / 32% / 55% |
| chosen path safe to tick 1 / to the end | 100% / 99% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 99% | 100% / 99% | 100% / 100% |
| chose a puddle when a dry path existed | 46% | 62% | 77% | 62% | 54% | - | 54% | 46% | 46% | 31% |
| water / puddle / mud on the path (share of length) | 0.0% / 1.3% / 22.5% | 0.0% / 1.3% / 25.4% | 0.0% / 1.8% / 25.4% | 0.0% / 1.3% / 25.5% | 0.0% / 1.3% / 25.6% | 0.0% / 0.9% / 18.1% | 0.0% / 1.4% / 21.6% | 0.0% / 1.2% / 21.2% | 0.0% / 1.2% / 21.5% | 0.0% / 1.0% / 21.3% |
| grass-bush / tree-log / no ground truth | 49.0% / 0.1% / 26.7% | 46.4% / 0.1% / 26.4% | 46.1% / 0.0% / 26.3% | 46.0% / 0.1% / 26.7% | 46.0% / 0.1% / 26.5% | 52.8% / 0.0% / 27.9% | 49.9% / 0.0% / 26.7% | 50.4% / 0.0% / 26.8% | 50.2% / 0.1% / 26.7% | 50.6% / 0.1% / 26.8% |
| closest hard obstacle (mean) | 2.72 m | 2.82 m | 2.85 m | 2.80 m | 2.91 m | 2.93 m | 2.83 m | 2.80 m | 2.83 m | 2.85 m |
| near misses (hard obstacle < 0.75 m) | 8.1% | 5.2% | 3.7% | 6.1% | 4.3% | 10.1% | 6.4% | 7.3% | 7.0% | 7.0% |
| ground slope (cm per m) | 5.2 | 5.0 | 4.8 | 4.9 | 5.0 | 5.3 | 5.4 | 5.5 | 5.4 | 5.6 |
| distance from the driven path at 5 / 10 / 15 m | 0.66 m / 1.21 m / 2.85 m | 0.56 m / 0.95 m / 1.70 m | 0.73 m / 1.16 m / 1.89 m | 0.61 m / 0.92 m / 1.49 m | 0.47 m / 0.63 m / 1.14 m | 0.00 m / 0.00 m / 0.02 m | 0.61 m / 1.06 m / 2.72 m | 0.62 m / 1.06 m / 2.79 m | 0.60 m / 1.04 m / 2.75 m | 0.57 m / 1.04 m / 3.07 m |
| distance at 15 m: mean [95% interval] | 2.85 m [2.72, 2.97] | 1.70 m [1.53, 1.88] | 1.89 m [1.72, 2.07] | 1.49 m [1.36, 1.63] | 1.14 m [1.02, 1.25] | 0.02 m [0.02, 0.02] | 2.72 m [2.50, 2.94] | 2.79 m [2.57, 3.01] | 2.75 m [2.54, 2.97] | 3.07 m [2.86, 3.29] |
| distance at 15 m: median / 90th percentile | 2.62 m / 4.00 m | 1.30 m / 3.74 m | 1.46 m / 3.88 m | 1.31 m / 2.93 m | 0.87 m / 2.16 m | 0.02 m / 0.03 m | 2.36 m / 5.29 m | 2.39 m / 5.40 m | 2.39 m / 5.30 m | 2.85 m / 5.59 m |
| mean distance over 5 / 10 / 15 m (ADE) [95% interval] | 1.57 m [1.50, 1.64] | 1.07 m [0.98, 1.16] | 1.26 m [1.17, 1.35] | 1.01 m [0.93, 1.08] | 0.75 m [0.69, 0.81] | 0.01 m [0.01, 0.01] | 1.46 m [1.36, 1.57] | 1.49 m [1.39, 1.59] | 1.46 m [1.36, 1.56] | 1.56 m [1.46, 1.66] |

### Safe distance (ticks)

| metric | always "all" | test_Qwen3-VL-4B-Instruct A | test_Qwen3-VL-4B-Instruct B | test_Qwen3-VL-4B-Instruct_nohist A | test_Qwen3-VL-4B-Instruct_nohist B | test_Qwen3-VL-4B-Instruct_wronghist A | test_Qwen3-VL-4B-Instruct_wronghist B | test_Qwen3-VL-8B-Instruct A | test_Qwen3-VL-8B-Instruct B |
|---|---|---|---|---|---|---|---|---|---|
| ticks answered | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 |
| exact / within one tick | 99% / 100% | 1% / 80% | 2% / 67% | 2% / 75% | 1% / 66% | 1% / 78% | 2% / 63% | 40% / 51% | 50% / 61% |
| mean tick error (ticks) | 0.006 | 1.191 | 1.307 | 1.230 | 1.324 | 1.207 | 1.360 | 1.089 | 0.898 |
| too optimistic / too cautious | 1% / 0% | 0% / 99% | 0% / 98% | 0% / 98% | 0% / 99% | 0% / 99% | 0% / 98% | 0% / 60% | 0% / 50% |
| unsafe stretch called safe past it | 6/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| hazard flags ("not safe to the end"): labelled / flagged / both | 7 / 0 / 0 | 7 / 1115 / 7 | 7 / 1107 / 7 | 7 / 1106 / 7 | 7 / 1110 / 7 | 7 / 1115 / 7 | 7 / 1106 / 6 | 7 / 678 / 6 | 7 / 567 / 6 |
| hazard flags: precision / recall / F1 | - / 0.00 / - | 0.01 / 1.00 / 0.01 | 0.01 / 1.00 / 0.01 | 0.01 / 1.00 / 0.01 | 0.01 / 1.00 / 0.01 | 0.01 / 1.00 / 0.01 | 0.01 / 0.86 / 0.01 | 0.01 / 0.86 / 0.02 | 0.01 / 0.86 / 0.02 |
| label "none": answered exactly | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) |
| label "tick 1": answered exactly | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) |
| label "tick 2": answered exactly | 0/7 | 7/7 | 5/7 | 4/7 | 3/7 | 7/7 | 2/7 | 1/7 | 1/7 |
| label "all": answered exactly | 1116/1116 | 8/1116 | 16/1116 | 17/1116 | 13/1116 | 8/1116 | 16/1116 | 444/1116 | 555/1116 |

### Continue / stop

| metric | always continue | test_Qwen3-VL-4B-Instruct A | test_Qwen3-VL-4B-Instruct B | test_Qwen3-VL-4B-Instruct_nohist A | test_Qwen3-VL-4B-Instruct_nohist B | test_Qwen3-VL-4B-Instruct_wronghist A | test_Qwen3-VL-4B-Instruct_wronghist B | test_Qwen3-VL-8B-Instruct A | test_Qwen3-VL-8B-Instruct B |
|---|---|---|---|---|---|---|---|---|---|
| action correct | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| stop frames (label) / said stop / both | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| stop recall / precision | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - |
| stop F1 | - | - | - | - | - | - | - | - | - |

### Paired comparison, Mode A (driver match on the same frames)

Only the frames where the two disagree decide; a p-value below 0.05 means the difference is unlikely to be luck.

| metric | test_Qwen3-VL-4B-Instruct vs straightest | test_Qwen3-VL-4B-Instruct vs cheapest | test_Qwen3-VL-4B-Instruct_nohist vs straightest | test_Qwen3-VL-4B-Instruct_nohist vs cheapest | test_Qwen3-VL-4B-Instruct_wronghist vs straightest | test_Qwen3-VL-4B-Instruct_wronghist vs cheapest | test_Qwen3-VL-8B-Instruct vs straightest | test_Qwen3-VL-8B-Instruct vs cheapest |
|---|---|---|---|---|---|---|---|---|
| frames | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 |
| both right / both wrong | 122 / 89 | 89 / 116 | 124 / 87 | 90 / 113 | 125 / 87 | 89 / 111 | 107 / 79 | 80 / 112 |
| only the first right | 20 | 53 | 22 | 56 | 22 | 58 | 30 | 57 |
| only the second right | 97 | 70 | 95 | 69 | 94 | 70 | 112 | 79 |
| p-value (exact McNemar) | 0.000 | 0.149 | 0.000 | 0.283 | 0.000 | 0.331 | 0.000 | 0.071 |

### Paired comparison, Mode B (driver match on the same frames)

Only the frames where the two disagree decide; a p-value below 0.05 means the difference is unlikely to be luck.

| metric | test_Qwen3-VL-4B-Instruct vs straightest | test_Qwen3-VL-4B-Instruct vs cheapest | test_Qwen3-VL-4B-Instruct_nohist vs straightest | test_Qwen3-VL-4B-Instruct_nohist vs cheapest | test_Qwen3-VL-4B-Instruct_wronghist vs straightest | test_Qwen3-VL-4B-Instruct_wronghist vs cheapest | test_Qwen3-VL-8B-Instruct vs straightest | test_Qwen3-VL-8B-Instruct vs cheapest |
|---|---|---|---|---|---|---|---|---|
| frames | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 |
| both right / both wrong | 130 / 81 | 100 / 111 | 123 / 77 | 99 / 113 | 130 / 82 | 99 / 111 | 123 / 77 | 94 / 108 |
| only the first right | 28 | 58 | 32 | 56 | 27 | 58 | 32 | 61 |
| only the second right | 89 | 59 | 96 | 60 | 89 | 60 | 96 | 65 |
| p-value (exact McNemar) | 0.000 | 1.000 | 0.000 | 0.781 | 0.000 | 0.927 | 0.000 | 0.789 |

### Speed (as measured on the machine that ran the selector)

| metric | test_Qwen3-VL-4B-Instruct A | test_Qwen3-VL-4B-Instruct B | test_Qwen3-VL-4B-Instruct_nohist A | test_Qwen3-VL-4B-Instruct_nohist B | test_Qwen3-VL-4B-Instruct_wronghist A | test_Qwen3-VL-4B-Instruct_wronghist B | test_Qwen3-VL-8B-Instruct A | test_Qwen3-VL-8B-Instruct B |
|---|---|---|---|---|---|---|---|---|
| answers timed | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 |
| seconds per answer: mean / median / 95th percentile | 1.33 / 1.28 / 1.46 | 1.33 / 1.26 / 1.46 | 1.32 / 1.28 / 1.42 | 1.29 / 1.26 / 1.43 | 1.48 / 1.43 / 1.57 | 1.46 / 1.38 / 1.65 | 1.01 / 0.96 / 1.11 | 0.97 / 0.91 / 1.10 |
| answer length (tokens, mean) | 86 | 86 | 86 | 86 | 86 | 86 | 64 | 61 |

### Does the choice follow the word? ("bear left" vs "bear right", Mode B)

| metric |  |
|---|---|
| frames (2+ candidates, both answered) | 326 |
| choice further left with "bear left" (followed) | 89% |
| same candidate both times (ignored) | 11% |
| choice further left with "bear right" (reversed) | 0% |
| leftmost with "bear left" / rightmost with "bear right" | 48% / 97% |

### Does a wrong history pull the choice? (Mode A: test_Qwen3-VL-4B-Instruct vs test_Qwen3-VL-4B-Instruct_wronghist)

| metric |  |
|---|---|
| frames (2+ candidates, both answered) | 325 |
| moved toward the made-up turn | 2% |
| moved away from it | 2% |
| same candidate | 95% |

### Does a wrong history pull the choice? (Mode B: test_Qwen3-VL-4B-Instruct vs test_Qwen3-VL-4B-Instruct_wronghist)

| metric |  |
|---|---|
| frames (2+ candidates, both answered) | 325 |
| moved toward the made-up turn | 3% |
| moved away from it | 5% |
| same candidate | 92% |

