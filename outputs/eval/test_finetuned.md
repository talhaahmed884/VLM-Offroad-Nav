## Evaluation: test, 328 frames with candidates

Frames without any candidate (the vehicle stops without asking): 3. Baselines and runs are scored on the same frames.

[a, b] = 95% confidence interval (Wilson for shares, normal approximation for means): with this many frames the true value is probably in that range. ADE = mean distance from the driven path over 5 / 10 / 15 m. For the random baseline the distances are averages over all candidates of a frame.

### Path choice, Mode A (keep going)

| metric | random | straightest | cheapest | driver (oracle) | driven path (reference) | test_ft-nomatch-skip-epoch4 | test_ft-nomatch-keep-epoch5 | test_ft-nomatch-skip-epoch4_nohist | test_ft-nomatch-keep-epoch5_nohist | test_Qwen3-VL-4B-Instruct_noreason | test_Qwen3-VL-4B-Instruct | test_Qwen3-VL-8B-Instruct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frames with a choice | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 |
| choice matches the driver [95% interval] | 43% [38, 48] | 67% [62, 72] | 48% [43, 54] | 89% [86, 92] | 100% [99, 100] | 66% [61, 71] | 59% [53, 64] | 67% [62, 72] | 63% [57, 68] | 44% [38, 49] | 43% [38, 49] | 42% [37, 47] |
| ... in frames with a real choice | 43% | 67% | 48% | 90% | 100% | 67% | 59% | 67% | 63% | 44% | 43% | 42% |
| ... in straight frames / turn frames (driver's word) | 46% [40, 53] / 33% [24, 44] | 75% [69, 80] / 43% [33, 54] | 51% [44, 57] / 42% [32, 53] | 95% [92, 97] / 72% [62, 81] | 100% [98, 100] / 100% [96, 100] | 70% [64, 75] / 57% [46, 67] | 62% [56, 68] / 49% [39, 60] | 70% [64, 76] / 58% [47, 68] | 67% [61, 73] / 51% [40, 61] | 52% [46, 58] / 18% [11, 28] | 52% [46, 58] / 18% [11, 28] | 49% [43, 56] / 19% [12, 29] |
| choice is the driver's closest candidate | 31% | 58% | 43% | 100% | 100% | 61% | 51% | 59% | 54% | 31% | 31% | 23% |
| wrong choices: left / right of the driver's candidate | 57% / 43% | 61% / 39% | 74% / 26% | - / - | - / - | 63% / 37% | 64% / 36% | 64% / 36% | 63% / 37% | 38% / 62% | 38% / 62% | 19% / 81% |
| chosen position (2+ candidates): leftmost / middle / rightmost | 31% / 39% / 31% | 10% / 83% / 7% | 35% / 52% / 13% | 21% / 63% / 16% | - / - / - | 29% / 54% / 18% | 36% / 45% / 20% | 30% / 55% / 15% | 32% / 50% / 18% | 16% / 44% / 39% | 17% / 44% / 39% | 10% / 22% / 68% |
| chosen path safe to tick 1 / to the end | 100% / 99% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 99% | 100% / 99% | 100% / 100% |
| chose a puddle when a dry path existed | 46% | 62% | 77% | 54% | - | 62% | 62% | 69% | 77% | 46% | 54% | 31% |
| water / puddle / mud on the path (share of length) | 0.0% / 1.3% / 22.5% | 0.0% / 1.3% / 25.4% | 0.0% / 1.8% / 25.4% | 0.0% / 1.3% / 25.6% | 0.0% / 0.9% / 18.1% | 0.0% / 1.5% / 25.5% | 0.0% / 1.6% / 25.3% | 0.0% / 1.4% / 25.9% | 0.0% / 1.6% / 26.0% | 0.0% / 1.2% / 21.3% | 0.0% / 1.3% / 21.3% | 0.0% / 1.0% / 20.0% |
| grass-bush / tree-log / no ground truth | 49.0% / 0.1% / 26.7% | 46.4% / 0.1% / 26.4% | 46.1% / 0.0% / 26.3% | 46.0% / 0.1% / 26.5% | 52.8% / 0.0% / 27.9% | 46.2% / 0.1% / 26.4% | 46.3% / 0.1% / 26.4% | 46.1% / 0.0% / 26.3% | 45.7% / 0.1% / 26.3% | 50.4% / 0.1% / 26.7% | 50.2% / 0.1% / 26.8% | 51.8% / 0.1% / 26.8% |
| closest hard obstacle (mean) | 2.72 m | 2.82 m | 2.85 m | 2.91 m | 2.93 m | 2.90 m | 2.90 m | 2.88 m | 2.88 m | 2.80 m | 2.78 m | 2.83 m |
| near misses (hard obstacle < 0.75 m) | 8.1% | 5.2% | 3.7% | 4.3% | 10.1% | 4.9% | 4.9% | 4.9% | 4.9% | 6.1% | 6.7% | 9.1% |
| ground slope (cm per m) | 5.2 | 5.0 | 4.8 | 5.0 | 5.3 | 4.9 | 4.9 | 4.8 | 4.8 | 5.5 | 5.5 | 5.8 |
| distance from the driven path at 5 / 10 / 15 m | 0.66 m / 1.21 m / 2.85 m | 0.56 m / 0.95 m / 1.70 m | 0.73 m / 1.16 m / 1.89 m | 0.47 m / 0.63 m / 1.14 m | 0.00 m / 0.00 m / 0.02 m | 0.60 m / 0.90 m / 1.43 m | 0.63 m / 0.93 m / 1.63 m | 0.62 m / 0.91 m / 1.45 m | 0.63 m / 0.95 m / 1.61 m | 0.59 m / 1.10 m / 2.89 m | 0.58 m / 1.10 m / 2.94 m | 0.55 m / 1.12 m / 3.50 m |
| distance at 15 m: mean [95% interval] | 2.85 m [2.72, 2.97] | 1.70 m [1.53, 1.88] | 1.89 m [1.72, 2.07] | 1.14 m [1.02, 1.25] | 0.02 m [0.02, 0.02] | 1.43 m [1.28, 1.58] | 1.63 m [1.50, 1.76] | 1.45 m [1.31, 1.59] | 1.61 m [1.46, 1.77] | 2.89 m [2.66, 3.12] | 2.94 m [2.71, 3.17] | 3.50 m [3.26, 3.74] |
| distance at 15 m: median / 90th percentile | 2.62 m / 4.00 m | 1.30 m / 3.74 m | 1.46 m / 3.88 m | 0.87 m / 2.16 m | 0.02 m / 0.03 m | 1.17 m / 2.66 m | 1.35 m / 3.00 m | 1.24 m / 2.61 m | 1.31 m / 2.94 m | 2.55 m / 5.64 m | 2.55 m / 5.73 m | 3.24 m / 6.60 m |
| mean distance over 5 / 10 / 15 m (ADE) [95% interval] | 1.57 m [1.50, 1.64] | 1.07 m [0.98, 1.16] | 1.26 m [1.17, 1.35] | 0.75 m [0.69, 0.81] | 0.01 m [0.01, 0.01] | 0.98 m [0.90, 1.06] | 1.06 m [0.99, 1.13] | 0.99 m [0.92, 1.07] | 1.07 m [0.99, 1.15] | 1.53 m [1.43, 1.63] | 1.54 m [1.44, 1.64] | 1.73 m [1.61, 1.84] |

### Path choice, Mode B (keep going + the driver's direction word)

| metric | random | straightest | cheapest | toward the word | driver (oracle) | driven path (reference) | test_ft-nomatch-skip-epoch4 | test_ft-nomatch-keep-epoch5 | test_ft-nomatch-skip-epoch4_nohist | test_ft-nomatch-keep-epoch5_nohist | test_Qwen3-VL-4B-Instruct_noreason | test_Qwen3-VL-4B-Instruct | test_Qwen3-VL-8B-Instruct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frames with a choice | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 |
| choice matches the driver [95% interval] | 43% [38, 48] | 67% [62, 72] | 48% [43, 54] | 67% [62, 72] | 89% [86, 92] | 100% [99, 100] | 66% [61, 71] | 60% [54, 65] | 68% [63, 73] | 64% [58, 69] | 48% [43, 54] | 48% [43, 54] | 47% [42, 53] |
| ... in frames with a real choice | 43% | 67% | 48% | 67% | 90% | 100% | 66% | 60% | 68% | 64% | 48% | 48% | 47% |
| ... in straight frames / turn frames (driver's word) | 46% [40, 53] / 33% [24, 44] | 75% [69, 80] / 43% [33, 54] | 51% [44, 57] / 42% [32, 53] | 75% [69, 80] / 45% [34, 55] | 95% [92, 97] / 72% [62, 81] | 100% [98, 100] / 100% [96, 100] | 70% [64, 76] / 54% [44, 65] | 63% [57, 69] / 51% [40, 61] | 72% [66, 77] / 58% [47, 68] | 68% [62, 74] / 51% [40, 61] | 51% [44, 57] / 42% [32, 53] | 51% [44, 57] / 41% [31, 52] | 51% [45, 58] / 35% [26, 46] |
| choice is the driver's closest candidate | 31% | 58% | 43% | 62% | 100% | 100% | 61% | 52% | 61% | 56% | 34% | 34% | 30% |
| wrong choices: left / right of the driver's candidate | 57% / 43% | 61% / 39% | 74% / 26% | 73% / 27% | - / - | - / - | 63% / 37% | 63% / 37% | 62% / 38% | 62% / 38% | 34% / 66% | 33% / 67% | 24% / 76% |
| chosen position (2+ candidates): leftmost / middle / rightmost | 31% / 39% / 31% | 10% / 83% / 7% | 35% / 52% / 13% | 26% / 64% / 10% | 21% / 63% / 16% | - / - / - | 29% / 53% / 18% | 34% / 46% / 20% | 29% / 55% / 16% | 31% / 51% / 17% | 16% / 44% / 41% | 15% / 44% / 40% | 13% / 32% / 55% |
| chosen path safe to tick 1 / to the end | 100% / 99% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 99% | 100% / 100% | 100% / 100% |
| chose a puddle when a dry path existed | 46% | 62% | 77% | 62% | 54% | - | 54% | 62% | 69% | 77% | 46% | 54% | 31% |
| water / puddle / mud on the path (share of length) | 0.0% / 1.3% / 22.5% | 0.0% / 1.3% / 25.4% | 0.0% / 1.8% / 25.4% | 0.0% / 1.3% / 25.5% | 0.0% / 1.3% / 25.6% | 0.0% / 0.9% / 18.1% | 0.0% / 1.4% / 25.3% | 0.0% / 1.6% / 25.5% | 0.0% / 1.4% / 25.9% | 0.0% / 1.6% / 26.0% | 0.0% / 1.2% / 21.3% | 0.0% / 1.4% / 21.6% | 0.0% / 1.0% / 21.3% |
| grass-bush / tree-log / no ground truth | 49.0% / 0.1% / 26.7% | 46.4% / 0.1% / 26.4% | 46.1% / 0.0% / 26.3% | 46.0% / 0.1% / 26.7% | 46.0% / 0.1% / 26.5% | 52.8% / 0.0% / 27.9% | 46.6% / 0.0% / 26.4% | 46.2% / 0.1% / 26.4% | 46.1% / 0.0% / 26.3% | 45.7% / 0.1% / 26.3% | 50.4% / 0.0% / 26.8% | 49.9% / 0.0% / 26.7% | 50.6% / 0.1% / 26.8% |
| closest hard obstacle (mean) | 2.72 m | 2.82 m | 2.85 m | 2.80 m | 2.91 m | 2.93 m | 2.90 m | 2.89 m | 2.90 m | 2.88 m | 2.81 m | 2.83 m | 2.85 m |
| near misses (hard obstacle < 0.75 m) | 8.1% | 5.2% | 3.7% | 6.1% | 4.3% | 10.1% | 5.2% | 4.9% | 4.6% | 4.6% | 6.4% | 6.4% | 7.0% |
| ground slope (cm per m) | 5.2 | 5.0 | 4.8 | 4.9 | 5.0 | 5.3 | 4.9 | 4.9 | 4.8 | 4.9 | 5.4 | 5.4 | 5.6 |
| distance from the driven path at 5 / 10 / 15 m | 0.66 m / 1.21 m / 2.85 m | 0.56 m / 0.95 m / 1.70 m | 0.73 m / 1.16 m / 1.89 m | 0.61 m / 0.92 m / 1.49 m | 0.47 m / 0.63 m / 1.14 m | 0.00 m / 0.00 m / 0.02 m | 0.61 m / 0.90 m / 1.44 m | 0.62 m / 0.92 m / 1.59 m | 0.62 m / 0.90 m / 1.42 m | 0.62 m / 0.92 m / 1.54 m | 0.61 m / 1.05 m / 2.79 m | 0.61 m / 1.06 m / 2.72 m | 0.57 m / 1.04 m / 3.07 m |
| distance at 15 m: mean [95% interval] | 2.85 m [2.72, 2.97] | 1.70 m [1.53, 1.88] | 1.89 m [1.72, 2.07] | 1.49 m [1.36, 1.63] | 1.14 m [1.02, 1.25] | 0.02 m [0.02, 0.02] | 1.44 m [1.29, 1.59] | 1.59 m [1.46, 1.73] | 1.42 m [1.28, 1.56] | 1.54 m [1.40, 1.68] | 2.79 m [2.57, 3.01] | 2.72 m [2.50, 2.94] | 3.07 m [2.86, 3.29] |
| distance at 15 m: median / 90th percentile | 2.62 m / 4.00 m | 1.30 m / 3.74 m | 1.46 m / 3.88 m | 1.31 m / 2.93 m | 0.87 m / 2.16 m | 0.02 m / 0.03 m | 1.16 m / 2.66 m | 1.34 m / 2.97 m | 1.17 m / 2.61 m | 1.30 m / 2.81 m | 2.39 m / 5.29 m | 2.36 m / 5.29 m | 2.85 m / 5.59 m |
| mean distance over 5 / 10 / 15 m (ADE) [95% interval] | 1.57 m [1.50, 1.64] | 1.07 m [0.98, 1.16] | 1.26 m [1.17, 1.35] | 1.01 m [0.93, 1.08] | 0.75 m [0.69, 0.81] | 0.01 m [0.01, 0.01] | 0.98 m [0.90, 1.07] | 1.05 m [0.98, 1.11] | 0.98 m [0.90, 1.05] | 1.03 m [0.95, 1.10] | 1.48 m [1.38, 1.58] | 1.46 m [1.36, 1.57] | 1.56 m [1.46, 1.66] |

### Safe distance (ticks)

| metric | always "all" | test_ft-nomatch-skip-epoch4 A | test_ft-nomatch-skip-epoch4 B | test_ft-nomatch-keep-epoch5 A | test_ft-nomatch-keep-epoch5 B | test_ft-nomatch-skip-epoch4_nohist A | test_ft-nomatch-skip-epoch4_nohist B | test_ft-nomatch-keep-epoch5_nohist A | test_ft-nomatch-keep-epoch5_nohist B | test_Qwen3-VL-4B-Instruct_noreason A | test_Qwen3-VL-4B-Instruct_noreason B | test_Qwen3-VL-4B-Instruct A | test_Qwen3-VL-4B-Instruct B | test_Qwen3-VL-8B-Instruct A | test_Qwen3-VL-8B-Instruct B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ticks answered | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 |
| exact / within one tick | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% | 1% / 79% | 1% / 62% | 1% / 80% | 2% / 67% | 40% / 51% | 50% / 61% |
| mean tick error (ticks) | 0.006 | 0.006 | 0.006 | 0.007 | 0.005 | 0.006 | 0.006 | 0.006 | 0.006 | 1.198 | 1.365 | 1.191 | 1.307 | 1.089 | 0.898 |
| too optimistic / too cautious | 1% / 0% | 1% / 0% | 1% / 0% | 0% / 0% | 0% / 0% | 1% / 0% | 1% / 0% | 0% / 0% | 0% / 0% | 0% / 99% | 0% / 99% | 0% / 99% | 0% / 98% | 0% / 60% | 0% / 50% |
| unsafe stretch called safe past it | 6/6 | 6/6 | 6/6 | 3/6 | 2/6 | 6/6 | 6/6 | 3/6 | 3/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| hazard flags ("not safe to the end"): labelled / flagged / both | 7 / 0 / 0 | 7 / 0 / 0 | 7 / 0 / 0 | 7 / 7 / 3 | 7 / 7 / 4 | 7 / 0 / 0 | 7 / 0 / 0 | 7 / 6 / 3 | 7 / 6 / 3 | 7 / 1118 / 7 | 7 / 1110 / 7 | 7 / 1115 / 7 | 7 / 1107 / 7 | 7 / 678 / 6 | 7 / 567 / 6 |
| hazard flags: precision / recall / F1 | - / 0.00 / - | - / 0.00 / - | - / 0.00 / - | 0.43 / 0.43 / 0.43 | 0.57 / 0.57 / 0.57 | - / 0.00 / - | - / 0.00 / - | 0.50 / 0.43 / 0.46 | 0.50 / 0.43 / 0.46 | 0.01 / 1.00 / 0.01 | 0.01 / 1.00 / 0.01 | 0.01 / 1.00 / 0.01 | 0.01 / 1.00 / 0.01 | 0.01 / 0.86 / 0.02 | 0.01 / 0.86 / 0.02 |
| label "none": answered exactly | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) |
| label "tick 1": answered exactly | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) |
| label "tick 2": answered exactly | 0/7 | 0/7 | 0/7 | 3/7 | 4/7 | 0/7 | 0/7 | 3/7 | 3/7 | 7/7 | 2/7 | 7/7 | 5/7 | 1/7 | 1/7 |
| label "all": answered exactly | 1116/1116 | 1116/1116 | 1116/1116 | 1112/1116 | 1113/1116 | 1116/1116 | 1116/1116 | 1113/1116 | 1113/1116 | 5/1116 | 13/1116 | 8/1116 | 16/1116 | 444/1116 | 555/1116 |

### Continue / stop

| metric | always continue | test_ft-nomatch-skip-epoch4 A | test_ft-nomatch-skip-epoch4 B | test_ft-nomatch-keep-epoch5 A | test_ft-nomatch-keep-epoch5 B | test_ft-nomatch-skip-epoch4_nohist A | test_ft-nomatch-skip-epoch4_nohist B | test_ft-nomatch-keep-epoch5_nohist A | test_ft-nomatch-keep-epoch5_nohist B | test_Qwen3-VL-4B-Instruct_noreason A | test_Qwen3-VL-4B-Instruct_noreason B | test_Qwen3-VL-4B-Instruct A | test_Qwen3-VL-4B-Instruct B | test_Qwen3-VL-8B-Instruct A | test_Qwen3-VL-8B-Instruct B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| action correct | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| stop frames (label) / said stop / both | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| stop recall / precision | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - |
| stop F1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

### Paired comparison, Mode A (driver match on the same frames)

Only the frames where the two disagree decide; a p-value below 0.05 means the difference is unlikely to be luck.

| metric | test_ft-nomatch-skip-epoch4 vs straightest | test_ft-nomatch-skip-epoch4 vs cheapest | test_ft-nomatch-keep-epoch5 vs straightest | test_ft-nomatch-keep-epoch5 vs cheapest | test_ft-nomatch-skip-epoch4_nohist vs straightest | test_ft-nomatch-skip-epoch4_nohist vs cheapest | test_ft-nomatch-keep-epoch5_nohist vs straightest | test_ft-nomatch-keep-epoch5_nohist vs cheapest | test_Qwen3-VL-4B-Instruct_noreason vs straightest | test_Qwen3-VL-4B-Instruct_noreason vs cheapest | test_Qwen3-VL-4B-Instruct vs straightest | test_Qwen3-VL-4B-Instruct vs cheapest | test_Qwen3-VL-8B-Instruct vs straightest | test_Qwen3-VL-8B-Instruct vs cheapest | test_ft-nomatch-skip-epoch4 vs test_Qwen3-VL-4B-Instruct_noreason | test_ft-nomatch-skip-epoch4 vs test_ft-nomatch-keep-epoch5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frames | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 |
| both right / both wrong | 168 / 59 | 137 / 88 | 148 / 64 | 124 / 100 | 170 / 59 | 138 / 87 | 157 / 60 | 132 / 95 | 123 / 89 | 88 / 114 | 122 / 89 | 89 / 116 | 107 / 79 | 80 / 112 | 108 / 75 | 187 / 104 |
| only the first right | 50 | 81 | 45 | 69 | 50 | 82 | 49 | 74 | 20 | 55 | 20 | 53 | 30 | 57 | 110 | 31 |
| only the second right | 51 | 22 | 71 | 35 | 49 | 21 | 62 | 27 | 96 | 71 | 97 | 70 | 112 | 79 | 35 | 6 |
| p-value (exact McNemar) | 1.000 | 0.000 | 0.020 | 0.001 | 1.000 | 0.000 | 0.255 | 0.000 | 0.000 | 0.181 | 0.000 | 0.149 | 0.000 | 0.071 | 0.000 | 0.000 |

### Paired comparison, Mode B (driver match on the same frames)

Only the frames where the two disagree decide; a p-value below 0.05 means the difference is unlikely to be luck.

| metric | test_ft-nomatch-skip-epoch4 vs straightest | test_ft-nomatch-skip-epoch4 vs cheapest | test_ft-nomatch-keep-epoch5 vs straightest | test_ft-nomatch-keep-epoch5 vs cheapest | test_ft-nomatch-skip-epoch4_nohist vs straightest | test_ft-nomatch-skip-epoch4_nohist vs cheapest | test_ft-nomatch-keep-epoch5_nohist vs straightest | test_ft-nomatch-keep-epoch5_nohist vs cheapest | test_Qwen3-VL-4B-Instruct_noreason vs straightest | test_Qwen3-VL-4B-Instruct_noreason vs cheapest | test_Qwen3-VL-4B-Instruct vs straightest | test_Qwen3-VL-4B-Instruct vs cheapest | test_Qwen3-VL-8B-Instruct vs straightest | test_Qwen3-VL-8B-Instruct vs cheapest | test_ft-nomatch-skip-epoch4 vs test_Qwen3-VL-4B-Instruct_noreason | test_ft-nomatch-skip-epoch4 vs test_ft-nomatch-keep-epoch5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frames | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 |
| both right / both wrong | 167 / 59 | 137 / 89 | 149 / 62 | 125 / 98 | 173 / 58 | 140 / 85 | 159 / 59 | 131 / 91 | 128 / 78 | 98 / 108 | 130 / 81 | 100 / 111 | 123 / 77 | 94 / 108 | 127 / 79 | 190 / 105 |
| only the first right | 50 | 80 | 47 | 71 | 51 | 84 | 50 | 78 | 31 | 61 | 28 | 58 | 32 | 61 | 90 | 27 |
| only the second right | 52 | 22 | 70 | 34 | 46 | 19 | 60 | 28 | 91 | 61 | 89 | 59 | 96 | 65 | 32 | 6 |
| p-value (exact McNemar) | 0.921 | 0.000 | 0.042 | 0.000 | 0.685 | 0.000 | 0.391 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.789 | 0.000 | 0.000 |

### Speed (as measured on the machine that ran the selector)

| metric | test_ft-nomatch-skip-epoch4 A | test_ft-nomatch-skip-epoch4 B | test_ft-nomatch-keep-epoch5 A | test_ft-nomatch-keep-epoch5 B | test_ft-nomatch-skip-epoch4_nohist A | test_ft-nomatch-skip-epoch4_nohist B | test_ft-nomatch-keep-epoch5_nohist A | test_ft-nomatch-keep-epoch5_nohist B | test_Qwen3-VL-4B-Instruct_noreason A | test_Qwen3-VL-4B-Instruct_noreason B | test_Qwen3-VL-4B-Instruct A | test_Qwen3-VL-4B-Instruct B | test_Qwen3-VL-8B-Instruct A | test_Qwen3-VL-8B-Instruct B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| answers timed | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 | 328 |
| seconds per answer: mean / median / 95th percentile | 1.15 / 1.18 / 1.24 | 1.17 / 1.18 / 1.24 | 1.13 / 1.16 / 1.23 | 1.16 / 1.16 / 1.23 | 1.15 / 1.18 / 1.24 | 1.14 / 1.18 / 1.23 | 1.14 / 1.18 / 1.24 | 1.14 / 1.18 / 1.23 | 0.88 / 0.88 / 0.92 | 0.89 / 0.88 / 0.93 | 1.33 / 1.28 / 1.46 | 1.33 / 1.26 / 1.46 | 1.01 / 0.96 / 1.11 | 0.97 / 0.91 / 1.10 |
| answer length (tokens, mean) | 39 | 39 | 39 | 39 | 39 | 39 | 39 | 39 | 57 | 57 | 86 | 86 | 64 | 61 |

