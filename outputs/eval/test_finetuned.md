## Evaluation: test, 328 frames with candidates

Frames without any candidate (the vehicle stops without asking): 3. Baselines and runs are scored on the same frames.

### Path choice, Mode A (keep going)

| metric | random | straightest | cheapest | driver (oracle) | driven path (reference) | test_ft-nomatch-skip-epoch4 | test_ft-nomatch-keep-epoch5 | test_ft-nomatch-skip-epoch4_nohist | test_ft-nomatch-keep-epoch5_nohist | test_Qwen3-VL-4B-Instruct_noreason | test_Qwen3-VL-4B-Instruct | test_Qwen3-VL-8B-Instruct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frames with a choice | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 |
| choice matches the driver | 43% | 67% | 48% | 89% | 100% | 66% | 59% | 67% | 63% | 44% | 43% | 42% |
| ... in frames with a real choice | 43% | 67% | 48% | 90% | 100% | 67% | 59% | 67% | 63% | 44% | 43% | 42% |
| ... in turn frames (driver's word not straight) | 33% | 43% | 42% | 72% | 100% | 57% | 49% | 58% | 51% | 18% | 18% | 19% |
| choice is the driver's closest candidate | 31% | 58% | 43% | 100% | 100% | 61% | 51% | 59% | 54% | 31% | 31% | 23% |
| chosen path safe to tick 1 / to the end | 100% / 99% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 99% | 100% / 99% | 100% / 100% |
| chose a puddle when a dry path existed | 46% | 62% | 77% | 54% | - | 62% | 62% | 69% | 77% | 46% | 54% | 31% |
| water / puddle / mud on the path (share of length) | 0.0% / 1.3% / 22.5% | 0.0% / 1.3% / 25.4% | 0.0% / 1.8% / 25.4% | 0.0% / 1.3% / 25.6% | 0.0% / 0.9% / 18.1% | 0.0% / 1.5% / 25.5% | 0.0% / 1.6% / 25.3% | 0.0% / 1.4% / 25.9% | 0.0% / 1.6% / 26.0% | 0.0% / 1.2% / 21.3% | 0.0% / 1.3% / 21.3% | 0.0% / 1.0% / 20.0% |
| grass-bush / tree-log / no ground truth | 49.0% / 0.1% / 26.7% | 46.4% / 0.1% / 26.4% | 46.1% / 0.0% / 26.3% | 46.0% / 0.1% / 26.5% | 52.8% / 0.0% / 27.9% | 46.2% / 0.1% / 26.4% | 46.3% / 0.1% / 26.4% | 46.1% / 0.0% / 26.3% | 45.7% / 0.1% / 26.3% | 50.4% / 0.1% / 26.7% | 50.2% / 0.1% / 26.8% | 51.8% / 0.1% / 26.8% |
| closest hard obstacle (mean) | 2.72 m | 2.82 m | 2.85 m | 2.91 m | 2.93 m | 2.90 m | 2.90 m | 2.88 m | 2.88 m | 2.80 m | 2.78 m | 2.83 m |
| near misses (hard obstacle < 0.75 m) | 8.1% | 5.2% | 3.7% | 4.3% | 10.1% | 4.9% | 4.9% | 4.9% | 4.9% | 6.1% | 6.7% | 9.1% |
| ground slope (cm per m) | 5.2 | 5.0 | 4.8 | 5.0 | 5.3 | 4.9 | 4.9 | 4.8 | 4.8 | 5.5 | 5.5 | 5.8 |
| distance from the driven path at 5 / 10 / 15 m | 0.66 m / 1.21 m / 2.85 m | 0.56 m / 0.95 m / 1.70 m | 0.73 m / 1.16 m / 1.89 m | 0.47 m / 0.63 m / 1.14 m | 0.00 m / 0.00 m / 0.02 m | 0.60 m / 0.90 m / 1.43 m | 0.63 m / 0.93 m / 1.63 m | 0.62 m / 0.91 m / 1.45 m | 0.63 m / 0.95 m / 1.61 m | 0.59 m / 1.10 m / 2.89 m | 0.58 m / 1.10 m / 2.94 m | 0.55 m / 1.12 m / 3.50 m |

### Path choice, Mode B (keep going + the driver's direction word)

| metric | random | straightest | cheapest | toward the word | driver (oracle) | driven path (reference) | test_ft-nomatch-skip-epoch4 | test_ft-nomatch-keep-epoch5 | test_ft-nomatch-skip-epoch4_nohist | test_ft-nomatch-keep-epoch5_nohist | test_Qwen3-VL-4B-Instruct_noreason | test_Qwen3-VL-4B-Instruct | test_Qwen3-VL-8B-Instruct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frames with a choice | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 |
| choice matches the driver | 43% | 67% | 48% | 67% | 89% | 100% | 66% | 60% | 68% | 64% | 48% | 48% | 47% |
| ... in frames with a real choice | 43% | 67% | 48% | 67% | 90% | 100% | 66% | 60% | 68% | 64% | 48% | 48% | 47% |
| ... in turn frames (driver's word not straight) | 33% | 43% | 42% | 45% | 72% | 100% | 54% | 51% | 58% | 51% | 42% | 41% | 35% |
| choice is the driver's closest candidate | 31% | 58% | 43% | 62% | 100% | 100% | 61% | 52% | 61% | 56% | 34% | 34% | 30% |
| chosen path safe to tick 1 / to the end | 100% / 99% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 99% | 100% / 100% | 100% / 100% |
| chose a puddle when a dry path existed | 46% | 62% | 77% | 62% | 54% | - | 54% | 62% | 69% | 77% | 46% | 54% | 31% |
| water / puddle / mud on the path (share of length) | 0.0% / 1.3% / 22.5% | 0.0% / 1.3% / 25.4% | 0.0% / 1.8% / 25.4% | 0.0% / 1.3% / 25.5% | 0.0% / 1.3% / 25.6% | 0.0% / 0.9% / 18.1% | 0.0% / 1.4% / 25.3% | 0.0% / 1.6% / 25.5% | 0.0% / 1.4% / 25.9% | 0.0% / 1.6% / 26.0% | 0.0% / 1.2% / 21.3% | 0.0% / 1.4% / 21.6% | 0.0% / 1.0% / 21.3% |
| grass-bush / tree-log / no ground truth | 49.0% / 0.1% / 26.7% | 46.4% / 0.1% / 26.4% | 46.1% / 0.0% / 26.3% | 46.0% / 0.1% / 26.7% | 46.0% / 0.1% / 26.5% | 52.8% / 0.0% / 27.9% | 46.6% / 0.0% / 26.4% | 46.2% / 0.1% / 26.4% | 46.1% / 0.0% / 26.3% | 45.7% / 0.1% / 26.3% | 50.4% / 0.0% / 26.8% | 49.9% / 0.0% / 26.7% | 50.6% / 0.1% / 26.8% |
| closest hard obstacle (mean) | 2.72 m | 2.82 m | 2.85 m | 2.80 m | 2.91 m | 2.93 m | 2.90 m | 2.89 m | 2.90 m | 2.88 m | 2.81 m | 2.83 m | 2.85 m |
| near misses (hard obstacle < 0.75 m) | 8.1% | 5.2% | 3.7% | 6.1% | 4.3% | 10.1% | 5.2% | 4.9% | 4.6% | 4.6% | 6.4% | 6.4% | 7.0% |
| ground slope (cm per m) | 5.2 | 5.0 | 4.8 | 4.9 | 5.0 | 5.3 | 4.9 | 4.9 | 4.8 | 4.9 | 5.4 | 5.4 | 5.6 |
| distance from the driven path at 5 / 10 / 15 m | 0.66 m / 1.21 m / 2.85 m | 0.56 m / 0.95 m / 1.70 m | 0.73 m / 1.16 m / 1.89 m | 0.61 m / 0.92 m / 1.49 m | 0.47 m / 0.63 m / 1.14 m | 0.00 m / 0.00 m / 0.02 m | 0.61 m / 0.90 m / 1.44 m | 0.62 m / 0.92 m / 1.59 m | 0.62 m / 0.90 m / 1.42 m | 0.62 m / 0.92 m / 1.54 m | 0.61 m / 1.05 m / 2.79 m | 0.61 m / 1.06 m / 2.72 m | 0.57 m / 1.04 m / 3.07 m |

### Safe distance (ticks)

| metric | always "all" | test_ft-nomatch-skip-epoch4 A | test_ft-nomatch-skip-epoch4 B | test_ft-nomatch-keep-epoch5 A | test_ft-nomatch-keep-epoch5 B | test_ft-nomatch-skip-epoch4_nohist A | test_ft-nomatch-skip-epoch4_nohist B | test_ft-nomatch-keep-epoch5_nohist A | test_ft-nomatch-keep-epoch5_nohist B | test_Qwen3-VL-4B-Instruct_noreason A | test_Qwen3-VL-4B-Instruct_noreason B | test_Qwen3-VL-4B-Instruct A | test_Qwen3-VL-4B-Instruct B | test_Qwen3-VL-8B-Instruct A | test_Qwen3-VL-8B-Instruct B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ticks answered | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 | 1123/1123 |
| exact / within one tick | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% | 99% / 100% | 1% / 79% | 1% / 62% | 1% / 80% | 2% / 67% | 40% / 51% | 50% / 61% |
| too optimistic / too cautious | 1% / 0% | 1% / 0% | 1% / 0% | 0% / 0% | 0% / 0% | 1% / 0% | 1% / 0% | 0% / 0% | 0% / 0% | 0% / 99% | 0% / 99% | 0% / 99% | 0% / 98% | 0% / 60% | 0% / 50% |
| unsafe stretch called safe past it | 6/6 | 6/6 | 6/6 | 3/6 | 2/6 | 6/6 | 6/6 | 3/6 | 3/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
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

