## Evaluation: test, 328 frames with candidates

Frames without any candidate (the vehicle stops without asking): 3. Baselines and runs are scored on the same frames.

### Path choice, Mode A (keep going)

| metric | random | straightest | cheapest | driver (oracle) | driven path (reference) | test_ft-nomatch-keep-epoch5 |
|---|---|---|---|---|---|---|
| frames with a choice | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 |
| choice matches the driver | 43% | 67% | 48% | 89% | 100% | 59% |
| ... in frames with a real choice | 43% | 67% | 48% | 90% | 100% | 59% |
| ... in turn frames (driver's word not straight) | 33% | 43% | 42% | 72% | 100% | 49% |
| choice is the driver's closest candidate | 31% | 58% | 43% | 100% | 100% | 51% |
| chosen path safe to tick 1 / to the end | 100% / 99% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| chose a puddle when a dry path existed | 46% | 62% | 77% | 54% | - | 62% |
| water / puddle / mud on the path (share of length) | 0.0% / 1.3% / 22.5% | 0.0% / 1.3% / 25.4% | 0.0% / 1.8% / 25.4% | 0.0% / 1.3% / 25.6% | 0.0% / 0.9% / 18.1% | 0.0% / 1.6% / 25.3% |
| grass-bush / tree-log / no ground truth | 49.0% / 0.1% / 26.7% | 46.4% / 0.1% / 26.4% | 46.1% / 0.0% / 26.3% | 46.0% / 0.1% / 26.5% | 52.8% / 0.0% / 27.9% | 46.3% / 0.1% / 26.4% |
| closest hard obstacle (mean) | 2.72 m | 2.82 m | 2.85 m | 2.91 m | 2.93 m | 2.90 m |
| near misses (hard obstacle < 0.75 m) | 8.1% | 5.2% | 3.7% | 4.3% | 10.1% | 4.9% |
| ground slope (cm per m) | 5.2 | 5.0 | 4.8 | 5.0 | 5.3 | 4.9 |
| distance from the driven path at 5 / 10 / 15 m | 0.66 m / 1.21 m / 2.85 m | 0.56 m / 0.95 m / 1.70 m | 0.73 m / 1.16 m / 1.89 m | 0.47 m / 0.63 m / 1.14 m | 0.00 m / 0.00 m / 0.02 m | 0.63 m / 0.93 m / 1.63 m |

### Path choice, Mode B (keep going + the driver's direction word)

| metric | random | straightest | cheapest | toward the word | driver (oracle) | driven path (reference) | test_ft-nomatch-keep-epoch5 |
|---|---|---|---|---|---|---|---|
| frames with a choice | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 | 328/328 |
| choice matches the driver | 43% | 67% | 48% | 67% | 89% | 100% | 60% |
| ... in frames with a real choice | 43% | 67% | 48% | 67% | 90% | 100% | 60% |
| ... in turn frames (driver's word not straight) | 33% | 43% | 42% | 45% | 72% | 100% | 51% |
| choice is the driver's closest candidate | 31% | 58% | 43% | 62% | 100% | 100% | 52% |
| chosen path safe to tick 1 / to the end | 100% / 99% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| chose a puddle when a dry path existed | 46% | 62% | 77% | 62% | 54% | - | 62% |
| water / puddle / mud on the path (share of length) | 0.0% / 1.3% / 22.5% | 0.0% / 1.3% / 25.4% | 0.0% / 1.8% / 25.4% | 0.0% / 1.3% / 25.5% | 0.0% / 1.3% / 25.6% | 0.0% / 0.9% / 18.1% | 0.0% / 1.6% / 25.5% |
| grass-bush / tree-log / no ground truth | 49.0% / 0.1% / 26.7% | 46.4% / 0.1% / 26.4% | 46.1% / 0.0% / 26.3% | 46.0% / 0.1% / 26.7% | 46.0% / 0.1% / 26.5% | 52.8% / 0.0% / 27.9% | 46.2% / 0.1% / 26.4% |
| closest hard obstacle (mean) | 2.72 m | 2.82 m | 2.85 m | 2.80 m | 2.91 m | 2.93 m | 2.89 m |
| near misses (hard obstacle < 0.75 m) | 8.1% | 5.2% | 3.7% | 6.1% | 4.3% | 10.1% | 4.9% |
| ground slope (cm per m) | 5.2 | 5.0 | 4.8 | 4.9 | 5.0 | 5.3 | 4.9 |
| distance from the driven path at 5 / 10 / 15 m | 0.66 m / 1.21 m / 2.85 m | 0.56 m / 0.95 m / 1.70 m | 0.73 m / 1.16 m / 1.89 m | 0.61 m / 0.92 m / 1.49 m | 0.47 m / 0.63 m / 1.14 m | 0.00 m / 0.00 m / 0.02 m | 0.62 m / 0.92 m / 1.59 m |

### Safe distance (ticks)

| metric | always "all" | test_ft-nomatch-keep-epoch5 A | test_ft-nomatch-keep-epoch5 B |
|---|---|---|---|
| ticks answered | 1123/1123 | 1123/1123 | 1123/1123 |
| exact / within one tick | 99% / 100% | 99% / 100% | 99% / 100% |
| too optimistic / too cautious | 1% / 0% | 0% / 0% | 0% / 0% |
| unsafe stretch called safe past it | 6/6 | 3/6 | 2/6 |
| label "none": answered exactly | - (none) | - (none) | - (none) |
| label "tick 1": answered exactly | - (none) | - (none) | - (none) |
| label "tick 2": answered exactly | 0/7 | 3/7 | 4/7 |
| label "all": answered exactly | 1116/1116 | 1112/1116 | 1113/1116 |

### Continue / stop

| metric | always continue | test_ft-nomatch-keep-epoch5 A | test_ft-nomatch-keep-epoch5 B |
|---|---|---|---|
| action correct | 100% | 100% | 100% |
| stop frames (label) / said stop / both | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| stop recall / precision | - / - | - / - | - / - |

### Does the choice follow the word? ("bear left" vs "bear right", Mode B)

| metric |  |
|---|---|
| frames (2+ candidates, both answered) | 326 |
| choice further left with "bear left" (followed) | 14% |
| same candidate both times (ignored) | 86% |
| choice further left with "bear right" (reversed) | 0% |
| leftmost with "bear left" / rightmost with "bear right" | 37% / 26% |

### Does a wrong history pull the choice? (Mode A: test_ft-nomatch-keep-epoch5 vs test_ft-nomatch-keep-epoch5_wronghist)

| metric |  |
|---|---|
| frames (2+ candidates, both answered) | 325 |
| moved toward the made-up turn | 3% |
| moved away from it | 0% |
| same candidate | 97% |

### Does a wrong history pull the choice? (Mode B: test_ft-nomatch-keep-epoch5 vs test_ft-nomatch-keep-epoch5_wronghist)

| metric |  |
|---|---|
| frames (2+ candidates, both answered) | 325 |
| moved toward the made-up turn | 2% |
| moved away from it | 0% |
| same candidate | 97% |

