## Candidate comparison: strict vs relaxed (2220 decision frames, every 1)

### train

| metric | strict | relaxed |
|---|---|---|
| frames | 1553 | 1553 |
| candidates per frame (mean) | 2.01 | 3.04 |
| frames with 0 / 1 / 2 / 3 / 4 | 4% / 28% / 38% / 23% / 7% | 2% / 5% / 20% / 36% / 38% |
| best candidate to driven path, median (m) | 0.60 | 0.67 |
| best candidate to driven path, 90th pct (m) | 1.19 | 1.56 |
| driver match (a candidate within 1 m) | 84% | 83% |
| matching candidates per frame (mean) | 1.23 | 1.38 |
| real choice (2+ safe to tick 1) | 68% | 94% |
| stop (none safe to tick 1) | 7% | 2% |
| candidates safe: none / tick 1 / tick 2 / all | 2% / 4% / 10% / 85% | 0% / 1% / 4% / 95% |
| driver-matching candidates safe 'all' | 86% | 97% |
| full length (>= 14 m) | 85% | 97% |
| unknown ground share (mean) | 0.29 | 0.27 |
| first unsafe class (top 3) | - | tree 83, person 19, water 7 |
| candidates through a puddle (caution) | 7% | 6% |
| driver's candidate through a puddle | 6% | 6% |
| ... of those, with a dry alternative | 22% | 45% |

### val

| metric | strict | relaxed |
|---|---|---|
| frames | 336 | 336 |
| candidates per frame (mean) | 2.57 | 3.29 |
| frames with 0 / 1 / 2 / 3 / 4 | 5% / 8% / 28% / 41% / 17% | 1% / 4% / 12% / 32% / 51% |
| best candidate to driven path, median (m) | 0.76 | 0.80 |
| best candidate to driven path, 90th pct (m) | 1.15 | 1.24 |
| driver match (a candidate within 1 m) | 76% | 77% |
| matching candidates per frame (mean) | 1.20 | 1.11 |
| real choice (2+ safe to tick 1) | 87% | 96% |
| stop (none safe to tick 1) | 6% | 1% |
| candidates safe: none / tick 1 / tick 2 / all | 0% / 1% / 10% / 88% | 0% / 0% / 2% / 98% |
| driver-matching candidates safe 'all' | 94% | 99% |
| full length (>= 14 m) | 88% | 100% |
| unknown ground share (mean) | 0.28 | 0.27 |
| first unsafe class (top 3) | - | barrier 5, log 5, rubble 5 |
| candidates through a puddle (caution) | 22% | 19% |
| driver's candidate through a puddle | 21% | 21% |
| ... of those, with a dry alternative | 23% | 42% |

### test

| metric | strict | relaxed |
|---|---|---|
| frames | 331 | 331 |
| candidates per frame (mean) | 2.70 | 3.39 |
| frames with 0 / 1 / 2 / 3 / 4 | 1% / 4% / 36% / 42% / 17% | 1% / 1% / 10% / 35% / 53% |
| best candidate to driven path, median (m) | 0.51 | 0.54 |
| best candidate to driven path, 90th pct (m) | 1.04 | 1.01 |
| driver match (a candidate within 1 m) | 88% | 89% |
| matching candidates per frame (mean) | 1.35 | 1.43 |
| real choice (2+ safe to tick 1) | 95% | 98% |
| stop (none safe to tick 1) | 1% | 1% |
| candidates safe: none / tick 1 / tick 2 / all | 0% / 1% / 11% / 88% | 0% / 0% / 1% / 99% |
| driver-matching candidates safe 'all' | 94% | 100% |
| full length (>= 14 m) | 88% | 100% |
| unknown ground share (mean) | 0.27 | 0.27 |
| first unsafe class (top 3) | - | rubble 5, log 1 |
| candidates through a puddle (caution) | 7% | 6% |
| driver's candidate through a puddle | 7% | 6% |
| ... of those, with a dry alternative | 8% | 33% |

### all

| metric | strict | relaxed |
|---|---|---|
| frames | 2220 | 2220 |
| candidates per frame (mean) | 2.20 | 3.13 |
| frames with 0 / 1 / 2 / 3 / 4 | 4% / 22% / 36% / 29% / 10% | 1% / 4% / 17% / 36% / 42% |
| best candidate to driven path, median (m) | 0.61 | 0.68 |
| best candidate to driven path, 90th pct (m) | 1.14 | 1.34 |
| driver match (a candidate within 1 m) | 83% | 83% |
| matching candidates per frame (mean) | 1.24 | 1.34 |
| real choice (2+ safe to tick 1) | 75% | 95% |
| stop (none safe to tick 1) | 6% | 2% |
| candidates safe: none / tick 1 / tick 2 / all | 1% / 3% / 10% / 86% | 0% / 1% / 3% / 96% |
| driver-matching candidates safe 'all' | 89% | 98% |
| full length (>= 14 m) | 86% | 98% |
| unknown ground share (mean) | 0.28 | 0.27 |
| first unsafe class (top 3) | - | tree 83, person 19, rubble 11 |
| candidates through a puddle (caution) | 9% | 8% |
| driver's candidate through a puddle | 9% | 9% |
| ... of those, with a dry alternative | 21% | 42% |

### the driver's own path, labelled like a candidate

| metric | train | val | test | all |
|---|---|---|---|---|
| frames | 1553 | 336 | 331 | 2220 |
| with a camera label image | 99% | 100% | 100% | 99% |
| driven path safe: none / tick 1 / tick 2 / all | 0% / 0% / 0% / 100% | 0% / 0% / 0% / 100% | 0% / 0% / 0% / 100% | 0% / 0% / 0% / 100% |
| first unsafe class on the driven path (top 3) | person 4 | - | - | person 4 |
| driven path through a puddle (caution) | 5% | 12% | 7% | 6% |
