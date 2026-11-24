## Evaluation: val, 333 frames with candidates

Frames without any candidate (the vehicle stops without asking): 3. Baselines and runs are scored on the same frames.

### Path choice, Mode A (keep going)

| metric | random | straightest | cheapest | driver (oracle) | driven path (reference) | val_ft-nomatch-skip-epoch1 | val_ft-nomatch-skip-epoch2 | val_ft-nomatch-skip-epoch3 | val_ft-nomatch-skip-epoch4 | val_ft-nomatch-skip-epoch5 | val_ft-nomatch-keep-epoch1 | val_ft-nomatch-keep-epoch2 | val_ft-nomatch-keep-epoch3 | val_ft-nomatch-keep-epoch4 | val_ft-nomatch-keep-epoch5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frames with a choice | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 |
| choice matches the driver | 33% | 44% | 50% | 78% | 100% | 49% | 56% | 52% | 56% | 57% | 52% | 53% | 54% | 53% | 54% |
| ... in frames with a real choice | 34% | 45% | 52% | 81% | 100% | 51% | 57% | 54% | 58% | 59% | 54% | 55% | 56% | 55% | 56% |
| ... in turn frames (driver's word not straight) | 22% | 23% | 42% | 49% | 100% | 34% | 43% | 30% | 35% | 35% | 30% | 29% | 22% | 27% | 35% |
| choice is the driver's closest candidate | 33% | 45% | 52% | 100% | 100% | 55% | 61% | 56% | 62% | 63% | 57% | 57% | 54% | 59% | 59% |
| chosen path safe to tick 1 / to the end | 100% / 98% | 100% / 98% | 100% / 99% | 100% / 99% | 100% / 100% | 100% / 99% | 100% / 99% | 100% / 99% | 100% / 99% | 100% / 99% | 100% / 99% | 100% / 99% | 100% / 99% | 100% / 99% | 100% / 99% |
| chose a puddle when a dry path existed | 49% | 70% | 63% | 70% | - | 77% | 74% | 77% | 79% | 84% | 77% | 84% | 86% | 88% | 86% |
| water / puddle / mud on the path (share of length) | 0.0% / 3.4% / 9.9% | 0.0% / 4.0% / 10.9% | 0.0% / 4.0% / 11.6% | 0.0% / 4.1% / 10.8% | 0.0% / 2.2% / 1.7% | 0.0% / 4.0% / 11.3% | 0.0% / 4.0% / 11.8% | 0.0% / 4.1% / 11.5% | 0.0% / 4.4% / 11.7% | 0.0% / 4.4% / 11.5% | 0.0% / 4.2% / 11.6% | 0.0% / 4.5% / 11.2% | 0.0% / 4.5% / 10.9% | 0.0% / 4.6% / 11.4% | 0.0% / 4.5% / 11.6% |
| grass-bush / tree-log / no ground truth | 59.2% / 0.2% / 26.7% | 58.0% / 0.2% / 26.3% | 57.6% / 0.1% / 26.3% | 58.2% / 0.0% / 26.4% | 68.6% / 0.0% / 27.4% | 58.0% / 0.0% / 26.4% | 57.4% / 0.0% / 26.4% | 57.6% / 0.1% / 26.3% | 57.3% / 0.1% / 26.3% | 57.3% / 0.1% / 26.4% | 57.6% / 0.1% / 26.3% | 57.5% / 0.1% / 26.3% | 57.8% / 0.1% / 26.3% | 57.2% / 0.1% / 26.3% | 57.0% / 0.1% / 26.4% |
| closest hard obstacle (mean) | 3.44 m | 3.47 m | 3.47 m | 3.47 m | 3.39 m | 3.54 m | 3.54 m | 3.55 m | 3.52 m | 3.53 m | 3.56 m | 3.54 m | 3.57 m | 3.56 m | 3.52 m |
| near misses (hard obstacle < 0.75 m) | 9.0% | 7.5% | 4.5% | 6.9% | 8.1% | 5.1% | 5.4% | 5.4% | 5.7% | 5.7% | 4.8% | 4.8% | 4.2% | 5.1% | 5.7% |
| ground slope (cm per m) | 5.6 | 5.4 | 5.3 | 5.4 | 5.6 | 5.4 | 5.3 | 5.4 | 5.3 | 5.4 | 5.3 | 5.3 | 5.4 | 5.3 | 5.3 |
| distance from the driven path at 5 / 10 / 15 m | 0.86 m / 1.45 m / 2.99 m | 0.85 m / 1.27 m / 2.04 m | 0.87 m / 1.27 m / 1.90 m | 0.77 m / 0.98 m / 1.37 m | 0.00 m / 0.00 m / 0.02 m | 0.86 m / 1.26 m / 2.04 m | 0.82 m / 1.15 m / 1.64 m | 0.83 m / 1.19 m / 1.74 m | 0.82 m / 1.17 m / 1.69 m | 0.82 m / 1.16 m / 1.70 m | 0.84 m / 1.26 m / 1.86 m | 0.85 m / 1.24 m / 1.86 m | 0.85 m / 1.30 m / 2.03 m | 0.82 m / 1.19 m / 1.74 m | 0.82 m / 1.16 m / 1.74 m |

### Path choice, Mode B (keep going + the driver's direction word)

| metric | random | straightest | cheapest | toward the word | driver (oracle) | driven path (reference) | val_ft-nomatch-skip-epoch1 | val_ft-nomatch-skip-epoch2 | val_ft-nomatch-skip-epoch3 | val_ft-nomatch-skip-epoch4 | val_ft-nomatch-skip-epoch5 | val_ft-nomatch-keep-epoch1 | val_ft-nomatch-keep-epoch2 | val_ft-nomatch-keep-epoch3 | val_ft-nomatch-keep-epoch4 | val_ft-nomatch-keep-epoch5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frames with a choice | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 | 333/333 |
| choice matches the driver | 33% | 44% | 50% | 47% | 78% | 100% | 50% | 55% | 53% | 57% | 56% | 53% | 53% | 53% | 54% | 53% |
| ... in frames with a real choice | 34% | 45% | 52% | 48% | 81% | 100% | 52% | 57% | 54% | 59% | 58% | 54% | 54% | 55% | 55% | 55% |
| ... in turn frames (driver's word not straight) | 22% | 23% | 42% | 34% | 49% | 100% | 34% | 44% | 30% | 36% | 32% | 26% | 27% | 21% | 26% | 32% |
| choice is the driver's closest candidate | 33% | 45% | 52% | 50% | 100% | 100% | 55% | 61% | 56% | 62% | 61% | 56% | 56% | 53% | 58% | 59% |
| chosen path safe to tick 1 / to the end | 100% / 98% | 100% / 98% | 100% / 99% | 100% / 97% | 100% / 99% | 100% / 100% | 100% / 99% | 100% / 99% | 100% / 99% | 100% / 99% | 100% / 99% | 100% / 99% | 100% / 99% | 100% / 99% | 100% / 99% | 100% / 99% |
| chose a puddle when a dry path existed | 49% | 70% | 63% | 74% | 70% | - | 77% | 79% | 74% | 79% | 84% | 74% | 84% | 86% | 88% | 84% |
| water / puddle / mud on the path (share of length) | 0.0% / 3.4% / 9.9% | 0.0% / 4.0% / 10.9% | 0.0% / 4.0% / 11.6% | 0.0% / 4.0% / 11.0% | 0.0% / 4.1% / 10.8% | 0.0% / 2.2% / 1.7% | 0.0% / 4.0% / 11.2% | 0.0% / 4.0% / 11.8% | 0.0% / 4.2% / 11.4% | 0.0% / 4.4% / 11.7% | 0.0% / 4.5% / 11.5% | 0.0% / 4.1% / 11.2% | 0.0% / 4.5% / 11.3% | 0.0% / 4.4% / 10.9% | 0.0% / 4.6% / 11.4% | 0.0% / 4.5% / 11.6% |
| grass-bush / tree-log / no ground truth | 59.2% / 0.2% / 26.7% | 58.0% / 0.2% / 26.3% | 57.6% / 0.1% / 26.3% | 57.6% / 0.1% / 26.6% | 58.2% / 0.0% / 26.4% | 68.6% / 0.0% / 27.4% | 58.0% / 0.0% / 26.4% | 57.4% / 0.0% / 26.4% | 57.6% / 0.1% / 26.4% | 57.2% / 0.1% / 26.3% | 57.3% / 0.1% / 26.3% | 58.1% / 0.1% / 26.3% | 57.4% / 0.1% / 26.3% | 57.7% / 0.1% / 26.3% | 57.2% / 0.1% / 26.3% | 57.0% / 0.1% / 26.4% |
| closest hard obstacle (mean) | 3.44 m | 3.47 m | 3.47 m | 3.41 m | 3.47 m | 3.39 m | 3.54 m | 3.53 m | 3.56 m | 3.53 m | 3.55 m | 3.58 m | 3.55 m | 3.58 m | 3.59 m | 3.53 m |
| near misses (hard obstacle < 0.75 m) | 9.0% | 7.5% | 4.5% | 9.9% | 6.9% | 8.1% | 5.1% | 5.4% | 5.1% | 5.4% | 5.7% | 4.5% | 4.5% | 4.2% | 4.5% | 5.7% |
| ground slope (cm per m) | 5.6 | 5.4 | 5.3 | 5.4 | 5.4 | 5.6 | 5.3 | 5.3 | 5.3 | 5.3 | 5.3 | 5.3 | 5.4 | 5.4 | 5.3 | 5.3 |
| distance from the driven path at 5 / 10 / 15 m | 0.86 m / 1.45 m / 2.99 m | 0.85 m / 1.27 m / 2.04 m | 0.87 m / 1.27 m / 1.90 m | 0.84 m / 1.21 m / 1.94 m | 0.77 m / 0.98 m / 1.37 m | 0.00 m / 0.00 m / 0.02 m | 0.86 m / 1.27 m / 2.05 m | 0.82 m / 1.15 m / 1.64 m | 0.83 m / 1.18 m / 1.72 m | 0.82 m / 1.16 m / 1.68 m | 0.82 m / 1.16 m / 1.71 m | 0.85 m / 1.33 m / 1.98 m | 0.85 m / 1.25 m / 1.90 m | 0.85 m / 1.31 m / 2.05 m | 0.83 m / 1.21 m / 1.80 m | 0.82 m / 1.16 m / 1.74 m |

### Safe distance (ticks)

| metric | always "all" | val_ft-nomatch-skip-epoch1 A | val_ft-nomatch-skip-epoch1 B | val_ft-nomatch-skip-epoch2 A | val_ft-nomatch-skip-epoch2 B | val_ft-nomatch-skip-epoch3 A | val_ft-nomatch-skip-epoch3 B | val_ft-nomatch-skip-epoch4 A | val_ft-nomatch-skip-epoch4 B | val_ft-nomatch-skip-epoch5 A | val_ft-nomatch-skip-epoch5 B | val_ft-nomatch-keep-epoch1 A | val_ft-nomatch-keep-epoch1 B | val_ft-nomatch-keep-epoch2 A | val_ft-nomatch-keep-epoch2 B | val_ft-nomatch-keep-epoch3 A | val_ft-nomatch-keep-epoch3 B | val_ft-nomatch-keep-epoch4 A | val_ft-nomatch-keep-epoch4 B | val_ft-nomatch-keep-epoch5 A | val_ft-nomatch-keep-epoch5 B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ticks answered | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 | 1107/1107 |
| exact / within one tick | 98% / 100% | 96% / 100% | 97% / 100% | 98% / 100% | 98% / 100% | 98% / 100% | 98% / 100% | 98% / 100% | 98% / 100% | 98% / 100% | 98% / 100% | 98% / 100% | 98% / 100% | 95% / 100% | 95% / 100% | 98% / 100% | 98% / 100% | 98% / 100% | 98% / 100% | 98% / 100% | 98% / 100% |
| too optimistic / too cautious | 2% / 0% | 1% / 2% | 1% / 2% | 2% / 0% | 2% / 0% | 1% / 0% | 1% / 0% | 2% / 0% | 2% / 0% | 1% / 0% | 1% / 0% | 2% / 0% | 2% / 0% | 0% / 5% | 0% / 5% | 1% / 1% | 1% / 1% | 1% / 1% | 1% / 1% | 1% / 0% | 1% / 0% |
| unsafe stretch called safe past it | 15/15 | 14/15 | 14/15 | 15/15 | 15/15 | 14/15 | 14/15 | 15/15 | 15/15 | 12/15 | 13/15 | 15/15 | 15/15 | 4/15 | 4/15 | 11/15 | 8/15 | 11/15 | 11/15 | 11/15 | 11/15 |
| label "none": answered exactly | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) | - (none) |
| label "tick 1": answered exactly | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| label "tick 2": answered exactly | 0/17 | 3/17 | 3/17 | 0/17 | 0/17 | 2/17 | 2/17 | 0/17 | 0/17 | 3/17 | 2/17 | 0/17 | 0/17 | 13/17 | 13/17 | 5/17 | 9/17 | 5/17 | 5/17 | 5/17 | 5/17 |
| label "all": answered exactly | 1089/1089 | 1064/1089 | 1069/1089 | 1089/1089 | 1089/1089 | 1088/1089 | 1088/1089 | 1088/1089 | 1088/1089 | 1087/1089 | 1087/1089 | 1089/1089 | 1089/1089 | 1039/1089 | 1038/1089 | 1083/1089 | 1081/1089 | 1083/1089 | 1080/1089 | 1085/1089 | 1084/1089 |

### Continue / stop

| metric | always continue | val_ft-nomatch-skip-epoch1 A | val_ft-nomatch-skip-epoch1 B | val_ft-nomatch-skip-epoch2 A | val_ft-nomatch-skip-epoch2 B | val_ft-nomatch-skip-epoch3 A | val_ft-nomatch-skip-epoch3 B | val_ft-nomatch-skip-epoch4 A | val_ft-nomatch-skip-epoch4 B | val_ft-nomatch-skip-epoch5 A | val_ft-nomatch-skip-epoch5 B | val_ft-nomatch-keep-epoch1 A | val_ft-nomatch-keep-epoch1 B | val_ft-nomatch-keep-epoch2 A | val_ft-nomatch-keep-epoch2 B | val_ft-nomatch-keep-epoch3 A | val_ft-nomatch-keep-epoch3 B | val_ft-nomatch-keep-epoch4 A | val_ft-nomatch-keep-epoch4 B | val_ft-nomatch-keep-epoch5 A | val_ft-nomatch-keep-epoch5 B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| action correct | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| stop frames (label) / said stop / both | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| stop recall / precision | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - | - / - |

