## Phase 6: closed-loop runs in MAVS

Success: trail01 / curves / halfblocked = reach the end; fork = reach the end of the branch the word names (Mode B), either branch (Mode A); blocked = stop before the wall. Word-blind choosers (straightest, cheapest, oracle) are driven in Mode A; their Mode B is the same run (= A). The oracle uses the answer key (the candidate closest to the route): a reference, not a real chooser.

### 1. Per chooser

| chooser | mode | tests passed | collisions | off trail | stopped wrongly | mean completion (driving scenes) | mean distance from route (m) | closest gap to an obstacle (m) |
|---|---|---|---|---|---|---|---|---|
| straightest | A | 3 / 5 | 0 | 1 | 2 | 0.70 | 0.85 | 0.25 |
| straightest | B (= A) | 3 / 6 | 0 | 1 | 2 | 0.70 | 0.85 | 0.25 |
| cheapest | A | 5 / 5 | 0 | 1 | 0 | 0.87 | 0.49 | 0.14 |
| cheapest | B (= A) | 5 / 6 | 0 | 1 | 0 | 0.87 | 0.49 | 0.14 |
| oracle | A | 5 / 5 | 0 | 0 | 0 | 0.96 | 0.30 | 0.80 |
| oracle | B (= A) | 6 / 6 | 0 | 0 | 0 | 0.96 | 0.30 | 0.80 |
| zeroshot | A | 4 / 5 | 0 | 1 | 1 | 0.78 | 0.61 | 0.15 |
| zeroshot | B | 5 / 6 | 0 | 0 | 1 | 0.85 | 0.74 | 0.20 |
| finetuned | A | 5 / 5 | 0 | 1 | 0 | 0.86 | 0.66 | 0.06 |
| finetuned | B | 5 / 6 | 0 | 1 | 0 | 0.86 | 0.68 | 0.06 |

### 2. Per scene (success, outcome, share of the route driven)

| scene | straightest A | straightest B | cheapest A | cheapest B | oracle A | oracle B | zeroshot A | zeroshot B | finetuned A | finetuned B |
|---|---|---|---|---|---|---|---|---|---|---|
| trail01 | OK finished 98% | OK finished 98% | OK finished 98% | OK finished 98% | OK finished 98% | OK finished 98% | OK finished 98% | OK finished 98% | OK finished 98% | OK finished 98% |
| curves | x stopped 55% | x stopped 55% | OK finished 96% | OK finished 96% | OK finished 96% | OK finished 96% | x stopped 56% | x stopped 40% | OK finished 96% | OK finished 96% |
| fork-left | x stopped 55% | x stopped 55% | x off_trail 48% | x off_trail 48% | OK finished 95% | OK finished 95% | OK finished 95% | OK finished 95% | OK finished 96% | OK finished 96% |
| fork-right | x off_trail 49% | x off_trail 49% | OK finished 95% | OK finished 95% | OK finished 95% | OK finished 95% | x off_trail 46% | OK finished 95% | x off_trail 46% | x off_trail 46% |
| blocked | OK stopped 23% | OK stopped 23% | OK stopped 23% | OK stopped 23% | OK stopped 24% | OK stopped 24% | OK stopped 22% | OK stopped 23% | OK stopped 23% | OK stopped 23% |
| halfblocked | OK finished 96% | OK finished 96% | OK finished 96% | OK finished 96% | OK finished 96% | OK finished 96% | OK finished 96% | OK finished 96% | OK finished 96% | OK finished 96% |
| fork (Mode A: either branch) | x | - | OK | - | OK | - | OK | - | OK | - |

### 3. Blocked trail: who stopped, and how far before the wall

| chooser | mode | outcome | why it stopped | stopped this far before the wall (m) | first 'stop' answer this far before the wall (m) |
|---|---|---|---|---|---|
| straightest | A | stopped | no candidates | 9.60 | - |
| cheapest | A | stopped | no candidates | 9.00 | - |
| oracle | A | stopped | no candidates | 8.20 | - |
| zeroshot | A | stopped | no candidates | 10.30 | - |
| zeroshot | B | stopped | no candidates | 9.70 | - |
| finetuned | A | stopped | no candidates | 8.90 | - |
| finetuned | B | stopped | no candidates | 8.90 | - |

### 4. What the choosers did

| chooser | mode | decisions | with candidates | chose the straightest | valid answers | fallbacks | 'stop' answers | every path 'all' | s per answer |
|---|---|---|---|---|---|---|---|---|---|
| straightest | A | 706 | 694 | 1.00 | - | - | - | - | - |
| cheapest | A | 829 | 818 | 0.49 | - | - | - | - | - |
| oracle | A | 888 | 878 | 0.78 | - | - | - | - | - |
| zeroshot | A | 752 | 744 | 0.67 | 1.00 | 0 | 0 | 0.00 | 3.43 |
| zeroshot | B | 788 | 780 | 0.66 | 1.00 | 0 | 0 | 0.00 | 3.39 |
| finetuned | A | 823 | 817 | 0.79 | 1.00 | 0 | 0 | 0.77 | 5.25 |
| finetuned | B | 823 | 816 | 0.80 | 1.00 | 0 | 0 | 0.75 | 4.71 |

### 5. All runs

| scene | chooser | mode | outcome | why stopped | completion | max dist. (m) | closest gap (m) | held | decisions |
|---|---|---|---|---|---|---|---|---|---|
| trail01 | straightest | A | finished | - | 0.98 | 1.15 | 0.59 | 0 | 270 |
| curves | straightest | A | stopped | no candidates | 0.55 | 1.96 | 3.00 | 2 | 104 |
| fork-left | straightest | A | stopped | no candidates | 0.55 | 1.42 | 1.45 | 2 | 73 |
| fork-right | straightest | A | off_trail | - | 0.49 | 2.93 | 1.45 | 0 | 65 |
| blocked | straightest | A | stopped | no candidates | 0.23 | 1.22 | 4.09 | 3 | 38 |
| halfblocked | straightest | A | finished | - | 0.96 | 1.81 | 0.25 | 2 | 156 |
| trail01 | straightest | B (= A) | finished | - | 0.98 | 1.15 | 0.59 | 0 | 270 |
| curves | straightest | B (= A) | stopped | no candidates | 0.55 | 1.96 | 3.00 | 2 | 104 |
| fork-left | straightest | B (= A) | stopped | no candidates | 0.55 | 1.42 | 1.45 | 2 | 73 |
| fork-right | straightest | B (= A) | off_trail | - | 0.49 | 2.93 | 1.45 | 0 | 65 |
| blocked | straightest | B (= A) | stopped | no candidates | 0.23 | 1.22 | 4.09 | 3 | 38 |
| halfblocked | straightest | B (= A) | finished | - | 0.96 | 1.81 | 0.25 | 2 | 156 |
| trail01 | cheapest | A | finished | - | 0.98 | 0.59 | 0.84 | 0 | 270 |
| curves | cheapest | A | finished | - | 0.96 | 1.68 | 2.21 | 0 | 174 |
| fork-left | cheapest | A | off_trail | - | 0.48 | 2.85 | 1.95 | 0 | 64 |
| fork-right | cheapest | A | finished | - | 0.95 | 1.12 | 1.94 | 2 | 125 |
| blocked | cheapest | A | stopped | no candidates | 0.23 | 0.96 | 4.68 | 2 | 39 |
| halfblocked | cheapest | A | finished | - | 0.96 | 1.53 | 0.14 | 6 | 157 |
| trail01 | cheapest | B (= A) | finished | - | 0.98 | 0.59 | 0.84 | 0 | 270 |
| curves | cheapest | B (= A) | finished | - | 0.96 | 1.68 | 2.21 | 0 | 174 |
| fork-left | cheapest | B (= A) | off_trail | - | 0.48 | 2.85 | 1.95 | 0 | 64 |
| fork-right | cheapest | B (= A) | finished | - | 0.95 | 1.12 | 1.94 | 2 | 125 |
| blocked | cheapest | B (= A) | stopped | no candidates | 0.23 | 0.96 | 4.68 | 2 | 39 |
| halfblocked | cheapest | B (= A) | finished | - | 0.96 | 1.53 | 0.14 | 6 | 157 |
| trail01 | oracle | A | finished | - | 0.98 | 0.47 | 0.93 | 0 | 270 |
| curves | oracle | A | finished | - | 0.96 | 1.26 | 2.23 | 0 | 174 |
| fork-left | oracle | A | finished | - | 0.95 | 0.65 | 2.26 | 1 | 124 |
| fork-right | oracle | A | finished | - | 0.95 | 1.11 | 2.12 | 1 | 124 |
| blocked | oracle | A | stopped | no candidates | 0.24 | 0.33 | 4.79 | 2 | 40 |
| halfblocked | oracle | A | finished | - | 0.96 | 0.64 | 0.80 | 5 | 156 |
| trail01 | oracle | B (= A) | finished | - | 0.98 | 0.47 | 0.93 | 0 | 270 |
| curves | oracle | B (= A) | finished | - | 0.96 | 1.26 | 2.23 | 0 | 174 |
| fork-left | oracle | B (= A) | finished | - | 0.95 | 0.65 | 2.26 | 1 | 124 |
| fork-right | oracle | B (= A) | finished | - | 0.95 | 1.11 | 2.12 | 1 | 124 |
| blocked | oracle | B (= A) | stopped | no candidates | 0.24 | 0.33 | 4.79 | 2 | 40 |
| halfblocked | oracle | B (= A) | finished | - | 0.96 | 0.64 | 0.80 | 5 | 156 |
| trail01 | zeroshot | A | finished | - | 0.98 | 1.33 | 0.58 | 0 | 270 |
| curves | zeroshot | A | stopped | no candidates | 0.56 | 2.72 | 2.53 | 2 | 104 |
| fork-left | zeroshot | A | finished | - | 0.95 | 1.72 | 1.32 | 0 | 124 |
| fork-right | zeroshot | A | off_trail | - | 0.46 | 2.80 | 1.32 | 0 | 61 |
| blocked | zeroshot | A | stopped | no candidates | 0.22 | 0.95 | 3.83 | 2 | 37 |
| halfblocked | zeroshot | A | finished | - | 0.96 | 1.56 | 0.15 | 2 | 156 |
| trail01 | zeroshot | B | finished | - | 0.98 | 1.25 | 0.84 | 0 | 270 |
| curves | zeroshot | B | stopped | no candidates | 0.40 | 2.76 | 2.45 | 2 | 75 |
| fork-left | zeroshot | B | finished | - | 0.95 | 1.90 | 1.08 | 1 | 124 |
| fork-right | zeroshot | B | finished | - | 0.95 | 1.41 | 1.90 | 0 | 124 |
| blocked | zeroshot | B | stopped | no candidates | 0.23 | 0.90 | 4.79 | 2 | 38 |
| halfblocked | zeroshot | B | finished | - | 0.96 | 1.66 | 0.20 | 1 | 157 |
| trail01 | finetuned | A | finished | - | 0.98 | 1.03 | 0.75 | 0 | 270 |
| curves | finetuned | A | finished | - | 0.96 | 2.01 | 2.32 | 0 | 173 |
| fork-left | finetuned | A | finished | - | 0.95 | 1.74 | 1.20 | 1 | 124 |
| fork-right | finetuned | A | off_trail | - | 0.46 | 2.86 | 1.20 | 0 | 61 |
| blocked | finetuned | A | stopped | no candidates | 0.23 | 0.71 | 4.30 | 3 | 39 |
| halfblocked | finetuned | A | finished | - | 0.96 | 1.80 | 0.06 | 1 | 156 |
| trail01 | finetuned | B | finished | - | 0.98 | 1.11 | 0.75 | 0 | 270 |
| curves | finetuned | B | finished | - | 0.96 | 2.01 | 2.30 | 0 | 173 |
| fork-left | finetuned | B | finished | - | 0.95 | 1.74 | 1.20 | 1 | 124 |
| fork-right | finetuned | B | off_trail | - | 0.46 | 2.86 | 1.20 | 0 | 61 |
| blocked | finetuned | B | stopped | no candidates | 0.23 | 0.71 | 4.30 | 3 | 39 |
| halfblocked | finetuned | B | finished | - | 0.96 | 1.60 | 0.06 | 2 | 156 |

