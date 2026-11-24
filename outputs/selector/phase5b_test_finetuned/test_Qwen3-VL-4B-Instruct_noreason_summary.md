## Selector: Qwen/Qwen3-VL-4B-Instruct (4bit), reasoning not asked, test every 1, history own, relaxed candidates, picture version 2

| metric | Mode A | Mode B |
|---|---|---|
| frames (asked / no candidates) | 328 / 3 | 328 / 3 |
| valid answers | 100% (328/328) | 100% (328/328) |
| choice matches the driver (1 m) | 44% (143/328) | 48% (159/328) |
| ... in frames with a real choice | 44% (142/326) | 48% (158/326) |
| choice is the driver's closest candidate | 31% (102/328) | 34% (112/328) |
| chosen candidate safe to tick 1 / to the end | 100% (328/328) / 99% (326/328) | 100% (328/328) / 99% (326/328) |
| chose a puddle when a dry path existed | 46% (6/13) | 46% (6/13) |
| ticks answered | 100% (1123/1123) | 100% (1123/1123) |
| ticks exact / within one | 1% (12/1123) / 79% (891/1123) | 1% (15/1123) / 62% (698/1123) |
| ticks too optimistic / too cautious | 0% (0/1123) / 99% (1111/1123) | 0% (0/1123) / 99% (1108/1123) |
| action correct | 100% (328/328) | 100% (328/328) |
| said stop / stop is right | 0 / 0 | 0 / 0 |
| seconds per answer (mean) | 0.9 | 0.9 |

Mode B, choice matches the driver per direction word:

- bear left: 0% (0/3)
- slight left: 42% (25/60)
- slight right: 50% (10/20)
- straight: 51% (124/245)
