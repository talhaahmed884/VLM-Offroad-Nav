## Selector: Qwen/Qwen3-VL-4B-Instruct (4bit), test every 1, history wrong, relaxed candidates, picture version 2

| metric | Mode A | Mode B |
|---|---|---|
| frames (asked / no candidates) | 328 / 3 | 328 / 3 |
| valid answers | 100% (328/328) | 100% (328/328) |
| choice matches the driver (1 m) | 45% (147/328) | 48% (157/328) |
| ... in frames with a real choice | 45% (146/326) | 48% (156/326) |
| choice is the driver's closest candidate | 33% (107/328) | 34% (113/328) |
| chosen candidate safe to tick 1 / to the end | 100% (328/328) / 99% (326/328) | 100% (328/328) / 99% (326/328) |
| chose a puddle when a dry path existed | 54% (7/13) | 46% (6/13) |
| ticks answered | 100% (1123/1123) | 100% (1123/1123) |
| ticks exact / within one | 1% (15/1123) / 78% (877/1123) | 2% (18/1123) / 63% (703/1123) |
| ticks too optimistic / too cautious | 0% (0/1123) / 99% (1108/1123) | 0% (1/1123) / 98% (1104/1123) |
| action correct | 100% (328/328) | 100% (328/328) |
| said stop / stop is right | 0 / 0 | 0 / 0 |
| seconds per answer (mean) | 1.5 | 1.5 |

Mode B, choice matches the driver per direction word:

- bear left: 0% (0/3)
- slight left: 37% (22/60)
- slight right: 50% (10/20)
- straight: 51% (125/245)
