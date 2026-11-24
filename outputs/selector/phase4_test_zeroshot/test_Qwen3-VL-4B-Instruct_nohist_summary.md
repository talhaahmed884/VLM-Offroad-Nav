## Selector: Qwen/Qwen3-VL-4B-Instruct (4bit), test every 1, history none, relaxed candidates, picture version 2

| metric | Mode A | Mode B |
|---|---|---|
| frames (asked / no candidates) | 328 / 3 | 328 / 3 |
| valid answers | 100% (328/328) | 100% (328/328) |
| choice matches the driver (1 m) | 45% (146/328) | 47% (155/328) |
| ... in frames with a real choice | 44% (145/326) | 47% (154/326) |
| choice is the driver's closest candidate | 30% (100/328) | 35% (114/328) |
| chosen candidate safe to tick 1 / to the end | 100% (328/328) / 100% (327/328) | 100% (328/328) / 99% (325/328) |
| chose a puddle when a dry path existed | 46% (6/13) | 46% (6/13) |
| ticks answered | 100% (1123/1123) | 100% (1123/1123) |
| ticks exact / within one | 2% (21/1123) / 75% (846/1123) | 1% (16/1123) / 66% (743/1123) |
| ticks too optimistic / too cautious | 0% (0/1123) / 98% (1102/1123) | 0% (0/1123) / 99% (1107/1123) |
| action correct | 100% (328/328) | 100% (328/328) |
| said stop / stop is right | 0 / 0 | 0 / 0 |
| seconds per answer (mean) | 1.3 | 1.3 |

Mode B, choice matches the driver per direction word:

- bear left: 0% (0/3)
- slight left: 38% (23/60)
- slight right: 50% (10/20)
- straight: 50% (122/245)
