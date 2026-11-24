## Selector: Qwen/Qwen3-VL-8B-Instruct (4bit), test every 1, history own, relaxed candidates, picture version 2

| metric | Mode A | Mode B |
|---|---|---|
| frames (asked / no candidates) | 328 / 3 | 328 / 3 |
| valid answers | 100% (328/328) | 100% (328/328) |
| choice matches the driver (1 m) | 42% (137/328) | 47% (155/328) |
| ... in frames with a real choice | 42% (136/326) | 47% (154/326) |
| choice is the driver's closest candidate | 23% (76/328) | 30% (99/328) |
| chosen candidate safe to tick 1 / to the end | 100% (328/328) / 100% (327/328) | 100% (328/328) / 100% (328/328) |
| chose a puddle when a dry path existed | 31% (4/13) | 31% (4/13) |
| ticks answered | 100% (1123/1123) | 100% (1123/1123) |
| ticks exact / within one | 40% (445/1123) / 51% (578/1123) | 50% (556/1123) / 61% (681/1123) |
| ticks too optimistic / too cautious | 0% (1/1123) / 60% (677/1123) | 0% (1/1123) / 50% (566/1123) |
| action correct | 100% (328/328) | 100% (328/328) |
| said stop / stop is right | 0 / 0 | 0 / 0 |
| seconds per answer (mean) | 1.0 | 1.0 |

Mode B, choice matches the driver per direction word:

- bear left: 0% (0/3)
- slight left: 33% (20/60)
- slight right: 45% (9/20)
- straight: 51% (126/245)
