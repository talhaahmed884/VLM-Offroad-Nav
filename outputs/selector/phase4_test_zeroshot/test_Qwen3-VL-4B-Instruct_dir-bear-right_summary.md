## Selector: Qwen/Qwen3-VL-4B-Instruct (4bit), test every 1, history own, relaxed candidates, picture version 2

| metric | Mode B |
|---|---|
| frames (asked / no candidates) | 328 / 3 |
| valid answers | 100% (328/328) |
| choice matches the driver (1 m) | 35% (114/328) |
| ... in frames with a real choice | 35% (113/326) |
| choice is the driver's closest candidate | 17% (56/328) |
| chosen candidate safe to tick 1 / to the end | 100% (328/328) / 100% (327/328) |
| chose a puddle when a dry path existed | 92% (12/13) |
| ticks answered | 100% (1123/1123) |
| ticks exact / within one | 3% (35/1123) / 55% (623/1123) |
| ticks too optimistic / too cautious | 0% (1/1123) / 97% (1087/1123) |
| action correct | 100% (328/328) |
| said stop / stop is right | 0 / 0 |
| seconds per answer (mean) | 1.4 |

Mode B, choice matches the driver per direction word:

- bear left: 0% (0/3)
- slight left: 10% (6/60)
- slight right: 50% (10/20)
- straight: 40% (98/245)
