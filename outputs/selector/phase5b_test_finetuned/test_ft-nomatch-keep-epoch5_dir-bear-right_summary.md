## Selector: Qwen/Qwen3-VL-4B-Instruct (4bit) + adapter outputs/adapters/nomatch-keep/epoch5, reasoning not asked, test every 1, history own, relaxed candidates, picture version 2

| metric | Mode B |
|---|---|
| frames (asked / no candidates) | 328 / 3 |
| valid answers | 100% (328/328) |
| choice matches the driver (1 m) | 61% (200/328) |
| ... in frames with a real choice | 61% (199/326) |
| choice is the driver's closest candidate | 52% (170/328) |
| chosen candidate safe to tick 1 / to the end | 100% (328/328) / 100% (328/328) |
| chose a puddle when a dry path existed | 62% (8/13) |
| ticks answered | 100% (1123/1123) |
| ticks exact / within one | 99% (1115/1123) / 100% (1123/1123) |
| ticks too optimistic / too cautious | 0% (4/1123) / 0% (4/1123) |
| action correct | 100% (328/328) |
| said stop / stop is right | 0 / 0 |
| seconds per answer (mean) | 1.2 |

Mode B, choice matches the driver per direction word:

- bear left: 0% (0/3)
- slight left: 45% (27/60)
- slight right: 55% (11/20)
- straight: 66% (162/245)
