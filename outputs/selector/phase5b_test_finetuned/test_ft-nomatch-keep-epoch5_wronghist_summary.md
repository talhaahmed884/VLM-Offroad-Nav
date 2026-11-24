## Selector: Qwen/Qwen3-VL-4B-Instruct (4bit) + adapter outputs/adapters/nomatch-keep/epoch5, reasoning not asked, test every 1, history wrong, relaxed candidates, picture version 2

| metric | Mode A | Mode B |
|---|---|---|
| frames (asked / no candidates) | 328 / 3 | 328 / 3 |
| valid answers | 100% (328/328) | 100% (328/328) |
| choice matches the driver (1 m) | 61% (201/328) | 62% (203/328) |
| ... in frames with a real choice | 61% (200/326) | 62% (202/326) |
| choice is the driver's closest candidate | 53% (173/328) | 53% (175/328) |
| chosen candidate safe to tick 1 / to the end | 100% (328/328) / 100% (328/328) | 100% (328/328) / 100% (328/328) |
| chose a puddle when a dry path existed | 62% (8/13) | 62% (8/13) |
| ticks answered | 100% (1123/1123) | 100% (1123/1123) |
| ticks exact / within one | 99% (1115/1123) / 100% (1123/1123) | 99% (1116/1123) / 100% (1123/1123) |
| ticks too optimistic / too cautious | 0% (4/1123) / 0% (4/1123) | 0% (3/1123) / 0% (4/1123) |
| action correct | 100% (328/328) | 100% (328/328) |
| said stop / stop is right | 0 / 0 | 0 / 0 |
| seconds per answer (mean) | 1.1 | 1.2 |

Mode B, choice matches the driver per direction word:

- bear left: 0% (0/3)
- slight left: 52% (31/60)
- slight right: 55% (11/20)
- straight: 66% (161/245)
