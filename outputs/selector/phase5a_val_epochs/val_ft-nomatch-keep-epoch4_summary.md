## Selector: Qwen/Qwen3-VL-4B-Instruct (4bit) + adapter outputs/adapters/nomatch-keep/epoch4, reasoning not asked, val every 1, history own, relaxed candidates, picture version 2

| metric | Mode A | Mode B |
|---|---|---|
| frames (asked / no candidates) | 333 / 3 | 333 / 3 |
| valid answers | 100% (333/333) | 100% (333/333) |
| choice matches the driver (1 m) | 53% (178/333) | 54% (179/333) |
| ... in frames with a real choice | 55% (177/321) | 55% (178/321) |
| choice is the driver's closest candidate | 59% (195/333) | 58% (194/333) |
| chosen candidate safe to tick 1 / to the end | 100% (333/333) / 99% (331/333) | 100% (333/333) / 99% (331/333) |
| chose a puddle when a dry path existed | 88% (38/43) | 88% (38/43) |
| ticks answered | 100% (1107/1107) | 100% (1107/1107) |
| ticks exact / within one | 98% (1088/1107) / 100% (1106/1107) | 98% (1085/1107) / 100% (1106/1107) |
| ticks too optimistic / too cautious | 1% (13/1107) / 1% (6/1107) | 1% (13/1107) / 1% (9/1107) |
| action correct | 100% (333/333) | 100% (333/333) |
| said stop / stop is right | 0 / 0 | 0 / 0 |
| seconds per answer (mean) | 1.1 | 1.1 |

Mode B, choice matches the driver per direction word:

- bear left: 0% (0/4)
- bear right: 0% (0/7)
- slight left: 35% (19/55)
- slight right: 9% (1/11)
- straight: 62% (159/256)
