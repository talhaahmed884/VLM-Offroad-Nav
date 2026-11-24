## Selector: Qwen/Qwen3-VL-4B-Instruct (4bit) + adapter outputs/adapters/nomatch-keep/epoch3, reasoning not asked, val every 1, history own, relaxed candidates, picture version 2

| metric | Mode A | Mode B |
|---|---|---|
| frames (asked / no candidates) | 333 / 3 | 333 / 3 |
| valid answers | 100% (333/333) | 100% (333/333) |
| choice matches the driver (1 m) | 54% (181/333) | 53% (178/333) |
| ... in frames with a real choice | 56% (180/321) | 55% (177/321) |
| choice is the driver's closest candidate | 54% (181/333) | 53% (178/333) |
| chosen candidate safe to tick 1 / to the end | 100% (333/333) / 99% (331/333) | 100% (333/333) / 99% (331/333) |
| chose a puddle when a dry path existed | 86% (37/43) | 86% (37/43) |
| ticks answered | 100% (1107/1107) | 100% (1107/1107) |
| ticks exact / within one | 98% (1088/1107) / 100% (1106/1107) | 98% (1090/1107) / 100% (1106/1107) |
| ticks too optimistic / too cautious | 1% (13/1107) / 1% (6/1107) | 1% (9/1107) / 1% (8/1107) |
| action correct | 100% (333/333) | 100% (333/333) |
| said stop / stop is right | 0 / 0 | 0 / 0 |
| seconds per answer (mean) | 1.1 | 1.1 |

Mode B, choice matches the driver per direction word:

- bear left: 0% (0/4)
- bear right: 0% (0/7)
- slight left: 27% (15/55)
- slight right: 9% (1/11)
- straight: 63% (162/256)
