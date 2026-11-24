## Selector: Qwen/Qwen3-VL-4B-Instruct (4bit) + adapter outputs/adapters/nomatch-skip/epoch4, reasoning not asked, val every 1, history own, relaxed candidates, picture version 2

| metric | Mode A | Mode B |
|---|---|---|
| frames (asked / no candidates) | 333 / 3 | 333 / 3 |
| valid answers | 100% (333/333) | 100% (333/333) |
| choice matches the driver (1 m) | 56% (188/333) | 57% (189/333) |
| ... in frames with a real choice | 58% (187/321) | 59% (188/321) |
| choice is the driver's closest candidate | 62% (205/333) | 62% (205/333) |
| chosen candidate safe to tick 1 / to the end | 100% (333/333) / 99% (331/333) | 100% (333/333) / 99% (331/333) |
| chose a puddle when a dry path existed | 79% (34/43) | 79% (34/43) |
| ticks answered | 100% (1107/1107) | 100% (1107/1107) |
| ticks exact / within one | 98% (1088/1107) / 100% (1106/1107) | 98% (1088/1107) / 100% (1106/1107) |
| ticks too optimistic / too cautious | 2% (18/1107) / 0% (1/1107) | 2% (18/1107) / 0% (1/1107) |
| action correct | 100% (333/333) | 100% (333/333) |
| said stop / stop is right | 0 / 0 | 0 / 0 |
| seconds per answer (mean) | 1.1 | 1.1 |

Mode B, choice matches the driver per direction word:

- bear left: 0% (0/4)
- bear right: 0% (0/7)
- slight left: 49% (27/55)
- slight right: 9% (1/11)
- straight: 63% (161/256)
