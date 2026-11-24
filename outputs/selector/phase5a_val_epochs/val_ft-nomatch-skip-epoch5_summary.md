## Selector: Qwen/Qwen3-VL-4B-Instruct (4bit) + adapter outputs/adapters/nomatch-skip/epoch5, reasoning not asked, val every 1, history own, relaxed candidates, picture version 2

| metric | Mode A | Mode B |
|---|---|---|
| frames (asked / no candidates) | 333 / 3 | 333 / 3 |
| valid answers | 100% (333/333) | 100% (333/333) |
| choice matches the driver (1 m) | 57% (190/333) | 56% (186/333) |
| ... in frames with a real choice | 59% (189/321) | 58% (185/321) |
| choice is the driver's closest candidate | 63% (209/333) | 61% (203/333) |
| chosen candidate safe to tick 1 / to the end | 100% (333/333) / 99% (331/333) | 100% (333/333) / 99% (331/333) |
| chose a puddle when a dry path existed | 84% (36/43) | 84% (36/43) |
| ticks answered | 100% (1107/1107) | 100% (1107/1107) |
| ticks exact / within one | 98% (1090/1107) / 100% (1106/1107) | 98% (1089/1107) / 100% (1106/1107) |
| ticks too optimistic / too cautious | 1% (15/1107) / 0% (2/1107) | 1% (16/1107) / 0% (2/1107) |
| action correct | 100% (333/333) | 100% (333/333) |
| said stop / stop is right | 0 / 0 | 0 / 0 |
| seconds per answer (mean) | 1.1 | 1.1 |

Mode B, choice matches the driver per direction word:

- bear left: 0% (0/4)
- bear right: 0% (0/7)
- slight left: 44% (24/55)
- slight right: 9% (1/11)
- straight: 63% (161/256)
