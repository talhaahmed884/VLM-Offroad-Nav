## Selector: Qwen/Qwen3-VL-8B-Instruct (4bit), val every 4, history own, relaxed candidates

| metric | Mode A | Mode B |
|---|---|---|
| frames (asked / no candidates) | 83 / 1 | 83 / 1 |
| valid answers | 100% (83/83) | 100% (83/83) |
| choice matches the driver (1 m) | 19% (16/83) | 27% (22/83) |
| ... in frames with a real choice | 20% (16/80) | 28% (22/80) |
| choice is the driver's closest candidate | 27% (22/83) | 34% (28/83) |
| chosen candidate safe to tick 1 / to the end | 100% (83/83) / 99% (82/83) | 100% (83/83) / 99% (82/83) |
| chose a puddle when a dry path existed | 42% (5/12) | 42% (5/12) |
| ticks answered | 100% (284/284) | 100% (284/284) |
| ticks exact / within one | 37% (104/284) / 49% (140/284) | 45% (127/284) / 56% (158/284) |
| ticks too optimistic / too cautious | 0% (0/284) / 63% (180/284) | 0% (0/284) / 55% (157/284) |
| action correct | 100% (83/83) | 100% (83/83) |
| said stop / stop is right | 0 / 0 | 0 / 0 |
| seconds per answer (mean) | 4.5 | 4.2 |

Mode B, choice matches the driver per direction word:

- bear right: 0% (0/2)
- slight left: 27% (4/15)
- slight right: 0% (0/2)
- straight: 28% (18/64)
