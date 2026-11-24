## Selector: Qwen/Qwen3-VL-4B-Instruct (4bit), val every 4, history own, relaxed candidates

| metric | Mode A | Mode B |
|---|---|---|
| frames (asked / no candidates) | 83 / 1 | 83 / 1 |
| valid answers | 100% (83/83) | 100% (83/83) |
| choice matches the driver (1 m) | 25% (21/83) | 28% (23/83) |
| ... in frames with a real choice | 26% (21/80) | 29% (23/80) |
| choice is the driver's closest candidate | 31% (26/83) | 34% (28/83) |
| chosen candidate safe to tick 1 / to the end | 100% (83/83) / 98% (81/83) | 100% (83/83) / 98% (81/83) |
| chose a puddle when a dry path existed | 25% (3/12) | 17% (2/12) |
| ticks answered | 100% (284/284) | 100% (284/284) |
| ticks exact / within one | 2% (5/284) / 72% (205/284) | 2% (5/284) / 61% (173/284) |
| ticks too optimistic / too cautious | 0% (0/284) / 98% (279/284) | 0% (0/284) / 98% (279/284) |
| action correct | 100% (83/83) | 100% (83/83) |
| said stop / stop is right | 0 / 0 | 0 / 0 |
| seconds per answer (mean) | 5.8 | 5.5 |

Mode B, choice matches the driver per direction word:

- bear right: 0% (0/2)
- slight left: 27% (4/15)
- slight right: 0% (0/2)
- straight: 30% (19/64)
