# Selector runs (answers of the VLM), by collection

Each run (one **tag**) has `<tag>_modeA.jsonl` / `<tag>_modeB.jsonl` (one line per frame: the
picture's colors, the labels, the raw reply and its score), `<tag>_summary.md`, and its first few
pictures (`<tag>_<sequence>_<frame>_mode<A|B>.png`). Folders are named so that they sort in the
order the runs were collected. `scripts/evaluate.py --runs <tag>` finds a tag in any folder;
`scripts/run_selector.py` refuses a tag that is already in a folder unless `--out-dir` points there.
All runs use the relaxed candidates; picture version 2 unless noted.

| folder | collected | split | what | report |
|---|---|---|---|---|
| `phase3a_val_zeroshot_picture-v1/` | 2026-11-23 | val, every 4th frame (84) | zero-shot 4B and 8B, with reasoning, **picture version 1** (lines could hide each other) | numbers in `docs/approach-2-plan.md`, phase 3 |
| `phase3b_val_zeroshot_picture-v2/` | 2026-11-23 | val, every 4th frame (84) | the same with picture version 2 (shared stretches side by side) | plan doc, phase 3 |
| `phase4_test_zeroshot/` | 2026-11-23 (Polaris) | test, all 331 frames | zero-shot 4B: own history, no history (`_nohist`), wrong history (`_wronghist`), forced word in Mode B (`_dir-bear-left`, `_dir-bear-right`); zero-shot 8B | `outputs/eval/test_evaluation.md` |
| `phase5a_val_epochs/` | 2026-11-24 (Polaris) | val, all 336 frames | fine-tuned 4B, every epoch of both datasets (`val_ft-nomatch-{skip,keep}-epoch1..5`), asked without reasoning | `outputs/eval/val_epochs.md` |
| `phase5b_test_finetuned/` | 2026-11-24 (Polaris) | test, all 331 frames | the chosen adapters, skip epoch 4 (**reported model**) and keep epoch 5, with the same history and word tests as phase 4; zero-shot 4B asked without reasoning (`_noreason`) for a fair comparison | `outputs/eval/test_finetuned.md`, `test_ft_skip_behaviour.md`, `test_ft_keep_behaviour.md` |

New runs are written to `outputs/selector/` itself; move them into a new folder
(`phase6_...`) once they are finished, and add a row here.
