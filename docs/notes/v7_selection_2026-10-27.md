# v7 dataset selection (2026-10-27)

中文: [v7_selection_2026-10-27.zh.md](v7_selection_2026-10-27.zh.md)

`wwqcleverghost/v7-handover-clean`: 53 episodes (handover2 38 + handover1 15), 17-dim. On HF (private) since 2026-10-29, revision `fee2b18a` (`images/` placeholder dirs not uploaded); also on server1 (`/media/data/wwq/data/v7-handover-clean`).
Spec: `examples/xlerobot/v7_handover_clean.json`; per episode: [v7_selection_2026-10-27.csv](v7_selection_2026-10-27.csv) (column `v7`).
Built at git `d0c28aca` (uncommitted changes in the tree) from `wwqcleverghost/v2-handover2` and `wwqcleverghost/v1-handover1` (local cache).

```bash
PYTHONPATH=src python examples/xlerobot/trim_stagnant_frames.py --spec examples/xlerobot/v7_handover_clean.json --dst wwqcleverghost/v7-handover-clean --reset_tol 10
```

User's request (no more manual review): drop tape standing at the start and tape put in the basket (as in v6), quality per the current rules, about 50 episodes, loosen the rules if too few.

1. Trim rule "leader start/stop": head = first leader-arm input, tail = reset basically done (`--reset_tol 10`). Cut straight from the raw recordings, not from an already trimmed copy (no second video re-encode).
2. Checklist (the v6 manual verdicts, `verdict` in [handover_grades_2026-10-23.csv](handover_grades_2026-10-23.csv)):
   handover2 drops 3 4 5 32 (tape in the basket), 26 (recording stops at the release), 14 30 (regrasp);
   handover1 drops 3 6 7 8 15 23 30 36 39 46 47 49 (tape standing at the start), 13 31 18 (messy regrasps), 2 4 10 16 21 45 (regrasp, recovery only).
3. The user's manual "drop" marks on the web page (`examples/xlerobot/dataset_labels.json`): handover1 7 13 15 17 24 39 46 48 49.
4. Quality: the `curate.py` default preset (snaps, pauses > 1 s, back-and-forth >= 2x, follower jerk >= 2x) left only 43, so two rules were loosened:
   snaps from "drop at 1" to "drop at 2 or more" (`snap.min=2`), follower jerk from 2x to 2.5x (`jerk.x=2.5`).

| Setting | Left |
|---|---|
| default preset + checklist + manual | 43 |
| jerk 2.5x | 48 |
| snaps >= 2 | 46 |
| **snaps >= 2 + jerk 2.5x (used)** | **53** |
| snaps >= 3 + jerk 2.5x | 56 |

Kept: handover2 0 2 6 7 8 9 10 11 12 13 15 16 17 18 19 20 21 22 23 27 28 29 31 33 34 36 37 38 39 40 42 43 44 45 46 47 48 49;
handover1 0 1 5 9 11 12 14 19 20 25 27 34 37 42 44.
