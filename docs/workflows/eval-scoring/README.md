# Handover eval scoring

中文版：[README.zh.md](README.zh.md)

Each model gets one score in 0-1: **total = 0.5 x completion + 0.5 x quality**, every number with a 95% confidence
interval (CI). The CI width comes from the number of trials: 10 trials give wide intervals, so two models are only
different when their intervals do not overlap. Rules agreed with the user on 2026-10-23.

```bash
python docs/workflows/eval-scoring/report.py <out_prefix> <human_ref_demo_dir> <dataset_dir> ...
```

- `human_ref_demo_dir`: human reference demos, currently `wwqcleverghost/v4-handover2-trimmed`.
- `dataset_dir`: eval datasets (have `success.csv`) and/or demo datasets (quality only). Datasets nested in `dataset/` work.
- Writes `<out_prefix>.csv` (one row per dataset, all scores and CIs) and `<out_prefix>.svg` (one stacked bar per
  dataset: blue = 0.5 x completion, orange = 0.5 x quality, whisker = CI of the total; demo datasets draw completion as a
  dashed 1.0 box for comparison). Put results in `outputs/eval_scores_<date>/`. Example: `outputs/eval_scores_20261023/`.
- Reads only `observation.state` and `action`; no video. lerobot env (no matplotlib needed).

## Completion (evals only)

Five stages, detected from the grippers by `score.py` (holding = commanded closed while the measured position stalls
>= 2 units above the command for >= 0.5 s):
1. reach (前往) — an arm moved > 30 units and opened its gripper
2. grasp (抓住) — the tape is held (regrasps allowed)
3. no drop (交接前不掉) — while the first arm still holds, the other arm starts holding or its open gripper is
   commanded closed (a take-over attempt). A failed take-over counts against "handover", not here (fixed
   2026-10-23; before that v6-flow_h16_s8 ep18 and ep34 were misattributed to this stage)
4. handover (交接成功) — both hold >= 0.5 s, then the first arm opens
5. place (放好) — the second arm opens afterwards

- **Stage pass rate = (passed + 1) / (reached + 2)**. A trial that never reached a stage does not count against it.
  The +1/+2 pulls stages with few trials toward 0.5 (1/2 becomes 0.5, 7/7 becomes 0.89).
- Each stage rate is **truncated**: rate <= 0.5 scores 0, 1.0 scores 1, linear in between. Completion = mean of the
  5 truncated rates.
- The human success label in `success.csv` is final (no video review): label success -> all stages pass; label fail
  but all 5 auto stages pass -> place fails. Auto and human agreed on 105/110 trials (2026-10-20).
- Also reported, not in the score: plain success rate (e.g. 7/10) with a Wilson CI, and each stage as
  `passed/reached [CI]` (the funnel: shows where a model fails).

## Quality (evals and demo datasets)

Per trial, weighted, compared with the human reference median:
- **smooth (0.4)** — small joint reversals per second of motion (`smooth.py`: 0.5 dead band, swing < 5)
- **speed (0.3)** — task time from first motion to the place release; successful trials only
- **idle (0.3)** — share of the task spent standing still (< 5 units/s for >= 0.5 s)

smooth and speed score **1 at or better than the human median, 0.5 at 2x worse, 0 at 4x worse** (each doubling costs
half). idle = 1 - idle share, truncated like the stage rates. Human demos themselves score about 0.9 (half of them are
worse than their own median); this is not comparable with numbers from other projects.

## Uncertainty

Stage and success rates: Wilson 95% CI. Completion, quality and total: bootstrap over trials (2000 resamples, 2.5-97.5
percentiles). 10 trials cannot separate 7/10 from 4/10 (Fisher p ≈ 0.37); say so when intervals overlap, and prefer
more trials of the top models over finer scoring.

## Optional: pick eval trials as training data (very strict, added with the user's OK on 2026-10-23)

Keep only successful trials that are no worse than the human demos on every measure; all must hold:
1. human label success and all 5 auto stages pass;
2. no regrasp: at most 2 gripper closes per arm (not the hold count: one grasp often splits into two holds, in the
   human demos too);
3. all three quality sub-scores are 1: jitter and task time at or below the human median, no idle time;
4. curation checklist: tape visible on the table in the last head frame, first holding arm follows the midline rule
   (L x<=0.49, R x>=0.51).

Tape standing at the start cannot be told reliably from one frame: look at the kept trials with `sheets.py`.

```bash
PYTHONPATH=src:docs/workflows/eval-scoring:docs/workflows/handover-dataset-curation:examples/xlerobot \
  python docs/workflows/eval-scoring/select_strict.py <out_prefix> <human_ref_demo_dir> <eval_dir> ...
```
Writes `<out_prefix>.csv` (pass/fail of every check per trial) and `<out_prefix>_spec.json` (for
`trim_stagnant_frames.py --spec`). Eval data is 12-dim (no head servos, no base), so it cannot share one spec with the
17-dim human demos.

## Optional: look at a trial

Only when asked to explain a failure. Contact sheets, one cell every 2 s, or every 1/3 s for a window:
`PYTHONPATH=src python docs/workflows/eval-scoring/sheets.py <eval_dataset_dir> <out_dir> [<ep> <t0> <t1>]`.
Gripper captions `L cmd/state`: 1 = closed; holding = command ~1-5 with state clearly higher; slip = state falls to the
command while it stays closed; release = command opens first. Tape hidden from the head camera at grasp time is normal.

History: before 2026-10-23 trials were scored 0-5 with every stage after the first failure counted as 0, and
smoothness was reported separately. HF `wwqcleverghost/scores-v4-v5` holds those scores.

## Optional: per-frame stage labels

`score.stage_frames(state, action)` returns one label per frame: 0 reaching, 1 arm A holds, 2 both hold, 3 arm B holds,
4 placed (B opened after its hold). Boundaries come from the same hold runs as the completion stages; an episode
without a full handover keeps its last reached label to the end. Checked 2026-10-29 on v4-handover2-trimmed: labels never
go backwards, and "reached 4" matches the place stage of `score()` on 50/50 episodes (also 50/50 on the v6-flow_h16_s8 eval).
Not yet written into any dataset; what the labels are for is still open.
