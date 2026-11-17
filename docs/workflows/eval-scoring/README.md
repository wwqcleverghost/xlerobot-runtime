# Handover eval scoring

中文版：[README.zh.md](README.zh.md)

Each model gets one score in 0-1: **total = 0.5 x completion + 0.5 x quality**, every number with a 95% confidence
interval (CI). The CI width comes from the number of trials: 10 trials give wide intervals, so two models are only
different when their intervals do not overlap. Rules agreed with the user on 2026-10-23; since 2026-11-14 everything is
scored with one set of stages, the 9 `fine9` stages (user: "whichever stages the score uses, score by those").

```bash
python docs/workflows/eval-scoring/report.py <out_prefix> <human_ref_demo_dir> <dataset_dir> ...
```

- `human_ref_demo_dir`: human reference demos, currently `wwqcleverghost/v4-handover2-trimmed`.
- `dataset_dir`: eval datasets (have `success.csv`) and/or demo datasets (quality only). Datasets nested in `dataset/` work.
- Writes `<out_prefix>.csv` (one row per dataset, all scores and CIs) and `<out_prefix>.svg` (one stacked bar per
  dataset: blue = 0.5 x completion, orange = 0.5 x quality, whisker = CI of the total; demo datasets draw completion as a
  dashed 1.0 box for comparison). Put results in `outputs/eval_scores_<date>/`. Example: `outputs/eval_scores_20261023/`.
- Reads `observation.state`, `action` and each dataset's `episode_end.csv` (stages and task end, written by `mark_end.py`,
  see below). When that file is missing `report.py` writes it first, which decodes the wrist videos once.
  lerobot env (no matplotlib needed).

## Stages

One set of stages for everything, scoring and display: the 9 `fine9` stages of [`../stage-labels/`](../stage-labels/README.md),
A前往, A对位, A抓住, A抓着, AB对位, B抓住, B抓着, B放下, 复位 (stage 0 to 8). Per frame they are `fine9` recomputed from the wrist
videos with the current rule (since 2026-11-17 also for evals that wrote `stages.csv`: that is the stage the model saw, under the
rule of the eval day; the 2026-11-17 touch rule is in [`../stage-labels/README.md`](../stage-labels/README.md)). `mark_end.py` stores per episode
the furthest stage (`top_stage`) and the task end (the first 复位 frame) in `episode_end.csv`.

## Completion (evals only)

- A trial **entered** a stage when its furthest stage is that one or a later one, and **passed** it when it entered the next
  one. The last stage (复位) is passed when the human label says success.
- **The human judgement is final** (no video review needed): label success -> all 9 stages pass. Label fail and the
  trial's note in the dataset viewer names a stage (`AB对位失败`: a stage name followed by 失败) -> the trial got to that
  stage and did not pass it. Label fail without such a note -> the furthest stage the rule saw. The rule reads a gripper
  closing on nothing as a later stage, so a failed trial without a note can be credited with too much: write the note.
- **Stage pass rate = (passed + 1) / (entered + 2)**. A trial that never entered a stage does not count against it.
  The +1/+2 pulls stages with few trials toward 0.5 (1/2 becomes 0.5, 7/7 becomes 0.89).
- Each stage rate is **truncated**: rate <= 0.5 scores 0, 1.0 scores 1, linear in between. Completion = mean of the
  9 truncated rates.
- Also reported, not in the score: plain success rate (e.g. 7/10) with a Wilson CI, and each stage as
  `passed/entered [CI]` (the funnel: shows where a model fails).

## Quality (evals and demo datasets)

Per trial, weighted, compared with the human reference median:
- **smooth (0.4)** — small joint reversals per second of motion (`smooth.py`: 0.5 dead band, swing < 5)
- **speed (0.3)** — task time from first motion to the task end = the first 复位 frame; only trials that are successful
  and reached 复位 (a success in which the rule finds no 复位 has no task time)
- **idle (0.3)** — share of the time from first motion to the task end spent standing still (< 5 units/s for >= 0.5 s);
  a trial that never reached 复位 runs to its `still` / recorded end

The human reference is measured the same way; a demo episode counts as done when it reaches 复位.

smooth and speed score **1 at or better than the human median, 0.5 at 2x worse, 0 at 4x worse** (each doubling costs
half). idle = 1 - idle share, truncated like the stage rates. Human demos themselves score about 0.9 (half of them are
worse than their own median); this is not comparable with numbers from other projects.

## Uncertainty

Stage and success rates: Wilson 95% CI. Completion, quality and total: bootstrap over trials (2000 resamples, 2.5-97.5
percentiles). 10 trials cannot separate 7/10 from 4/10 (Fisher p ≈ 0.37); say so when intervals overlap, and prefer
more trials of the top models over finer scoring.

## Optional: pick eval trials as training data (very strict, added with the user's OK on 2026-10-23)

Keep only successful trials that are no worse than the human demos on every measure; all must hold:
1. human label success and the stage rule saw the trial reach 复位 (`top_stage` 8);
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
From 2026-10-23 to 2026-11-14 completion used 5 stages judged from the grippers by `score.py` (reach, grasp, no drop,
handover, place), the task ended when the second arm opened, and a note never changed a score; the last commit with that
rule is `b9c9af96`. Scores before and after: [`../../notes/scoring_9stage_2026-11-14.md`](../../notes/scoring_9stage_2026-11-14.md).

## Stages and task end of each episode (`mark_end.py`)

The operator ends a trial by hand, a few seconds early or late. `mark_end.py` writes `<dataset dir>/episode_end.csv`
(episode, end_frame, frames, how, top_stage) for an eval or a demo dataset, without cutting anything. `top_stage` = the
highest `fine9` stage any frame is in (the rule alone, no human label). The scores read this file, so run it again after the
stage rule or a hand-set stage start (`stage_edits.json`) changes.

still = the frame from which every arm joint (command and follower) stays within 3 units of its last pose: the robot has stopped
and the rest is waiting.

Trials that reached 复位 (fine9 stage 8, recomputed offline from the wrist videos) end where 复位 starts (user, 2026-11-06): many were stopped mid-reset, so time up to the reset is what all of them share.

- `reset`: end = the first 复位 frame.

Other trials:

- `still`: end = still.
- `human`: moved until the last 0.5 s; end = the recorded length.

The dataset viewer's "阶段 / 进度" page draws it and greys the rest. Since 2026-11-14 the quality score uses it as the task end.

```bash
PYTHONPATH=src python docs/workflows/eval-scoring/mark_end.py ~/.cache/huggingface/lerobot/wwqcleverghost/eval-v8-flow_h16_s8
```

2026-11-06 (git 6e1599b9 + uncommitted changes), successful trials: `eval-v6-flow_h16_s8` 28 reset / 3 human, `eval-v8-flow_h16_s8`
4 reset, the other eval sets all reset except one human in `eval-v6-flow_h16_s8_stagepred`. The human successes are labelled success but fine9
finds no 复位 in them, so they keep the recorded length (v6 ep26, 32, 38).
