# Per-episode check of the handover datasets (2026-10-23)

中文: [handover_curation_2026-10-23.zh.md](handover_curation_2026-10-23.zh.md)

v4 = `v4-handover2-trimmed` (all 50 handover2 episodes), v5 = `handover-perfect` (75), strict = `handover-perfect-strict` (64).
All three are subsets of the raw `handover2` and `handover1` episodes, so each of those 100 raw episodes is checked once and
mapped back to the three datasets. Per-episode results: [handover_grades_2026-10-23.csv](handover_grades_2026-10-23.csv) (column `verdict`).

Sources: git `9a881cac` (with uncommitted changes); HF `wwqcleverghost/handover2` @ `46c0695d`, `wwqcleverghost/handover1` @ `5b057e25`
(local cache, recorded as the main sha at the time).

## Method
The per-episode checks and the quality check of `docs/workflows/handover-dataset-curation/README.md`; no new rules.
`docs/workflows/handover-dataset-curation/ep_score.py` automates the checklist: trim range (`keep_range`), tape pixels in the
last head frame, height/width of the tape blob at the start (standing), gripper closes per arm, first holding arm and the
midline rule, duration, longest pause, and the IQR flags of `lerobot_dataset_quality.py`. Episodes with more than 2 closes
on an arm were checked on `sheets.py` sheets for which close actually held.

## Findings
**The misplaced v4 episodes did not get into v5.** handover2 ep 3 4 5 32 (tape put in the basket) and 26 (recording stops at
the release) have no tape in their final frame; neither v5 nor strict contains them, v4 contains all five. All 100 episodes
follow the midline rule.

v5 still has 3 regrasps that were missed before:

| episode | in | problem |
|---|---|---|
| h2 14 | v4, v5 | L misses its first close at the handover, regrasps, then releases from height; the tape lands standing |
| h2 30 | v4, v5, strict | R misses its first close at the handover, reopens and grasps again |
| h1 18 | v5 | L fumbles the grasp for ~6 s (6 closes), slow handover |

| dataset | ep | drop per checklist | pause > 1 s (checklist: look) |
|---|---|---|---|
| v4 | 50 | 7 (h2 3 4 5 14 26 30 32) | h2 1 25 |
| v5 | 75 | 3 (h2 14 30, h1 18) | h2 1 25, h1 29 |
| strict | 64 | 1 (h2 30) | h2 1 |

- More than 2 closes but the extra one is empty, kept: h2 12, 40 (R closes once more after the handover / after placing),
  h1 24 (L closes in the air before reaching).
- Only the handover1 regrasp episodes 2 4 10 16 21 45 are clippable (cut the failed first grasp); per the earlier decision they
  stay in the `-recovery` dataset only.
- IQR flags (meaning "look at it"): the handover1 jerk flags are in ep 43–47, but all of handover1 has 1.2–5x the jerk of
  handover2 (see the 2026-10-20 report). Under the current rules a different background or style is no reason to drop, so
  no episode was dropped for jerk.

## Round 2: rank by quality, drop heavy jitter and pauses (user request, 2026-10-23)
Quality is the existing `report.py` score (smooth 0.4 + speed 0.3 + no idle 0.3, against the human median of
`v4-handover2-trimmed`), computed on each episode after trimming the idle head and tail; see the CSV columns `quality`,
`q_*`, `jitter_x`. Also dropped:
- heavy jitter: small back-and-forth reversals >= 2x the human median (smooth score <= 0.5 in `report.py`): h2 24 41, h1 38 41 48;
- pauses: standing still >= 0.5 s during the task (`score.py` idle), or a longest pause > 1 s (dataset quality check).

The two kinds of "jitter" differ: handover1 was called worse before because of action jerk (3rd difference: fast, abrupt
motion), 1.2–5x handover2's; the back-and-forth jitter used here has a handover1 median of only 1.27x the human median. So
under this round's rule 14 handover1 episodes stay.

Result: 25 handover2 + 14 handover1 = 39 human demos, plus the 4 eval trials picked by `select_strict.py` = **43 episodes**.
Of the eval regrasp trials, only 3 recovery segments have jitter under 2x (h24 ep6, h32k4 ep8, h64 ep8); on the sheets all
drop the tape from height or hand over messily, so none is used.

Also: `v4-react_h32_k8_s4` ep8 is labelled success, but on the sheets L hovers open over the tape for 19 s and never grasps;
the label is wrong (the auto stages' 1/5 is right).

## New dataset `wwqcleverghost/handover-curated` (43 eps, 12-dim, local only, not uploaded)
Spec: `examples/xlerobot/handover_curated.json`, built with `--arm_only` (action/state keep only the 12 arm joints, like the
training input and the eval recordings).
Build: `PYTHONPATH=src python examples/xlerobot/trim_stagnant_frames.py --arm_only --spec examples/xlerobot/handover_curated.json --dst wwqcleverghost/handover-curated`.

## Earlier version (superseded, never finished building)
Round 1 was 72 episodes = `handover_perfect.json` minus h2 14, 30 and h1 18, still with the > 1 s pauses h2 1, 25, h1 29.

## Round 3 = v6 (user's final request, 2026-10-23)
- Combined filter: checklist keep and no very low item: any `report.py` sub-score (smooth / speed / no idle) <= 0.5
  (= 2x worse than human), or median action jerk >= 2x handover2's. Drops h2 24 41 (jitter) and h1 24 26 27 28 32 33 34
  35 37 38 40 41 42 43 44 48 (jerk).
- Pauses: no longer drop the episode; cut only the part of a mid-episode pause > 1 s where joints, all three cameras and the
  command are still, so the join does not jump: h2 1 frames 625–640 (0.5 s), h1 29 frames 139–189 (1.67 s); joint jump at
  the join <= 0.08, image difference same as normal neighbouring frames. h2 25's "pause" is only the command standing while
  the robot still catches up, not a real pause; not cut.
- Result `wwqcleverghost/v6-handover-curated`: 41 handover2 + 13 handover1 + 4 eval = 58 episodes, 12-dim (`--arm_only`),
  local and on server6 only.
- `v4-react_h32_k8_s4` ep8 human label changed to 0 (local `success.csv`, `eval-scores-v4-v5` per_episode/summary/README; not pushed to HF).
- Training: server6 GPU0, `queue_v3.sh flow:v6` → `v6_flow_h16_s8`, 140K steps, save every 20K, otherwise the baseline recipe.
