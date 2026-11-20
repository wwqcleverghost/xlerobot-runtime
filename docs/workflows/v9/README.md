# v9: learning from the robot's own rollouts with an honest progress

中文：[README.zh.md](README.zh.md)

**In short.** Almost every failure of the v6 models now ends the same way: the A arm hovers next to the tape with the gripper
open and never closes (A对位). Feeding the model its progress did not change that, because the actions ignore the progress input.
v9 keeps progress only as something the model **predicts**, gives every frame of the 216 real-robot eval trials an honest progress
(it stays flat while the robot hovers), and uses those trials to teach the model what *not* to do there. At inference the
model's own progress can pick among sampled chunks, and a stall guard can retreat and try again.

## Why (measured before v9)

1. **The actions ignore the progress input** (2026-11-16, intervention test): shifting the input s by ±2 stages moves the joint
   commands 0.1–0.6°, no more than another noise seed; the gripper does not close when s is pushed past A对位.
2. **A model stalls where it almost never samples a grasp** (2026-11-20, laptop): at frames where a failed trial hovers (joints
   still for 1 s, A gripper open), share of 32 sampled chunks that close the A gripper:

   | stuck frames of \ model | v6 baseline | progdelta | progself |
   |---|---|---|---|
   | baseline's failures (74 frames) | **0.027** | 0.000 | 0.000 |
   | progdelta's failures (80) | 0.383 | **0.090** | 0.059 |
   | progself's failures (100) | 0.098 | 0.135 | **0.004** |

   On its own stalls (diagonal) each model closes in 0–9% of its samples, while another model at the same frames often would.
   So resampling the same model helps little; what the model has learned at those states has to change.
3. **The model's progress dims move with its actions**: within one frame, samples whose own predicted progress rises more are the
   ones that close (correlation −0.28 to −0.44 with the gripper command); picking the best of 32 by it doubles the closing share
   (progdelta 0.09 → 0.175, progself 0.059 → 0.125 on progdelta's stalls). Cheap to use, but limited by point 2.

## Ideas → v9 pieces

| Idea (user) | Related work | v9 piece |
|---|---|---|
| Progress should be implicit, not an input | copycat / causal confusion (Wen et al. 2020) | progress = 9 extra action dims, predicted only (`--policy.progress_pred_dims=9`) |
| Better progress definition; flat when stuck | ReWiND (rewound videos), SARM, RedFlow (groups chunks by progress + proprioception) | honest progress for every rollout frame: the road progress of the nearest demo frames of the same stage |
| Re-label the eval trials (DAgger-like), then RA-BC | π\*0.6 RECAP (value = steps to go, failures very negative), SARM RA-BC | advantage per frame = progress gained within 1 s vs the demos' pace; `v9...rabc` = RA-BC on the honest progress |
| Learn the progress of failures without their actions | AFIL, CCDP (failures as negative guidance) | `v9...negmask`: joint loss masked on bad rollout frames, progress dims still learned |
| Progress sign should control the action | RECAP / CFGRL (advantage-conditioned policy + classifier-free guidance) | `v9...adv`: indicator good / bad as input, dropped 30%; sampling `v_none + w (v_good − v_none)` |
| A supervisor that notices the stall and goes back to re-align | Sentinel (task-progression failures), Rewind-IL (detect, respawn to a checkpoint), RaC (recover then correct) | `STALL=rewind:<s>`: after <s> s without motion or progress, replay the commands before the stall backwards, then replan |
| Resampling gets it through | V-GPS, RoboMonkey (sample several, pick with a verifier) | `--policy.best_of_n=N`: keep the chunk whose own predicted progress gains most |

Not done: a learned retreat (reversed approach segments with a "back" command). The scripted rewind tests the same idea first
without new data; if it helps on the robot, the learned version is the next step.

Masking has a limit: flow matching denoises joints and progress together, so a masked frame's noisy joints still reach the
network input and still shape its progress output. The mask removes the joint targets, not every influence of the failed actions.

## Training set `v9-handover-curated-rollouts`

[`examples/xlerobot/build_v9.py`](../../../examples/xlerobot/build_v9.py), run on server2 (steps in its docstring). Sources
(HF revisions in the set's `v9_sources.json`): the 58 `v6-handover-curated` demos and the 11 v6-family eval sets
(`eval-v6-flow_h16_s8`, `_progdelta`, `_progdelta_rabc`, `_progpred`, `_progself`, `_progself_s2head`, `_progself_s2mix35`,
`_rabc`, `_stagelabel`, `_stagelabel_stagehead`, `_stagepred`).

Per frame:

- **stage**: fine9 (hand-set starts for the demos). A failed trial whose note in `dataset_labels.json` says `<stage>失败` never goes
  past that stage (the rule sees A抓住 where the reviewer saw A对位失败).
- **p** (progress inside the stage): demos = the road measure (`stage_progress_cum.py`); rollouts = the mean road progress of the 10
  nearest demo frames of the same stage (joints / demo std, A arm first, 5-frame smoothed). A hovering arm keeps the progress of
  the demo frames it looks like: flat.
- **v** = stage + p (0..9). **gain** = best v within the next 30 frames − v.
- **adv** = 1 when gain ≥ the 30% quantile of the demos' gain in that stage (RECAP's threshold), else 0. **neg** = rollout frame
  with adv 0.

Kept rollout frames: from the start to the task end (`episode_end.csv`); a run of adv 0 longer than 150 frames keeps its first 150
and its last 15 (a failure's final stall only the first 150). Every 5th trial of each eval set (`episode % 5 == 4`, 54 trials) is
held out for the offline checks (labels in `v9_labels_holdout.npz`).

Files in the set besides the LeRobot data: `frame_labels.parquet` (index, episode, source, source_episode, demo, success, v, adv,
neg), `sarm_progress.parquet` (v / 9, for RA-BC), `episodes_demo.json` / `episodes_succ.json` (subsets), `v9_spec.json`,
`v9_summary.txt`, `v9_holdout.json`, `v9_labels_holdout.npz`, `v9_sources.json`. Action = 12 joints + 9 progress dims (identity
normalisation, as `-cumprog`); no `observation.environment_state`.

Numbers (2026-11-20, `v9_summary.txt`): 274 episodes (58 demos, 216 rollouts of which 127 succeeded), 124,773 frames, 32% neg.
Gain thresholds per stage (stages over 30 frames): 0.27, 0.12, 0.47, 0.32, 0.17, 0.67, 0.48, 0.37, 0.06. Rollouts are recorded at
the eval loop's real ~25 fps but stamped 30 fps, like every eval recording; the chunks the model learns from them therefore span the
same real time as the ones it executes.

## Models (queue jobs, `queue_v3.sh`)

All: FLOW h16 s8, 140K steps, batch 24, from scratch, progress dims predicted (`progress_pred_dims=9`).

| Job | Model | Data | What it tests |
|---|---|---|---|
| `flow:v9prog` | `v9-flow_h16_s8_prog` | demos + successful rollouts | filtered BC: do successful own rollouts help by themselves? |
| `flow:v9adv` | `v9-flow_h16_s8_adv` | all | advantage conditioning + guidance (RECAP / CFGRL) |
| `flow:v9negmask` | `v9-flow_h16_s8_negmask` | all | failures teach only progress (the user's masking idea) |
| `flow:v9rabc` | `v9-flow_h16_s8_rabc` | demos + successful rollouts | RA-BC with the honest progress (lookahead 90, κ 0.15) |
| `flow:v9advdemo` | `v9-flow_h16_s8_adv_demo` | demos only | advantage conditioning without rollouts: do the rollouts matter? |
| `flow:v9advr2` | `v9-flow_h16_s8_adv_r2` | all | `adv` with another seed: how much the offline numbers move by chance |

Reference: `v6-flow_h16_s8` (same demos, no rollouts, no progress).

## Inference options (real-robot eval, `lerobot_record_eval.py`)

- `--policy.adv_guidance=<w>` (adv models): 1 = conditioned on good; 1.5–3 pushes further from what bad chunks did.
- `--policy.best_of_n=<N>` (any model with progress dims, v6 progress models too): N samples per plan, keep the best by its own
  predicted progress gain.
- `STALL=resample:<s>` / `STALL=rewind:<s>` (environment variable, any model): see `StallGuard`. `STALL_BOX` (3°),
  `STALL_REWIND_S` (1.5 s), `STALL_MAX` (3) adjust it; triggers are listed in `stalls.csv` of the eval recording.
- v9 models print `[v9] progress` and log it in `stages.csv` (pred = stage, progress = v / 9).

## Offline checks before the robot

[`examples/xlerobot/offline_v9.py`](../../../examples/xlerobot/offline_v9.py) on the 54 held-out trials: share of sampled chunks that
close the A gripper at stuck frames (and best-of-N by own progress), progress error vs the label, and how well the predicted gain
separates frames that move on from those that do not (AUC). adv models are run at w = 1, 1.5, 2, 3.

## Real-robot eval plan

Robot evals are the only real test (30 trials each, same tape positions as far as possible). Order by the offline stuck-close share;
first the best v9 model at its best w against `v6-flow_h16_s8`, then with `STALL=rewind:4`.

## Results

Offline and robot results are added here as they come in.

## Related work

- π\*0.6 / RECAP: [arXiv 2511.14759](https://arxiv.org/abs/2511.14759) — CFGRL: [arXiv 2505.23458](https://arxiv.org/abs/2505.23458)
- AFIL (failure rollouts as negative guidance): [arXiv 2605.08434](https://arxiv.org/abs/2605.08434) — CCDP: [arXiv 2503.15386](https://arxiv.org/abs/2503.15386) — RedFlow: [arXiv 2607.27782](https://arxiv.org/abs/2607.27782)
- RaC (recovery + correction data): [arXiv 2509.07953](https://arxiv.org/abs/2509.07953) — Rewind-IL: [arXiv 2604.16683](https://arxiv.org/abs/2604.16683)
- V-GPS: [arXiv 2410.13816](https://arxiv.org/abs/2410.13816) — RoboMonkey: [arXiv 2506.17811](https://arxiv.org/abs/2506.17811)
- ReWiND: [arXiv 2505.10911](https://arxiv.org/abs/2505.10911) — SARM: [arXiv 2509.25358](https://arxiv.org/abs/2509.25358) — Sentinel: [arXiv 2410.04640](https://arxiv.org/abs/2410.04640)
- Copycat agents: [arXiv 2010.14876](https://arxiv.org/abs/2010.14876)
