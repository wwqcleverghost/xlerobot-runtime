# Takeover corrections dataset (2026-10-23)

中文版：[takeover_corrections_2026-10-23.zh.md](takeover_corrections_2026-10-23.zh.md)

Built and uploaded `wwqcleverghost/v6-corrections` (private, revision `af487d8b48e142f92e187d6f48689384ea10bf91`):
20 episodes, 2179 frames (73 s), 12-D arm state/action (`--arm_only`, like the v6 policies). 15 human-takeover corrections
from 4 real-robot eval sets plus 5 regrasp recoveries cut out of the original handover1 demos.

| source | HF revision | kind | spans looked at | kept |
|---|---|---|---|---|
| `v4-react_h32_k8_s4_takeover/set1` | `4ce05238` | takeover, old code (switch jumps) | 11 | 8 |
| `v4-react_h32_k8_s4_takeover/set2` | `4ce05238` | takeover, old code | 10 | 3 |
| `v4-react_h24_k6_s4_takeover` (raw labels) | `f58938aa` | takeover, eased switches | 9 | 2 (merged into 1 segment) |
| `v4-react_h48_k6_s8_takeover` | `d5e67c07` | takeover, eased switches | 9 | 3 |
| `handover1` | `5b057e25` | regrasp after a failed grasp in a demo | 8 | 5 |

Build: git `95d09d21` + uncommitted `trim_stagnant_frames.py` (`root`/`segments` spec keys),
`PYTHONPATH=src python examples/xlerobot/trim_stagnant_frames.py --arm_only --spec examples/xlerobot/v6_corrections.json --dst wwqcleverghost/v6-corrections`.
Per-span takeover metrics: [takeover_review_2026-10-23.csv](takeover_review_2026-10-23.csv). Human labels were not
trusted; every span was judged from proprioception (score.py hold rule) plus frames around it.

## Cut rule
- Takeover: from the takeover (old-code sets: skip the first 0.5 s jump) to the open gripper positioned over the tape
  (align), or to the hold start + 0.2 s when the human closed the gripper (grasp). What the policy did afterwards is not
  included; it continued correctly after every kept span.
- Regrasp in demos (handover1 ep2, 4, 10, 16, 21): from 0.2 s before the gripper reopens after the failed grasp to the
  regrasp hold + 0.2 s. The failed attempt itself is not included. Not used: handover1 ep13/31 (messy, several
  regrasps), handover2 ep14; dropped after the quality check: handover1 ep45 (median jerk 23 966, above the outlier
  bound of v6-handover-curated).

## Dropped takeover spans (23 of 39; the 16 kept spans give 15 segments)
- Accidental presses (no human motion; the policy grasped by itself): 15.
- Human mistakes: tape dropped during a human-driven handover (h32s2 ep2 2nd, ep5 2nd; h48 ep7 2nd); h24 ep3 2nd
  (grasped then released back onto the table).
- Messy regrasps during the handover (h32s1 ep5 2nd, ep7 2nd); human finishing and resetting at the end of the episode
  (h32s1 ep7 3rd, h32s2 ep0 2nd).

## Switch quality (jerk in a 1.2 s window, × human demo p99 = 128 748 units/s³)
| sets | takeover | hand-back |
|---|---|---|
| h32 set1/set2 (old code) | 0.46–1.47 | 0.75–1.26 |
| h24, h48 (eased) | 0.02–0.63 | 0.02–0.49 |

## Quality check (`lerobot_dataset_quality.py`, medians over episodes)
```
dataset                        n flag    dur  medJerk  p95Jerk  maxVel  static maxPause
v6-corrections                20    4    3.5     5951    19942     124   15.7%      0.2
v6-handover-curated           58   18   19.1    12681    44176     196    6.3%      0.3
```
Smoother than the demos. Flags within v6-corrections (relative, IQR k=1.5): output ep0 (static 45%, slow precise
grasp, max pause 0.7 s), ep16 (handover1 ep4 regrasp, p95 jerk 58 969, still inside the v6-handover-curated range) — kept.

## Training: fine-tune of the v6 baseline
`queue_v3.sh flow:v6ft` on server6 → `v6ft_corrections_flow_h16_s8`: starts from
`v6_flow_h16_s8/checkpoints/last/pretrained_model` (140K steps on `v6-handover-curated`) after that run has finished,
8 000 steps, batch 24, lr 2e-5 (cosine, 500 warmup), EMA 0.9999, `--policy.do_mask_loss_for_padding=true` (short
segments otherwise teach "stop" through padded actions), checkpoint every 2 000 steps. Data dir
`~/wwq/data/v6-corrections-v6stats` = `v6-corrections` with `v6-handover-curated`'s `meta/stats.json`, so the
normalization stays the base model's. Config checked on server6 (CPU parse, dataset loads); not launched yet.
Risk: 8 000 × 24 samples ≈ 88 passes over 2 179 frames, so compare the 2K/4K/6K/8K checkpoints against the base on the
robot (forgetting of the full task). Alternative if it forgets: retrain from scratch on v6-handover-curated + these
segments.

## Result: offline loss after the fine-tune (2026-10-23)
Ran 18:22–19:02 on server6. Flow loss on 480 fixed frames per dataset (same noise per batch for every checkpoint, eval
crop, padded actions masked for all; ± = standard error over 20 batches):
```
checkpoint     v6-handover-curated        v6-corrections
base 140K          0.0081 ± 0.0022       0.1385 ± 0.0099
ft 2K              0.0291 ± 0.0030       0.0081 ± 0.0017
ft 4K              0.0422 ± 0.0040       0.0064 ± 0.0015
ft 6K              0.0513 ± 0.0051       0.0058 ± 0.0013
ft 8K              0.0556 ± 0.0056       0.0056 ± 0.0013
```
The corrections are learned by 2K (0.14 → 0.008), but the loss on the base training data rises 3.6x at 2K and 7x at 8K:
the fine-tune forgets the full task. Next: fine-tune again with the base data mixed in (rehearsal), or retrain mixed.
