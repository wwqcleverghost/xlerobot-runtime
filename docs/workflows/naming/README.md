# Naming: training sets, models, test sets

[中文](README.zh.md)

All repos live under `wwqcleverghost/` on Hugging Face; local copies use the same name
(`~/.cache/huggingface/lerobot/wwqcleverghost/<name>`).

| Kind | Pattern | Examples |
|---|---|---|
| Training set (dataset) | `v<N>-<content>` | `v7-handover-clean`, `v6-corrections`, `v6-handover-curated-corrx3` |
| Model | `v<N>-<method>_<params>[_variant]` | `v7-flow_h16_s8`, `v6-flow_h16_s8_corrmix_4k` |
| Test set (real-robot eval recordings, dataset) | `eval-<model name>` | `eval-v6-flow_h16_s8`, `eval-v4-react_h32_k8_s4_takeover` |
| Score tables | `scores-<what>` | `scores-v4-v5` |

- Hyphen between fields, underscore inside a field.
- `v<N>` = one change of the main training data. Methods and ablations trained on the same data share the number.
- Starts with `eval-` → test set. Otherwise, has method + params (`_h16_s8`, ...) → model. Otherwise → training set.
- Test sets: runs with human take-overs add `_takeover`; repeat runs of the same model go into `set1/`, `set2/` inside the repo.
- While recording, the lab PC still writes the local dataset as `eval_<model>_<time>` (lerobot requires the `eval_`
  prefix when a policy is given). Upload it as `eval-<model>`.
- Intermediate checkpoints go into the model repo under `checkpoints/<step>/` (+ `train.log`), not into separate repos.
- Record the revision (commit hash) of every dataset used, never just `main`.

Renamed on 2026-10-30: the 29 test sets that used the bare model name gained the `eval-` prefix;
`v1-eval_records_20261001` → `eval-v1-records_20261001`, `d2_train_records_20261006` → `eval-v2-records_20261006`,
`eval-scores-v4-v5` → `scores-v4-v5`. Old HF URLs redirect.
