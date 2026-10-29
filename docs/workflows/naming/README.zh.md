# 命名：训练集、模型、测试集

[English](README.md)

所有仓库都在 Hugging Face 的 `wwqcleverghost/` 下；本地副本用同样的名字
（`~/.cache/huggingface/lerobot/wwqcleverghost/<名字>`）。

| 类别 | 规则 | 例子 |
|---|---|---|
| 训练集（数据集） | `v<N>-<内容>` | `v7-handover-clean`、`v6-corrections`、`v6-handover-curated-corrx3` |
| 模型 | `v<N>-<方法>_<参数>[_变体]` | `v7-flow_h16_s8`、`v6-flow_h16_s8_corrmix_4k` |
| 测试集（真机评测录像，数据集） | `eval-<模型名>` | `eval-v6-flow_h16_s8`、`eval-v4-react_h32_k8_s4_takeover` |
| 分数表 | `scores-<内容>` | `scores-v4-v5` |

- 字段之间用连字符，字段内部用下划线。
- `v<N>` = 主训练数据的一次变化。同一份数据上的不同方法和消融共用这个编号。
- `eval-` 开头 → 测试集；否则带方法和参数（`_h16_s8` 等）→ 模型；其余 → 训练集。
- 测试集：有人工接管的评测加 `_takeover`；同一模型重复跑的放在仓库里的 `set1/`、`set2/`。
- 实验室电脑录制时本地仍叫 `eval_<模型>_<时间>`（给了 policy 时 lerobot 要求 `eval_` 前缀），上传时用 `eval-<模型>`。
- 中间检查点放进模型仓库的 `checkpoints/<step>/`（加 `train.log`），不单独建仓库。
- 记录用到的每个数据集的 revision（提交哈希），不要只写 `main`。

2026-10-30 改名：原来直接用模型名的 29 个测试集都加了 `eval-` 前缀；
`v1-eval_records_20261001` → `eval-v1-records_20261001`，`d2_train_records_20261006` → `eval-v2-records_20261006`，
`eval-scores-v4-v5` → `scores-v4-v5`。HF 旧链接会自动跳转。
