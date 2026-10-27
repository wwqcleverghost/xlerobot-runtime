# 人工接管纠正数据集（2026-10-23）

English: [takeover_corrections_2026-10-23.md](takeover_corrections_2026-10-23.md)

生成并上传了 `wwqcleverghost/v6-corrections`（私有，revision `af487d8b48e142f92e187d6f48689384ea10bf91`）：
20 个回合，2179 帧（73 秒），12 维手臂状态/动作（`--arm_only`，和 v6 模型一致）。其中 15 段来自 4 组实机评测里的人工接管纠正，
5 段是从原始示教 handover1 里切出来的"抓失败后重新抓"。

| 来源 | HF revision | 类型 | 看过的段 | 保留 |
|---|---|---|---|---|
| `v4-react_h32_k8_s4_takeover/set1` | `4ce05238` | 接管，旧代码（切换会跳） | 11 | 8 |
| `v4-react_h32_k8_s4_takeover/set2` | `4ce05238` | 接管，旧代码 | 10 | 3 |
| `v4-react_h24_k6_s4_takeover`（原始标注） | `f58938aa` | 接管，切换平滑 | 9 | 2（合成 1 段） |
| `v4-react_h48_k6_s8_takeover` | `d5e67c07` | 接管，切换平滑 | 9 | 3 |
| `handover1` | `5b057e25` | 示教里抓失败后重抓 | 8 | 5 |

生成方式：git `95d09d21` + 未提交的 `trim_stagnant_frames.py`（新增 `root`/`segments` 配置项），
`PYTHONPATH=src python examples/xlerobot/trim_stagnant_frames.py --arm_only --spec examples/xlerobot/v6_corrections.json --dst wwqcleverghost/v6-corrections`。
接管段的指标：[takeover_review_2026-10-23.csv](takeover_review_2026-10-23.csv)。人工标注不作为依据；每一段都按关节数据
（score.py 的"夹住"规则）判断，再看前后的画面。

## 截取规则
- 接管：从接管开始（旧代码那两组跳过开头 0.5 秒的跳变），到人把张开的夹爪对准胶带上方（对准型），或人合爪时到夹住开始
  + 0.2 秒（夹取型）。之后模型的动作不放进来；每段保留的纠正之后模型都接得上。
- 示教里的重抓（handover1 ep2、4、10、16、21）：从抓失败后夹爪重新张开前 0.2 秒，到重新夹住 + 0.2 秒。失败那一下不放进来。
  没用：handover1 ep13/31（反复重抓好几次，比较乱）、handover2 ep14；质量检测后去掉：handover1 ep45（中位 jerk 23 966，
  超过 v6-handover-curated 的离群上限）。

## 去掉的接管段（39 段里去掉 23 段；保留的 16 段合成 15 个片段）
- 误按（人没有动作，之后模型自己抓起）：15 段。
- 人工失误：人工交接时胶带掉落（h32s2 ep2 第 2 段、ep5 第 2 段；h48 ep7 第 2 段）；h24 ep3 第 2 段（抓起后又松手落回桌面）。
- 交接时反复开合，比较乱（h32s1 ep5 第 2 段、ep7 第 2 段）；回合末尾人自己放下并复位（h32s1 ep7 第 3 段、h32s2 ep0 第 2 段）。

## 切换质量（1.2 秒窗口内的 jerk，× 人工示教 p99 = 128 748 单位/秒³）
| 组 | 接管时 | 交还时 |
|---|---|---|
| h32 set1/set2（旧代码） | 0.46–1.47 | 0.75–1.26 |
| h24、h48（平滑切换） | 0.02–0.63 | 0.02–0.49 |

## 质量检测（`lerobot_dataset_quality.py`，各回合的中位数）
```
dataset                        n flag    dur  medJerk  p95Jerk  maxVel  static maxPause
v6-corrections                20    4    3.5     5951    19942     124   15.7%      0.2
v6-handover-curated           58   18   19.1    12681    44176     196    6.3%      0.3
```
比人工示教更平顺。数据集内部的相对标记（IQR k=1.5）：输出 ep0（静止 45%，慢慢精细合爪，最长停顿 0.7 秒）、ep16（handover1 ep4
的重抓，p95 jerk 58 969，仍在 v6-handover-curated 的范围内）——都保留。

## 训练：在 v6 基线上微调
server6 上 `queue_v3.sh flow:v6ft` → `v6ft_corrections_flow_h16_s8`：等 `v6_flow_h16_s8`（在 `v6-handover-curated` 上训
140K 步）训完后，从它的 `checkpoints/last/pretrained_model` 开始，8 000 步，batch 24，lr 2e-5（cosine，500 步预热），
EMA 0.9999，`--policy.do_mask_loss_for_padding=true`（否则短片段会通过补齐的动作教模型"停下"），每 2 000 步存一次。
数据目录 `~/wwq/data/v6-corrections-v6stats` = `v6-corrections` 换上 `v6-handover-curated` 的 `meta/stats.json`，
这样归一化和基线模型一致。配置已在 server6 上检查过（CPU 解析、数据集能加载）；还没启动。
风险：8 000 × 24 个样本 ≈ 把 2 179 帧过 88 遍，所以要在实机上把 2K/4K/6K/8K 存档和基线对比（看完整任务有没有忘）。
如果忘了：改成在 v6-handover-curated + 这些片段上从头训练。

## 结果：微调后的离线 loss（2026-10-23）
在 server6 上 18:22–19:02 训练完。每个数据集取 480 个固定帧算 flow loss（每个存档每批用同样的噪声，评估裁剪，补齐的动作都
不计入；± 为 20 个 batch 的标准误）：
```
checkpoint     v6-handover-curated        v6-corrections
base 140K          0.0081 ± 0.0022       0.1385 ± 0.0099
ft 2K              0.0291 ± 0.0030       0.0081 ± 0.0017
ft 4K              0.0422 ± 0.0040       0.0064 ± 0.0015
ft 6K              0.0513 ± 0.0051       0.0058 ± 0.0013
ft 8K              0.0556 ± 0.0056       0.0056 ± 0.0013
```
纠错片段 2K 步就学会了（0.14 → 0.008），但在基线训练数据上的 loss 2K 时涨到 3.6 倍、8K 时涨到 7 倍：微调把完整任务忘了。
下一步：带上基线数据再微调（复习式），或混合重训。
