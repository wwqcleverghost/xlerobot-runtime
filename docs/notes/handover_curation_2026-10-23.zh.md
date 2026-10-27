# 交接数据集逐 episode 检查（2026-10-23）

English: [handover_curation_2026-10-23.md](handover_curation_2026-10-23.md)

v4 = `v4-handover2-trimmed`（handover2 全部 50 条），v5 = `handover-perfect`（75），strict = `handover-perfect-strict`（64）。
三者都只是 `handover2` 和 `handover1` 原始 episode 的子集，所以对这 100 条原始 episode 逐条检查，再对应回三个数据集。
逐条结果：[handover_grades_2026-10-23.csv](handover_grades_2026-10-23.csv)（`verdict` 列是结论）。

来源：git `9a881cac`（工作区有未提交改动）；HF `wwqcleverghost/handover2` @ `46c0695d`，`wwqcleverghost/handover1` @ `5b057e25`
（本地缓存，按当时 main 的 sha 记录）。

## 方法
按 `docs/workflows/handover-dataset-curation/README.zh.md` 的逐 episode 检查和质量检测，没有新加规则。
`docs/workflows/handover-dataset-curation/ep_score.py` 把清单自动化：裁剪范围（`keep_range`）、最后一帧头部画面的黄色像素、
开始时胶带色块的高宽比（立着）、每只手臂夹爪闭合次数、先抓的手臂和中线规则、时长、最长停顿，以及
`lerobot_dataset_quality.py` 的 IQR 标记。闭合次数超过 2 的 episode 用 `sheets.py` 截图逐个看是哪次闭合真正抓住。

## 结论
**v4 里放错位置的 episode 没有混进 v5。** handover2 ep 3 4 5 32（胶带放进篮子）、26（录到松手就停）最后一帧都没有胶带；
v5 和 strict 都没收录它们，v4 里这 5 条都在。中线规则 100 条全部符合。

v5 里还有 3 条重抓，之前没发现：

| episode | 在哪些数据集 | 问题 |
|---|---|---|
| h2 14 | v4, v5 | 交接时 L 第一次合爪没夹到，重新抓；之后从高处松手，胶带立着落下 |
| h2 30 | v4, v5, strict | 交接时 R 第一次合爪没夹到，张开再抓 |
| h1 18 | v5 | L 抓取时来回摸索约 6 秒（合爪 6 次），交接也慢 |

| 数据集 | ep | 按清单应删 | 停顿 > 1 秒（清单要求看） |
|---|---|---|---|
| v4 | 50 | 7（h2 3 4 5 14 26 30 32） | h2 1 25 |
| v5 | 75 | 3（h2 14 30，h1 18） | h2 1 25，h1 29 |
| strict | 64 | 1（h2 30） | h2 1 |

- 闭合次数超过 2、但多出来的那次是空合的，保留：h2 12、40（接手后/放下后 R 空合一次），h1 24（L 伸过去之前空合）。
- 可剪的只有 handover1 的重抓 episode 2 4 10 16 21 45（剪掉失败的第一次抓取），按已有决定只放进 `-recovery` 数据集。
- IQR 标记（只是"去看看"）：handover1 的 jerk 标记集中在 ep 43–47，但整个 handover1 的 jerk 是 handover2 的 1.2–5 倍
  （见 2026-10-20 报告）。按现有规则，背景不同、风格不同都不是删除理由，所以没有因为 jerk 删 episode。

## 第二轮：按质量分排名，去掉严重抖动和停顿（用户 2026-10-23 要求）
质量分用现有的 `report.py`（平滑 0.4 + 速度 0.3 + 不停顿 0.3，对比 `v4-handover2-trimmed` 的人类中位数），算在裁掉首尾空闲后的
每条 episode 上，结果在 CSV 的 `quality`、`q_*`、`jitter_x` 列。另外删：
- 严重抖动：小幅来回次数 >= 人类中位数的 2 倍（`report.py` 里平滑分 <= 0.5）：h2 24 41，h1 38 41 48；
- 停顿：任务中有 >= 0.5 秒站着不动（`score.py` 的 idle），或最长停顿 > 1 秒（数据集质量检测）。

注意两种"抖"不是一回事：之前说 handover1 质量差，指的是动作 jerk（三阶差分，反映动作快、急），handover1 是 handover2 的
1.2–5 倍；而这里的抖动（小幅来回）handover1 中位数只有人类的 1.27 倍。所以按这一轮的规则，handover1 有 14 条留下。

结果：handover2 25 条 + handover1 14 条 = 39 条人类示范，加上 `select_strict.py` 选出的 4 个评测试次 = **43 条**。
评测的重抓试次里，抖动低于 2 倍的恢复片段只有 3 段（h24 ep6、h32k4 ep8、h64 ep8），看截图都是从高处丢下胶带或交接混乱，不收。

另外：`v4-react_h32_k8_s4` ep8 的人工标签是成功，但截图里 L 张着爪在胶带上方停了 19 秒、没抓，标签错了（自动阶段判 1/5 是对的）。

## 新数据集 `wwqcleverghost/handover-curated`（43 eps，12 维，只在本地，未上传）
spec：`examples/xlerobot/handover_curated.json`，构建加 `--arm_only`（动作/状态只留 12 个手臂关节，和训练输入、评测数据一致）。
构建：`PYTHONPATH=src python examples/xlerobot/trim_stagnant_frames.py --arm_only --spec examples/xlerobot/handover_curated.json --dst wwqcleverghost/handover-curated`。

## 旧版（已作废，未构建完）
第一轮是 72 条 = `handover_perfect.json` 再去掉 h2 14、30 和 h1 18；停顿超过 1 秒的 h2 1、25 和 h1 29 当时还保留。

## 第三轮 = v6（用户 2026-10-23 最终要求）
- 综合筛选：清单 keep，且没有任何极低项：`report.py` 平滑/速度/不停顿任一子分 <= 0.5（= 比人类差 2 倍），或动作 jerk
  中位数 >= handover2 的 2 倍。去掉 h2 24 41（抖动）和 h1 24 26 27 28 32 33 34 35 37 38 40 41 42 43 44 48（jerk）。
- 停顿：不再整条删除，只把中间超过 1 秒、且关节、三路画面、指令都静止的部分剪掉，拼接处不跳：
  h2 1 剪帧 625–640（0.5 秒），h1 29 剪帧 139–189（1.67 秒）；拼接处关节跳 <= 0.08，画面差异和正常相邻帧相同。
  h2 25 的"停顿"只是指令不动、机器人还在慢慢追，不是真停顿，不剪。
- 结果 `wwqcleverghost/v6-handover-curated`：handover2 41 + handover1 13 + 评测 4 = 58 条，12 维（`--arm_only`），只在本地和 server6。
- `v4-react_h32_k8_s4` ep8 的人工标签已改成 0（本地 `success.csv`、`eval-scores-v4-v5` 的 per_episode/summary/README，未上传 HF）。
- 训练：server6 GPU0，`queue_v3.sh flow:v6` → `v6_flow_h16_s8`，140K 步，每 20K 存，其余同基线配方。
