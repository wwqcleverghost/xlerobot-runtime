# v7 数据集筛选（2026-10-27）

English: [v7_selection_2026-10-27.md](v7_selection_2026-10-27.md)

`wwqcleverghost/v7-handover-clean`：53 条（handover2 38 + handover1 15），17 维，只在本地和 server1（`/media/data/wwq/data/v7-handover-clean`），未上传 HF。
spec：`examples/xlerobot/v7_handover_clean.json`；逐条结果：[v7_selection_2026-10-27.csv](v7_selection_2026-10-27.csv)（`v7` 列）。
构建：git `d0c28aca`（工作区有未提交改动），来源 `wwqcleverghost/v2-handover2`、`wwqcleverghost/v1-handover1`（本地缓存）。

```bash
PYTHONPATH=src python examples/xlerobot/trim_stagnant_frames.py --spec examples/xlerobot/v7_handover_clean.json --dst wwqcleverghost/v7-handover-clean --reset_tol 10
```

用户要求（不再人工审阅）：去掉开头胶带立着的、最后放进框里的（和 v6 一样），质量按当前规则，约 50 条，太少就放宽。

1. 裁剪规则"主臂起止"：头 = 主臂开始有输入，尾 = 复位基本完成（`--reset_tol 10`）。直接从原始录制剪，不从已剪过的版本再剪（避免视频二次编码）。
2. 清单（沿用 v6 人工确认的结果，[handover_grades_2026-10-23.csv](handover_grades_2026-10-23.csv) 的 `verdict`）：
   handover2 去掉 3 4 5 32（放进框里）、26（录到松手就停）、14 30（重抓）；
   handover1 去掉 3 6 7 8 15 23 30 36 39 46 47 49（开头立着）、13 31 18（重抓混乱）、2 4 10 16 21 45（重抓，只进 recovery）。
3. 用户在网页上手动标的"去掉"（`examples/xlerobot/dataset_labels.json`）：handover1 7 13 15 17 24 39 46 48 49。
4. 质量：`curate.py` 默认预设（突变、停顿 > 1 秒、小幅来回 >= 2 倍、从臂 jerk >= 2 倍）只剩 43 条，所以放宽两条：
   突变从"1 次就去掉"改成"2 次以上"（`snap.min=2`），从臂 jerk 从 2 倍改成 2.5 倍（`jerk.x=2.5`）。

| 设置 | 剩下 |
|---|---|
| 默认预设 + 清单 + 手动 | 43 |
| jerk 2.5 倍 | 48 |
| 突变 >= 2 次 | 46 |
| **突变 >= 2 次 + jerk 2.5 倍（采用）** | **53** |
| 突变 >= 3 次 + jerk 2.5 倍 | 56 |

保留：handover2 0 2 6 7 8 9 10 11 12 13 15 16 17 18 19 20 21 22 23 27 28 29 31 33 34 36 37 38 39 40 42 43 44 45 46 47 48 49；
handover1 0 1 5 9 11 12 14 19 20 25 27 34 37 42 44。
