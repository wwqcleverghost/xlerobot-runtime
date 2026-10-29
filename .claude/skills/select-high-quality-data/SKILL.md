---
name: select-high-quality-data
description: Pick the high-quality episodes of XLeRobot handover datasets (drop snaps, pauses, jitter, high jerk, ...) with the dataset viewer's curate rules, report what each rule removed, and optionally write a trim spec / build the curated dataset. Use when the user asks to select, filter or curate good episodes or to make a cleaner training set.
---

# 选高质量数据（curate）

网页和命令行用的是同一套代码：`docs/workflows/dataset-viewer/curate.py`。用户自己在网页
`http://localhost:8770/#quality`（"训练集质量 / 筛选"页）就能做；这个 skill 是替用户快速跑一遍。

## 步骤

1. 确认要筛的数据集（例如 `raw-handover-20260929-trimmed raw-handover-20261001-trimmed`；名字是 `~/.cache/huggingface/lerobot/wwqcleverghost/` 下的目录名）。
   正在构建的数据集不能用（数据文件不完整）。
2. 用默认预设 `default` 跑（规则见下表），只读、不改任何文件：
   ```bash
   PYTHONPATH=src /home/wwq/miniconda3/envs/lerobot/bin/python docs/workflows/dataset-viewer/curate.py \
     --datasets raw-handover-20260929-trimmed raw-handover-20261001-trimmed --report outputs/curate/<date>_<name>.csv
   ```
   输出每个数据集按规则顺序累计还剩多少条、每条规则去掉了哪些 episode、手动强制的条，以及合计保留数。
   第一次算一个数据集要解码首尾画面，十几秒；之后读缓存。
3. 想看新指标或改阈值时用 `--set`（不改预设文件），例如 `--set small_snap.on=1 --set stall.on=1`、`--set jerk.x=2.5`。
   把"默认结果"和"加了哪些条件后的结果"都报给用户，数字原样给。
4. 交给用户：每条规则去掉了几条、最后剩多少；给出网页链接 `http://localhost:8770/#quality?ds=<id1>,<id2>`，
   让用户在网页上点开被去掉的条看视频、需要时强制保留 / 去掉（存在 `examples/xlerobot/dataset_labels.json` 的 `force`）。
   "开头胶带立着"和"放进框里"由用户在网页训练集质量页的"起止画面"里人工标（`dataset_labels.json` 的 `manual`：standing / basket 去掉这条，ok = 看过）；
   curate.py 自动计入这些标记（累计里的"人工"一步），不要替用户标，也不要打开 standing / place 自动规则代替人工。
   服务没开就用 `.claude/launch.json` 的 `dataset-viewer` 或 `PYTHONPATH=src python docs/workflows/dataset-viewer/server.py` 启动。
5. 只有用户确认后才写 spec 或构建：
   ```bash
   ... curate.py --datasets ... --spec <spec 名> [--build <新数据集名> [--arm_only]]
   ```
   `--spec` 写 `examples/xlerobot/<spec 名>.json`（保留的条整条保留，其余进 `drop`），`--build` 在后台构建 `wwqcleverghost/<新数据集名>`，
   日志在 `outputs/dataset_viewer/builds/`。不要上传 HF、不要提交，除非用户要求。按项目规则记下 git commit、数据集名和 revision。

## 规则（默认预设 `presets/default.json`）

| 键 | 条件 | 默认 | 状态 |
|---|---|---|---|
| `snap` | 突变：从臂一帧跳 >20、指令该帧变化 <5（review_page.py） | 开，>=1 次 | 已有规则 |
| `pause` | 停顿：指令 <3 单位/秒持续 >= 1 秒（review_page.py） | 开 | 已有规则 |
| `rev` | 小幅来回抖动 >= 基准中位数 × 2（基准 v4-handover2-trimmed，1.467 次/秒） | 开 | 用户在 v6 精选时同意过 |
| `jerk` | 从臂 jerk >= 基准中位数 × 2（基准 v4-handover2-trimmed，19941） | 开 | 用户同意看从臂；基准用裁剪后的 handover2 |
| `small_snap` | 从臂小卡顿：指令几乎不动、从臂一帧跳 >6 的帧数 >= 15 | 关 | **新指标，待用户确认** |
| `stall` | 从臂卡住：指令在动、从臂连续 >=4 帧不动的次数 >= 8 | 关 | **新指标，待用户确认** |
| `place` / `regrasp` / `midline` | 清单：末帧胶带 >= 500 像素 / 每只手闭合 <= 2 次 / 中线规则 | 关 | 已有规则，可选 |
| `standing` | 清单：开头胶带立着（色块高/宽 >= 1.2） | 关 | **阈值待用户确认** |

v6-handover1/2-trimmed 上的参考数（2026-10-27）：去掉突变和停顿后 71 / 100，再加抖动和 jerk 后 61，再开小卡顿和卡住后 56。

## 不要自己做的事

- 不改规则定义和默认值（`curate.py` 的 `RULES`、`presets/default.json`）；新指标开成默认、改阈值、换基准都要先问用户，
  规则以 git 里的 `docs/` 为准（AGENTS.md）。临时试验用 `--set`。
- 不覆盖已有 spec（同名会报错）；构建目标已存在时不要加 `--overwrite`，除非用户明确要求（旧的会改名为 `.old_<时间>`，不删除）。
