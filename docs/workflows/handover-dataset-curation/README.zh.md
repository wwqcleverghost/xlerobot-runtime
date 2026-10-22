# 交接数据集整理

English: [README.md](README.md)

工具：`examples/xlerobot/trim_stagnant_frames.py`。单个数据源：`--src <repo> --dst <repo>`。合并/筛选：
`--spec examples/xlerobot/<spec>.json --dst wwqcleverghost/<name>`，spec 是
`{"repo", "note", "drop": [episodes], "start_s": {"<ep>": 相对裁剪后起点的秒数}}` 组成的列表。它会重新编码视频
（75 个 episode 约 30 分钟，笔记本 lerobot 环境，`PYTHONPATH=src`）；要脱离终端运行（`setsid -f`），
笔记本曾在构建中途重启过。已有 spec：`handover_perfect.json`（75 eps）、
`handover_perfect_strict.json`（64）、`handover_perfect_recovery.json`（81）。

裁剪规则（用户定的）：只裁掉开头和结尾的空闲帧，保留 episode 中间的停顿。任务起点判断标准是手臂指令偏离起始姿态
超过 2°，忽略第 0–1 帧的录制抖动和头部舵机。

构建前逐 episode 检查（原始数据 + `keep_range`）：
- 最终放置：最后一帧头部画面里的黄色像素数（0 = 不在可见桌面上，比如放进了篮子）；看松手时刻的截图表确认
- 开始时胶带是立着的（窄高或正面的色块）— 丢弃；结束时立着没关系
- 每只手臂夹爪开合次数（2/2 是干净的；更多 = 重新抓取）、哪次闭合真正抓住（状态保持在指令之上）、
  抓取臂是否符合中线规则（L x<=0.49，R x>=0.51）
- 时长、episode 中间最长的静止段（超过 1 秒的停顿会教策略原地等待）
- 录制被提前截断（episode 正好在松手时结束）

要遵守的用户决定：不同背景（handover1）可以用；重新抓取/恢复数据只保留去掉失败尝试的部分（episode 从夹爪重新张开前开始），
且只放在单独的数据集里；数据集改进只作为 baseline（flow_h16_s8）的消融实验测试，不进主对比。

## 质量检测（每个新建或重建的数据集都要跑）

标准改编自 [huggingface/lerobot#3760](https://github.com/huggingface/lerobot/issues/3760)：
`python src/lerobot/scripts/lerobot_dataset_quality.py <dataset_root> [<dataset_root> ...]`（只读 parquet 的
`action` 列，很快）。逐 episode 计算，只用 12 维手臂：
- `median_jerk`、`p95_jerk` — 真正的三阶差分 × fps³（动作有多抖）
- `max_velocity` — 一阶差分峰值 × fps
- `static_fraction` — 手臂速度 < 3 单位/s 的帧占比；`max_pause_s` — 最长连续静止秒数
- `duration_s`、`final_dist` — 末帧姿态离数据集均值的距离

标记是相对的，不是及格线：某个值超出该数据集自身的 IQR 范围（k=1.5）就会被标记。标记的意思是"去看看这个 episode"，
不是"删掉它"；删不删按上面的逐 episode 检查决定。比较不同数据集用汇总表（静止阈值是固定的，所以数字可以跨数据集比较）。
报告示例：`docs/notes/dataset_quality_2026-10-20.zh.md`。

验证构建结果：从日志看 episode/帧数，用 LeRobotDataset 抽查几个 episode；拷到服务器时先 rsync 到 `<name>.tmp/` 再 `mv`
（meta/ 比 videos/ 先到，等待中的任务可能提前启动）。
