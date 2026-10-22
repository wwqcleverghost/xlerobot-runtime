# 评测网格视频

English: [README.md](README.md)

评测数据集在 HF 上，名为 `wwqcleverghost/eval_<model>_<YYYYMMDD_HHMMSS>`（私有）。每个有 10 个 episode、
`success.csv`（episode, success, policy）、`run.log`，有时还有评测脚本自己生成的 `trials_grid.mp4`（只有头部相机）。

1. 下载时跳过旧网格：`snapshot_download(repo, repo_type="dataset", local_dir=<dir>, ignore_patterns=["trials_grid.mp4"])`。
   只有 1–2 个 episode 的 run（中途中止）除非用户要求否则跳过；看 `run.log` 里的采样器参数
   （`noise_scheduler_type`、`num_inference_steps`、`react_blocks`）判断这个 run 属于哪一组。
2. `python docs/workflows/eval-grid-video/make_grid.py <dir> <out.mp4>`（lerobot 环境；每个 run 约 1.5 分钟，
   可多个并行）。数据分散在多个 parquet/mp4 文件里也能处理。
3. 成品放到 `outputs/eval_grids_<date>/<model>.mp4`，用 SendUserFile 发给用户。

标签含义：`arm` = 关节最先转动超过 15° 的那只手臂（通常是抓取臂）；`tape x` = 第一帧头部画面里黄色胶带的
质心横坐标（0 最左，1 最右；看不到时为 `?`）。示范数据中 x <= 0.49 用左臂 L，x >= 0.51 用右臂 R。
右下角是计时器；试次结束后画面定格并显示 "end"。
