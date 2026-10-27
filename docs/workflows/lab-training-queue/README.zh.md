# 实验室训练队列

English: [README.md](README.md)

机器（共用 `marmot` 账号，见记忆 `lab-training-servers`）：server6 1x3090（~/wwq），server1 4x3090
（/media/data/wwq），server4 5090（~/wwq，疑似内存坏：DataLoader worker 会段错误），server2 4x3090
（/media/data/wwq），实验室电脑 b2-unit-366 3080 Ti 12 GB（~/wwq/lerobot git 仓库，conda 环境 lerobot-wwq，
只能通过它自己开的反向隧道连接：`ssh -J server6 -p <2222|2223> marmotlab@localhost`）。
先看 `nvidia-smi` 和进程归属；实验室其他人的任务随时会来去。

本目录的脚本（用 rsync 拷到机器上，不要用 scp 覆盖正在运行的副本）：
- `queue_v3.sh` — 在一张 GPU 上依次跑 Flow 任务：`flow`、`flow:strict`、
  `flow:sepenc`、`flow:noshift`；名字在磁盘、HF、W&B 上统一以 `dp_` 开头（V5 数据）。跳过已完成的 run，崩溃后从
  最后一个 checkpoint 续训（最多 5 次）。`WQ=/media/data/wwq GPU=<n> NUM_WORKERS=<n> setsid -f bash queue_v3.sh <jobs> >> logs/queue_v3_gpu<n>.log 2>&1 < /dev/null`
  `fb:<base>`、`fbstale:<base>`、`fbstate:<base>` 在已完成的 flow run `<base>` 之上、用它的数据集训练反馈去噪
  （“Catch Me If You Can”，`--policy.feedback`），名字为 `<base>_fb` 等：planner 冻结，只训练约 1M 参数的小模块，
  40K 步，batch 12，6 个 worker（每个样本每个相机读 9 帧，每个 batch 约 1.2 GB）。`fbstale` 是不给新帧的对照组，
  `fbstate` 额外输入关节状态。
  `fb:<base>@<step>` 用 `<step>` 处的 checkpoint 当 planner，那个 checkpoint 一存好就启动（`<base>_fb_p<step/1000>k`）。
  `flow:v7` = 在 `v7-handover-clean` 上训练 `v7-flow_h16_s8`，140K 步；跑完只保留最后三个 checkpoint，三个都完整。
- `finish_v2.sh` — 在 `data/handover2-trimmed-v2`（V4，50 个 episode）上完成 `d2t_*_v2` REACT run：用规范的数据集 ID 和 root 续训，
  或从头重训。
- `prune_checkpoints.py <run-output-dir>` — 保留最后三个模型快照，只保留最新的完整训练状态。删除前先验证发布出去的
  `last` checkpoint；忽略更新的保存。
- 笔记本上的 `outputs/hf_upload/auto_upload.sh` — 每 10 分钟轮询 RUNS 列表，把每个完成的 run 的最终 EMA 策略上传到私有 HF 仓库 `dp_*`，
  并加入 Handover V1–V5 collection；笔记本重启后要重新启动它（`setsid -f`）。
  每次轮询都执行保留三个 checkpoint 的规则，包括 server6 上剩下的旧数据 h64 run；上传后保留的 checkpoint 仍留在服务器上。

训练配方：batch 24，100K 步，每 20K 保存，EMA 0.9999，resize 240x320 crop 216x288，AMP，图像增强在 GPU 上做，
`dataset.drop_names` 去掉头部舵机和底盘速度，wandb entity wwqcleverghost project
`handover`，按 V1–V5 分组。续训的 run 在训练结束前保留原来的 project；
`outputs/hf_upload/move_wandb_hp.py` 每次轮询更新名字/分组/标签并移动已结束的 run。
Key 来自 `<W>/secrets.env`（用户自己用 `read -rsp` 写入；实验室电脑上是 `~/wwq/.wandb_api_key`）。
`PYTHONNOUSERSITE=1`（~/.local 里有其他用户的包）。

踩过的坑：
- 停止训练：杀掉训练的 python 主进程。杀 tmux 窗格的 shell 会给训练发 SIGHUP 一起杀掉；
  `pgrep -f <pattern>` 也会匹配到你自己的 ssh 命令 — 从 `ps` 输出里挑 pid。
- `ssh host 'nohup cmd &'` 可能让 ssh 卡住：用 `setsid -f ... < /dev/null`。
- 崩溃 run 的输出目录会挡住新的启动（FileExistsError）：续训，或者把它 `mv` 走。
- 续训 = `lerobot_train --resume=true --config_path=<out>/checkpoints/last/pretrained_model/train_config.json`；
  额外的命令行参数会覆盖保存的配置（数据集换位置时用）。
- 速度：一张 3090 单独跑约 7 小时/run；server1 上同时跑 4 个会受 CPU 限制（每个约 13 小时）。
- 最新指令（2026-10-17）：保留全部四个 Flow 任务（baseline、sepenc、noshift、strict），只取消并删除 75 个 episode 的
  REACT 任务及其队列。保留旧的 50 个 episode 的 REACT 训练。实际机器/GPU 分配见 `outputs/train_plan/PLAN.md` 开头。
