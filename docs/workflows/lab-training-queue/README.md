# Lab training queue

中文版：[README.zh.md](README.zh.md)

Machines (shared `marmot` account, see memory `lab-training-servers`): server6 1x3090 (~/wwq), server1 4x3090
(/media/data/wwq), server4 5090 (~/wwq, suspected bad RAM: DataLoader workers segfault), server2 4x3090
(/media/data/wwq), lab computer b2-unit-366 3080 Ti 12 GB (~/wwq/lerobot git repo, conda env lerobot-wwq,
only reachable via a reverse tunnel it opens: `ssh -J server6 -p <2222|2223> marmotlab@localhost`).
Check `nvidia-smi` + process owners first; other lab members' jobs come and go.

Scripts in this directory (copy to the machine with rsync, never scp over a running copy):
- `queue_v3.sh` — runs Flow jobs one after another on one GPU: `flow`, `flow:strict`,
  `flow:sepenc`, `flow:noshift`; names start with `dp_` consistently on disk, HF and W&B (V5 data). Skips finished runs, resumes from
  the last checkpoint after a crash (5 tries). `WQ=/media/data/wwq GPU=<n> NUM_WORKERS=<n> setsid -f bash queue_v3.sh <jobs> >> logs/queue_v3_gpu<n>.log 2>&1 < /dev/null`
  `fb:<base>`, `fbstale:<base>`, `fbstate:<base>` train feedback denoising ("Catch Me If You Can",
  `--policy.feedback`) on top of the finished flow run `<base>`, on its dataset, as `<base>_fb` etc.: frozen
  planner, only a ~1M-parameter module trains, 40K steps, batch 12, 6 workers (9 frames per camera per sample,
  about 1.2 GB per batch). `fbstale` is the control without fresh frames, `fbstate` also sees the joint state.
  `fb:<base>@<step>` uses the checkpoint at `<step>` as the planner and starts as soon as it is saved
  (`<base>_fb_p<step/1000>k`). `flow:v7` = `v7-flow_h16_s8` on `v7-handover-clean`, 140K steps; when it finishes
  only the last three checkpoints are kept, all complete.
- `finish_v2.sh` — finishes `d2t_*_v2` REACT runs on `data/handover2-trimmed-v2` (V4, 50 episodes): resume
  with the canonical dataset ID and root, or restart from scratch.
- `prune_checkpoints.py <run-output-dir>` — keeps the last three model snapshots and only the latest full
  training state. Validates the published `last` checkpoint before deleting anything; ignores newer saves.
- Laptop `outputs/hf_upload/auto_upload.sh` — polls the RUNS list every 10 min and uploads each finished run's
  final EMA policy to a private HF repo `dp_*` and adds it to the Handover V1–V5 collection; restart it after a laptop reboot (`setsid -f`).
  Every poll applies the three-checkpoint retention rule, including the remaining old-data h64 run on server6;
  retained checkpoints remain on the server after upload.

Recipe: batch 24, 100K steps, save every 20K, EMA 0.9999, resize 240x320 crop 216x288, AMP, on-GPU image
transforms, `dataset.drop_names` drops head servos and base velocities, wandb entity wwqcleverghost project
`handover`, grouped by V1–V5. Resumed runs retain their original project until training finishes;
`outputs/hf_upload/move_wandb_hp.py` updates names/groups/tags and moves ended runs each poll.
Key from `<W>/secrets.env` (user writes it themselves with `read -rsp`; on the lab
computer `~/wwq/.wandb_api_key`). `PYTHONNOUSERSITE=1` (other users' packages in ~/.local).

Pitfalls learned the hard way:
- Stopping: kill the training's python main process. Killing a tmux pane's shell SIGHUPs the training too;
  `pgrep -f <pattern>` also matches your own ssh command — pick pids from `ps` output.
- `ssh host 'nohup cmd &'` can hang the ssh: use `setsid -f ... < /dev/null`.
- A crashed run's output dir blocks a fresh start (FileExistsError): resume, or `mv` it aside.
- Resume = `lerobot_train --resume=true --config_path=<out>/checkpoints/last/pretrained_model/train_config.json`;
  extra CLI args override the saved config (use it when a dataset moved).
- Speed: ~7 h/run alone on a 3090; 4 runs on server1 are CPU-bound (~13 h each).
- Latest 2026-10-17 instruction: keep all four Flow jobs (baseline, sepenc, noshift, strict), cancel and
  delete only the 75-episode REACT jobs and their queues. Keep the old 50-episode REACT training.
  Actual machine/GPU assignments are at the top of `outputs/train_plan/PLAN.md`.
