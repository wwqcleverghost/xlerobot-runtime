# Eval grid video

中文版：[README.zh.md](README.zh.md)

Eval datasets live on HF as `wwqcleverghost/eval_<model>_<YYYYMMDD_HHMMSS>` (private). Each has 10 episodes,
`success.csv` (episode, success, policy), `run.log` and sometimes the evaluator's own `trials_grid.mp4`
(head camera only).

1. Download without the old grid: `snapshot_download(repo, repo_type="dataset", local_dir=<dir>, ignore_patterns=["trials_grid.mp4"])`.
   Skip runs with 1-2 episodes (aborted) unless asked; check `run.log` for the sampler
   (`noise_scheduler_type`, `num_inference_steps`, `react_blocks`) to know which group a run belongs to.
2. `python docs/workflows/eval-grid-video/make_grid.py <dir> <out.mp4>` (lerobot env; ~1.5 min per run, run
   several in parallel). It handles datasets split over several parquet/mp4 files.
3. Put deliverables in `outputs/eval_grids_<date>/<model>.mp4` and send them with SendUserFile.

Label meaning: `arm` = the arm whose joints first move > 15 deg (usually the grasping arm), `tape x` = yellow
centroid in the first head frame (0 left, 1 right; `?` when not visible). Demos use L for x <= 0.49 and R for
x >= 0.51. Timer bottom right; a finished trial freezes and shows "end".
