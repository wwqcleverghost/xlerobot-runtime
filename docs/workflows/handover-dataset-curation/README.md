# Handover dataset curation

中文版：[README.zh.md](README.zh.md)

Tool: `examples/xlerobot/trim_stagnant_frames.py`. Single source: `--src <repo> --dst <repo>`. Merge/filter:
`--spec examples/xlerobot/<spec>.json --dst wwqcleverghost/<name>`, spec = list of
`{"repo", "note", "drop": [episodes], "start_s": {"<ep>": seconds after the trimmed start}}`. It re-encodes
videos (~30 min for 75 episodes, laptop lerobot env with `PYTHONPATH=src`); run it detached (`setsid -f`),
the laptop has rebooted mid-build before. Existing specs: `handover_perfect.json` (75 eps),
`handover_perfect_strict.json` (64), `handover_perfect_recovery.json` (81).

Trim rule (the user's): cut only the idle head and tail, keep pauses inside episodes. Task start is judged by
the arm command leaving the start pose by > 2 deg, ignoring the frame 0-1 recording jitter and head servos.

Audit before building, per episode (raw data + `keep_range`):
- final placement: head-camera yellow pixels at the last frame (0 = not on the visible table, e.g. put in the
  basket); look at release-moment sheets to confirm
- tape standing at the start (narrow/tall or face-on blob) — drop; standing at the END is fine
- gripper open/close cycles per arm (2/2 is clean; more = regrasp), which close actually holds (state stays
  above the command), grasping arm vs the midline rule (L x<=0.49, R x>=0.51)
- duration, longest mid-episode still stretch (> 1 s pauses teach the policy to wait)
- recording cut short (episode ends right at the release)

User decisions to keep: different backgrounds (handover1) are fine; regrasp/recovery data only without the
failed attempt (start the episode just before the gripper reopens) and only in a separate dataset; dataset
improvements are tested only as baseline (flow_h16_s8) ablations, not in the main comparison.

## Quality check (run on every new or rebuilt dataset)

Standard adapted from [huggingface/lerobot#3760](https://github.com/huggingface/lerobot/issues/3760):
`python src/lerobot/scripts/lerobot_dataset_quality.py <dataset_root> [<dataset_root> ...]` (reads only the parquet
`action` column, fast). Per episode, on the 12 arm dims only:
- `median_jerk`, `p95_jerk` — true 3rd difference × fps³ (how shaky the motion is)
- `max_velocity` — peak 1st difference × fps
- `static_fraction` — share of frames with arm speed < 3 units/s; `max_pause_s` — longest still stretch
- `duration_s`, `final_dist` — final pose distance from the dataset mean

Flags are relative, not pass/fail: a value is flagged when outside the dataset's own IQR range (k=1.5). A flag means
"look at this episode", not "drop it"; decide drops with the audit above. Compare datasets with the summary table
(the static threshold is fixed, so numbers are comparable across datasets). Example report:
`docs/notes/dataset_quality_2026-10-20.md`.

Verify a build: episode/frame counts from the log, spot-check a few episodes with LeRobotDataset; when copying
to a server rsync to `<name>.tmp/` then `mv` (meta/ arrives before videos/, a waiting job could start early).
