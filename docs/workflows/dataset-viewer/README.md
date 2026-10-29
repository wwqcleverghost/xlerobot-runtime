# Dataset viewer

中文：[README.zh.md](README.zh.md)

One local web page, four pages under one shared top bar: **trim review**, **batch playback**, **training-set quality / curate**, **eval scores**.
Datasets are listed automatically from `~/.cache/huggingface/lerobot/wwqcleverghost/` (eval datasets included, also when nested in
`<dir>/dataset/` or `<dir>/<set>/dataset/`; symlinks under old names and meta-only directories without data files are skipped). A dataset with `success.csv` is an eval; one with human takeovers (`interventions.csv`, `*_takeover`) is DAgger training data, listed with the training sets and not in the eval scores. The top bar shows the dataset's HF revision (for local
builds: build time and git commit) and the git commit of the current code.

```bash
PYTHONPATH=src /home/wwq/miniconda3/envs/lerobot/bin/python docs/workflows/dataset-viewer/server.py   # then open http://localhost:8770
```

Port: `--port`, or the `PORT` environment variable (default 8770). Python standard library `http.server` only, no new dependency;
listens on 127.0.0.1 only. In Claude Code it can also be started with the `dataset-viewer` configuration of `.claude/launch.json`.

## Video

The page plays the dataset's own mp4 files (`videos/observation.images.<cam>/chunk-XXX/file-XXX.mp4`, h264 / yuv420p, which browsers play
directly) and seeks to each episode with its `from_timestamp` from `meta/episodes`; nothing is transcoded. The server supports HTTP Range
(raw recordings keep the moov at the end of the file) and answers at most 2 MB at a time, so dozens of players share the browser's
6 connections per host. Under every video sits a still decoded by the server (`outputs/dataset_viewer/cache/frames`), and the video only
shows once it has seeked, so loading, switching episodes and looping never flash black.

## 1. Trim review (`#review`)

Pick a spec (`examples/xlerobot/*.json`); "新建 spec…" (new spec) creates `examples/xlerobot/<name>_trim.json` for
any dataset. One card per episode, with everything `review_page.py` had.
Segment specs (`"segments"`, e.g. `takeover_corrections.json` = 15 segments cut from the 4 human-takeover recordings, `v6_corrections.json` = 20, `takeover_raw_segments.json` = all 39 raw takeover segments) get one
card per segment: the blue handles move the segment's start / end (written back into `segments`), "这段不要" (drop this segment)
writes its index into `"drop_seg": {"<ep>": [i]}`, which `trim_stagnant_frames.py` skips; no cuts; labels are keyed `"<ep>.<i>"`.
The thin frame is the segment as first written. "排序" (sort) defaults to the quality score, high to low: `report.py`'s quality
(0.4 smooth + 0.3 task time + 0.3 idle vs the human demos) of each card's kept range when the page loads; task time only counts
where the kept part is a whole task, so takeover segments are scored on smooth + idle. Trim cards:

- Full-width timeline: grey = cut, green = kept, yellow = pause > 1 s, red line = snap, thin frame = the range the rule gives.
- A card on screen plays its three cameras muted in a loop, with the playhead drawn on the timeline and the curves; cards off screen
  release their players (only the cards near the screen decode at a time).
- Hold and drag on the timeline = scrub: playback pauses and the picture follows the pointer; on release it plays on from there; a click plays from there.
- Drag the blue start / end handles (the video jumps to that frame while dragging); add a cut; click a cut or its orange handle to select it,
  then Delete / Backspace removes it (narrow cuts can be selected too).
- Drop the episode, reset (back to the spec as it was before the first write + the current rule), OK / problem / note / tags, export JSON.
  "Drop" here and "force drop" / the manual marks on the training-set quality page are one decision: both live in `dataset_labels.json`
  (`force` / `manual`) and each page shows the other's; a rebuild writes these dropped episodes into the spec's `drop` too.
- Rule parameters: tail tolerance `reset_tol` (v6 default 10), `--to_target`, snap rule. The rule functions are called directly:
  `keep_range` / `reach_start` from `trim_stagnant_frames.py`, `snaps` / `pauses` / `propose` from `review_page.py`, nothing is rewritten.
  Changing them only affects episodes without a fixed range in the spec.
- Joint curves under the video: one row per joint of the grasping arm (6), solid = follower `observation.state`, dashed = leader command
  `action`, to see stalls, snaps and fine-tuning during the reset; click or drag on the curves to jump to a frame, at the top right are "左臂 ⇄" (switch left / right arm)
  and "▾ 关节曲线" (click to fold / unfold this one, it stays in place); or fold all at once with "折叠全部曲线" (fold all curves) in the top bar (remembered).
- Filters: source, tag, only with snaps / with pauses / edited, hide dropped; "只播保留段" (play kept part) loops only the kept part.
- Keys: A / D previous / next episode, space play / pause, `[` `]` set the start / end to the current playhead, Delete removes the selected cut.
- Playback speed is in the top bar, 1x / 2x / 4x / 8x, used by every video on every page (`<` `>` or `,` `.` to change it), remembered.

### Where edits are stored

- Cuts go into the spec file itself (in git): `drop`, `cut`, and the exact kept range of an episode `"range": {"<ep>": [first, end]}`
  (with it `trim_stagnant_frames.py` skips the head/tail rule, `--to_target` and `start_s` for that episode; `cut` still applies; the script
  self-checks it). Before the first write the original spec is copied to `outputs/dataset_viewer/spec_orig/`, which "reset" uses.
- OK / problem, notes and tags go into `examples/xlerobot/dataset_labels.json` (in git), by dataset directory name and episode index,
  shared by all four pages.
- Browser localStorage only keeps UI preferences (last spec, selected datasets, folded curves, cell width).

### Rebuild with the edits

"按修改重建数据集…" (rebuild) first writes every episode's exact range into the spec's `range` (edited ones keep theirs, the rest get the
current rule; drops suggested by the snap rule go into `drop`), the rule parameters into `rule` and the target name into `dst`, then runs
`trim_stagnant_frames.py --spec outputs/dataset_viewer/builds/<time>_<target>.spec.json --dst wwqcleverghost/<target>` in the background
(`setsid`) for every target. Sources with the same target name are merged into one dataset. An existing target needs the overwrite box;
the old copy is renamed to `<target>.old_<time>`, not deleted; a target that is being built cannot be started again.
"构建日志" (build log) at the bottom right shows the progress; the command, git commit and log of every build are in
`outputs/dataset_viewer/builds/`. After a rebuild every episode has a fixed range; only ranges that differ from the current rule are marked edited.

## 2. Batch playback (`#batch`)

Pick one or more datasets; as many columns as the page width fits ("格子宽" cell-width slider), 25 / 50 / 100 per page, each dataset in its own block. All cells run on one
clock: play / pause together (space); finished episodes freeze on their last frame marked "结束" (end), and when the longest one is done all
start again; the speed is the global one from the top bar; head camera only or all three. Filter by source, tag, eval result (success / failure) and quality-score range;
each cell shows dataset, episode index, success / failure, duration, quality score and tags. Click a cell's title for the large view:
three cameras + joint curves + labels, A / D to move.

## 3. Training-set quality (`#quality`)

- **History table**: every training / demo dataset (v1–v6 and the early ones without a version prefix); each column is the dataset's median
  over episodes: duration, median / p95 jerk, peak velocity, static fraction, longest pause, small back-and-forth rate; quality score with
  95% CI; number of IQR-flagged episodes. Click a row to add / remove it from the comparison (at most 8).
- **Rank distribution**: pick a metric; one line per dataset (x = rank inside the dataset, small to large), the colour stays with the dataset;
  hover for values, click a point to watch that episode.
- **Per-episode table**: sortable; only IQR-flagged; click a row to watch it.
- Sources: `episode_metrics` / `flag_outliers` from `lerobot_dataset_quality.py` (jerk and peak velocity on the follower `observation.state`,
  pauses on the leader command `action`, IQR flag = above Q3 + 1.5 x IQR, duration also flagged when too short), and `trials` / `quality` /
  `summarize` from `eval-scoring/report.py` (reference: medians of `v4-handover2-trimmed`).

### Pick high-quality data (lower part of the training-set quality page)

Pick one or more datasets; one row per episode with all the metrics. Every condition has an on/off switch and an adjustable threshold; any
change shows at once how many episodes remain per dataset after each condition in order (cumulative) and which conditions drop each
episode. Click a row to watch it with the joint curves (A / D to move); the buttons at the end of a row force keep / drop
(`force` in `examples/xlerobot/dataset_labels.json`, which wins over every condition).

| Condition | Definition | Default |
|---|---|---|
| Snap | `snaps()` of `review_page.py`: a follower joint jumps > 20 in one frame while its command moves < 5; dropped at >= 1 | on |
| Pause | `pauses()` of `review_page.py`: command < 3 units/s for >= 1 s (the still start does not count) | on |
| Small back-and-forth | `rev_rate` of `report.py` >= reference median x 2 (reference `v4-handover2-trimmed`, 1.467 /s; agreed by the user for the v6 curation) | on |
| Follower jerk | median of the norm of the 3rd difference of the first 12 `observation.state` dims x 30^3 >= reference median x 2 (reference `v4-handover2-trimmed`, 19941; the user agreed to use the follower) | on |
| Follower small snap | frames where the command barely moves (all 10 non-gripper joints \|Δaction\| < 2) but a follower joint jumps > 6; >= 15. **New metric, waiting for the user's OK** | off |
| Follower stall | the command moves (max \|Δaction\| > 1) while the follower stands still (max \|Δstate\| < 0.3) for >= 4 frames in a row counts once; >= 8. **New metric, waiting for the user's OK** | off |
| Checklist: placement / regrasp / midline | tape >= 500 yellow pixels in the last head frame / <= 2 gripper closes per arm / first holding arm follows the midline rule (`ep_score.py`, `select_strict.py`) | off |
| Checklist: tape standing at the start | height / width of the tape blob in the first frame >= 1.2. The checklist has no number for it: **threshold waiting for the user's OK** | off |

The jerk reference is the trimmed handover2 (`v4-handover2-trimmed`). The raw `handover2` contains the still head and tail, its median is
only 14172, and with it as the reference many more episodes go (48 left of v6-handover1/2-trimmed instead of 61).
Reference numbers on v6-handover1/2-trimmed (2026-10-27): 71 / 100 left without snaps and pauses, 61 with jitter and jerk added, 56 with
small snaps and stalls on as well.

- **All / only kept / only dropped**: one switch for the result table and the start / end frame review (remembered in the browser).
  "批量播放保留的 N 条" / "批量播放去掉的 N 条" (play the kept / dropped) hand the current result (all datasets together) to the batch page,
  which shows "来自筛选：预设 X，共 N 条" (from curate: preset X, N episodes) and a back button, and can still filter by source / tag /
  score; in the large view A / D stay inside this batch and the title says kept or dropped and why.
- **Start / end frames** ("起止画面" in the "表格 / 起止画面" table / frames switch at the top): for the manual check of "tape standing at the start" and "tape put in the
  basket". 20 cards a page, each with the first and the last frame of the kept range from the head camera, large (the datasets on this
  page are already trimmed, so the kept range is the whole episode, as in the spec it writes). Three buttons per card: standing at the start,
  in the basket, fine; keys 1 / 2 / 3, A / D or the arrow keys to move, it moves on after a mark, pressing again removes the mark.
  Marks go into `manual` in `dataset_labels.json` (`standing` / `basket` / `ok`): the first two drop the episode (reason "人工：开头立着 /
  放进框里", the last step "manual" of the cumulative counts); "fine" only records it as seen (dimmed). Show only the unseen ones or only
  those with a hint (0 tape pixels in the last frame: "maybe in the basket"; tape blob height / width >= the threshold at the start: "maybe
  standing"; hints only, nothing is dropped automatically, the automatic rules standing / place stay off by default).
  The manual marks go through the same `curate.py` computation, so the command line gives the same result as the page.
- **Presets**: a set of switches and thresholds saved as `presets/<name>.json` (in git), switched and saved from the page. The default
  preset `default.json` is the table above.
- **Spec and build**: enter a name; "按筛选结果生成 spec" (make spec) writes `examples/xlerobot/<name>.json`: kept episodes whole (`range` =
  the whole episode, the dataset's existing trim), the rest in `drop`, the date and the conditions in `note`, the target dataset named the
  same by default with all sources merged into one. "构建数据集…" (build) then runs the rebuild above in the background (`--arm_only`
  optional). The sources are already trimmed datasets, so the videos are encoded once more.
- Metrics are cached in `outputs/dataset_viewer/cache/scores/<dataset>.curate.json` (the first time a dataset needs its first and last head
  frames decoded, a dozen seconds).

### Command line (the same functions as the page)

```bash
PYTHONPATH=src python docs/workflows/dataset-viewer/curate.py --datasets raw-handover-20260929-trimmed raw-handover-20261001-trimmed \
  [--preset default] [--set jerk.x=2.5 --set stall.on=1 ...] [--report out.csv] [--spec <name> [--build <new dataset> [--arm_only]]]
```

It prints, per dataset, how many remain after each condition, which episodes each condition drops, the forced ones and the total kept.
Steps for agents: `.claude/skills/select-high-quality-data/SKILL.md`.

## 4. Eval scores (`#eval`)

With the rules of [`../eval-scoring/README.md`](../eval-scoring/README.md) (calls `report.py` directly):

- **History table**: every eval, newest version first, by total within a version; success rate (Wilson), completion, quality and total
  with 95% CI, stage funnel `passed/reached`. Click a row for details.
- **Details**: stage funnel (per stage `passed/reached`, Wilson interval, and the (passed+1)/(reached+2) used for the score) and the per-trial
  table (human label, the 5 stages, smooth / speed / idle, quality, task time).
- Failed trials have a "▶ failure moment" button: playback starts 1.5 s before it, a red dashed line marks it on the curves. The moment is
  approximate: 0.5 s before the last grasp ended (tape dropped or released); 3 s before the end when the tape was never held.

Both score pages cache their results in `outputs/dataset_viewer/cache/scores/`; they are recomputed when the dataset's files or a scoring
script (report.py, score.py, smooth.py, lerobot_dataset_quality.py) change. The first time a history table opens, a background thread
computes what is missing (about 1 minute for all) and the table fills in as it goes; after that it opens in under a second.
Bump `SCORE_VERSION` when `_quality` / `_evaluation` in `server.py` change what they return.

## Files

- `server.py`: server and JSON API (`/api/datasets`, `episodes`, `curves`, `frame`, `review`, `review/edit`, `label`, `rebuild`, `builds`,
  `quality`, `eval`, `history`, `curate/*`, plus `/video`). `python server.py` runs its self-check first.
- `index.html`: top bar and styles; `app.js`: the four pages (old `#curate` links open the training-set quality page).
- `curate.py`: curate metrics, conditions and command line; `presets/`: curate presets (in git).
- Generated at run time (not in git): `outputs/dataset_viewer/` (still and score caches, spec backups, build records and logs).
