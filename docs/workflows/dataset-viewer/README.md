# Dataset viewer

中文：[README.zh.md](README.zh.md)

One local web page, five pages under one shared top bar: **trim review**, **batch playback**, **training-set quality / curate**, **eval scores**, **stages / progress**.
Datasets are listed automatically from `~/.cache/huggingface/lerobot/wwqcleverghost/` and, as `eval/<run>`, from the eval runs not yet uploaded in `outputs/eval/<run>/dataset/` (eval datasets included, also when nested in
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

Pick a dataset at the top, in three groups: **raw recordings** (`v1-handover1`, `v2-handover2` and the human-takeover / DAgger
recordings that have `interventions.csv`; add new raw recordings to `RAW` in `server.py`), **built datasets**, **evals**. One card per
episode, in **review mode** by default: watch and label only, no data is changed.

**Labels only describe the episode itself** (note, tags, the start / end frame marks "standing at the start" / "in the basket"); they are
stored in `examples/xlerobot/dataset_labels.json` by dataset and episode index and saved on click; a label never drops an episode by
itself. "通过 / 有问题" (OK / problem) was removed (2026-10-30).

**Trim mode** (button "裁剪" or key T) is only available for raw recordings. Trimming works through a spec: a spec is a set of operations
(sources + which episodes to drop + the kept range of each + the cuts) that **belongs to the dataset it builds**, and its file name is that
dataset's name: `examples/xlerobot/<dataset to build>.json`. "生成" (builds) lists every spec that uses this recording:

- **草稿 (draft)**: not built yet, editable. "新建…" (new) asks for the name of the dataset to build and creates a draft with this recording.
- **已构建 · 只读 (built, read-only)**: the spec has `"built"` (written by the build: time + git commit); the server refuses edits. To change
  it, "另存为…" (save as) a draft under a new name. Specs built before 2026-10-30 were renamed after their datasets and marked `"built"` by
  hand (`v5-handover-perfect*.json`, `v6-handover-curated.json`, `v6-corrections.json`, `takeover-corrections.json`, `v7-handover-clean.json`;
  `v6_trim_all.json` built the two `raw-handover-*-trimmed`, since deleted).

The rest of this section is the trim mode.
Segment specs (`"segments"`, e.g. `takeover-corrections.json` = 15 segments cut from the 4 human-takeover recordings, `v6-corrections.json` = 20, `takeover_raw_segments.json` = all 39 raw takeover segments) get one
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
- Drop the episode (written only into the spec's `drop`, not into the labels), reset (back to the spec as it was before the first write +
  the current rule), export JSON.
- Rule parameter: tail tolerance `reset_tol` (v6 default 10). The `--to_target` and snap-rule switches were removed (2026-10-30, user: not useful). The rule functions are called directly:
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
- Notes and tags go into `examples/xlerobot/dataset_labels.json` (in git), by dataset directory name and episode index,
  shared by all four pages.
- Browser localStorage only keeps UI preferences (last spec, selected datasets, folded curves, cell width).

### Build the dataset

"构建数据集…" (build; drafts only) first writes every episode's exact range into the spec's `range` (edited ones keep theirs, the rest get the
current rule), the rule parameters into `rule`, the target name (default = the spec's name) into `dst` and the build time + git commit into
`built` (read-only from then on), then runs
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

- **History table**: every training / demo dataset (newest version first); each column is the dataset's median
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
  those with a hint (0 tape pixels in the last frame: "maybe in the basket"; a hint only, nothing is dropped automatically, the automatic
  rule place is off by default). The automatic "tape standing at the start" rule (blob height / width in the first frame) was unreliable and
  was removed at the user's request (2026-10-30); standing is only marked by hand here.
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

## 5. Stages / progress (`#stage`)

For the stage models (stage input, stage head, `*-stagepred`, `*-progpred`, see [`../stage-labels/README.md`](../stage-labels/README.md)).
Pick one or more evals. At the top, **model comparison**, one line per eval (sortable; click a line to jump to its trials):
success k/n, how many trials reached each of the 9 stages (one bar per stage, its height = the share of trials that got there,
hover for counts), median time of the successful trials (task end, see below) with min–max, rule and model ▼ per trial, and
"模型 = 规则" = the share of frames before the end where the model's stage equals the rule's.

Below, one row per trial, all rows on one time axis (gridline every 5 s), so lengths compare at a glance:

- Top band: the rule stage. The live `fine9` stage fed to the model when the eval wrote `stages.csv`, else `fine9` recomputed
  offline (decodes the wrist videos the first time, one trial at a time).
- Second band: the model's own stage guess (stage head / `*-stagepred` argmax / `*-progpred` progress → stage); faint where its
  probability is low. "（没有模型阶段输出）" = the eval has no `stages.csv` or the model has no stage output.
- Plot 0..1: black = progress of the rule stage, blue = the model's progress (`*-progpred`: its own output; others: from its stages).
  Progress per stage uses the fixed v6 shares `stage_labels.PROG_SHARES`.
- Marks on the bottom edge of the plot (filled = model, hollow = rule): red ▼ = the stage goes down (back), orange ▲ = a stage is
  skipped, or the predicted progress steps by > 0.1 in one frame; counts next to the trial (▼n ▲n), and the model's ▼ total in the panel header.
- Dashed line = the task end (always inside 复位 when the trial got there), grey after it (nothing is cut): `episode_end.csv` from
  [`../eval-scoring/mark_end.py`](../eval-scoring/README.md); without it, computed on the fly from `stages.csv`, else not shown (run
  mark_end.py: it recomputes fine9 from the videos). The label shows end / recorded seconds.
- Hover: time, both stages, probability / progress. Click: the trial's player and joint curves from that moment, greyed after the end.

`stages.csv` (episode, frame, rule, pred, prob, progress; -1 / empty = none) is written next to `success.csv` by
`lerobot_record_eval.py` for every saved trial of a model with a stage input or output (not by `eval_remote_trials.py` yet).

## Auto stage labels (top bar "阶段")

A select in the top bar draws per-frame task stages as a colored band above every set of joint curves (review cards and the episode
pop-up on every page); hover the band for the stage name and its time range. The stage definitions, rules and their accuracy are in
[`../stage-labels/README.md`](../stage-labels/README.md); the code is [`examples/xlerobot/stage_labels.py`](../../../examples/xlerobot/stage_labels.py).
The select offers `fine9` (causal, what the robot can compute live, the labels training uses) and `eval5` (eval scoring); the reference
`std9` is only on the command line, for checking (`--compare`). `fine9` measures the tape in the wrist cameras: the first time an episode is shown it decodes both wrist videos (about 6 s),
cached in `outputs/dataset_viewer/cache/tape/`.

**Hand-set stage starts**: tick "改阶段" in the top bar (stage rule `fine9`) and every boundary on the stage band gets a handle; drag it
like a trim handle, it is saved on release and the video follows it. Double-click a handle to go back to the rule. Hand-set handles are
orange, the orange dashed line is where the rule put it. Saved in the git-tracked `examples/xlerobot/stage_edits.json`
(`{dataset: {ep: {stage name: start frame}}}`, `<name>-stagelabel` shares the edits of `<name>`); `add_stage_labels.py` applies them.
Note: the stage the robot computes live in eval has no hand edits (see "Hand corrections" in [`../stage-labels/README.md`](../stage-labels/README.md)).

## Files

- `server.py`: server and JSON API (`/api/datasets`, `episodes`, `curves`, `stages`, `stageview`, `stage_rules`, `frame`, `review`, `review/edit`, `label`, `rebuild`, `builds`,
  `quality`, `eval`, `history`, `curate/*`, plus `/video`). `python server.py` runs its self-check first.
- `index.html`: top bar and styles; `app.js`: the five pages (old `#curate` links open the training-set quality page).
- `curate.py`: curate metrics, conditions and command line; `presets/`: curate presets (in git).
- Generated at run time (not in git): `outputs/dataset_viewer/` (still and score caches, spec backups, build records and logs).
