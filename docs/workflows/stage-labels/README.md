# Stage labels

中文：[README.zh.md](README.zh.md)

Labels every frame of a handover episode with its task stage, as an extra model input (`observation.environment_state`) so the model knows
how far the task has got and stops confusing the start with the end. Code: [`examples/xlerobot/stage_labels.py`](../../../examples/xlerobot/stage_labels.py);
the "阶段" select in the dataset viewer's top bar shows them ([`../dataset-viewer/README.md`](../dataset-viewer/README.md)).
The 5 eval-scoring stages (`score.stage_frames`, see [`../eval-scoring/README.md`](../eval-scoring/README.md)) are rule `eval5` here.

**In short**: label a training set → train on `<name>-stagelabel` → in real-robot eval the scripts compute the stage live and feed it to the
model **automatically** (see "Eval"). When the rules change, label already-labelled datasets again (see "Writing them into a dataset"), or
the training labels will not match what eval computes.

## Conventions

- **A** = the first arm that holds the tape, **B** = the other arm.
- Every stage boundary is one event. Each event is defined once and needs only the current and past frames; it has two frames:
  **confirmed** (enough evidence) and **onset** (the first frame of that evidence, found by looking back).
- Frame t uses the follower state of frame t and the **command of frame t−1** (the state at frame 0): the recorded action of frame t is
  decided after observing frame t, and live the robot only knows the command it sent last.
- Gripper values: ~1 = closed on nothing, ~30 = open; holding the tape it reads ~7–12.
  "Closing / letting go" use the leader command (`action`), "holding / fully open" the follower state (`observation.state`).
- Tape diameter = the larger of the horizontal and vertical extent (2nd–98th percentile) of the yellow-green tape pixels in the
  wrist camera (PIL HSV: hue 28–80 of 255, saturation > 70, value > 60; computed at 160×120), as a fraction of the frame width.
  The gripper fingers hide the middle of the tape but rarely its edges, so this is steadier than the area.
- Gripper tip positions come from forward kinematics of the joint readings ([`examples/xlerobot/xlerobot_fk.py`](../../../examples/xlerobot/xlerobot_fk.py), see "Forward kinematics" below).

## 9 stages

| # | Stage | Starts at event | Onset / confirmed | How it looks in the video |
|---|---|---|---|---|
| 0 | A前往 (A reaches) | start | | the arm moves toward the tape, the tape is still small in its wrist camera |
| 1 | A对位 (A aligns) | **E0**: tape diameter >= 0.18 (half the grasp-time diameter) in an arm's wrist camera with that gripper **fully open** (reading > 10 and still, change < 1) | first of 5 frames in a row / the 5th | the gripper is open and the tape close, lining up |
| 2 | A抓住 (A grasps) | **E1**: an arm starts closing **and its tip has nearly stopped** (< 2 cm/s over the last 5 frames **along the gripper axis** (wrist → tip, about the wrist z axis); sliding along the table does not count, e.g. v8 ep32), whichever is later; the first to do so is A. Closing while still moving down to align does not count | the later of the closing onset (more than 2 below the peak of this opening) and the frame the tip stopped / 5 below and still | the tip stops, the gripper closes |
| 3 | A抓着 (A holds) | **E2**: A holds = the gripper reading averaged over 5 frames sits at a tape width (4–20), changes by < 0.5 within 5 frames (small jitter averages out), and the command is not opening it (command < reading + 2) | the frame the reading settled / the frame it is confirmed (about 9 later) | A holds the tape |
| 4 | AB对位 (B aligns) | **E3**: B enters A's wrist-camera view = angle between B's gripper tip and A's gripper axis (wrist → tip) < 35° | first of 5 frames in a row / the 5th | B shows up in A's wrist camera |
| 5 | B抓住 (B grasps) | **E4**: B starts closing (the closing of E1, **no stillness required**: B moves in to meet A), after B had opened (command peak >= 15; a close from half open does not count, e.g. v8 ep26 f234 peaked at 11). A letting go while B is closing also counts (the hand-off is under way) | onset of this closing / 5 below the peak | B is closing |
| 6 | B抓着 (B holds) | **E5**: **B holds** (as E2, with the reading more than 2 below where the closing started). Does not wait for A: A lets go within B抓着 (A lets go = A's command more than 4 above its tightest grip reading and > 10; **loosening a little does not count**) | the frame the reading settled / the frame it is confirmed | B holds the tape, A opens, moving to the drop place |
| 7 | B放下 (B puts down) | **E6**: B lets go (as A letting go, for B) | first frame of the run of commands above (let-go threshold − 2) (an earlier loosening does not count) / the frame it passes the threshold | B releases the tape |
| 8 | 复位 (reset) | **E7**: after letting go B's arm **heads back to its start pose** = B's joint distance to the start pose (the episode's first frame) falls every frame for 5 frames, by at least 2 (recorded units) in all. Lifting / moving away right after letting go does not count; the gripper is not looked at | first frame of that falling run / the 5th | the arms head back |

Holding tight does not require "command well below the reading": once the tape is held the operator often eases the leader to about the
reading (~10). A reading <= 4 is closed on nothing, ~30 is a motor that has not started moving; 4–20 excludes both.

## Failures (no failure labels; stages only go up)

Within an episode the stage number **never goes down**, except when the tape is confirmed dropped: live, the robot has to know it
must pick it up again, or the tape lies on the table while the stage still says "A holds".

| Case | Rule | Stage |
|---|---|---|
| A misses / reopens halfway and retries | | stays in A抓住 until it holds tight |
| B misses / retries | | stays in B抓住 until B holds |
| A drops it / lets go without a handover | A closed all the way (reading <= 4 for 5 frames), or A lets go before B started closing | back to A前往 (A and B decided again) |
| dropped during the hand-off | A has let go and B is closed all the way | back to A前往 |
| B drops it after the handover | B closed all the way | back to A前往 |
| episode trimmed before the end | no later event | stays in the last stage |

After going back to A前往, only closings that start afterwards count (the earlier closing cannot immediately count as a new grasp).
A letting go after B started closing but before B holds tight is still a hand-off: it stays in B抓住 until B holds tight.

## Two rules, one set of events

- **`fine9`**: switches stages when events are **confirmed**; the robot computes exactly the same live (`StageTracker.step`, `LiveStage` in eval).
  **Training inputs use it**: the robot can only compute `fine9` at test time and training inputs must match test time; trained on
  `std9`, the model would learn stages that come a few frames earlier than it will see. The viewer's select only offers it.
- **`std9`**: the same events, switched at their **onsets**. Used for checking (against the videos, `--compare`) and, if stages ever become a
  **training target** (e.g. SARM or a stage classifier), for that. Both rules have exactly the same stage order; they only differ by a few
  frames at each boundary.

## Forward kinematics

[`examples/xlerobot/xlerobot_fk.py`](../../../examples/xlerobot/xlerobot_fk.py) computes the wrist and gripper-tip positions of both arms
from the joint readings (right-arm base frame, x forward, y left, z up):

- Arm chain: `so101_new_calib.urdf` of TheRobotStudio/SO-ARM100 (commit 385e8d7c68, 2025-07-02); joint zero = middle of each joint's
  range, as in LeRobot's calibration.
- Readings: recorded with `use_degrees=False`, so values are -100…100 (0 = middle of the calibrated range);
  angle = value / 100 × half the range in ticks × 360° / 4096. The ranges come from the calibration file `xlr26009.json` on the lab PC
  (2026-08-20).
- Left arm base: fitted on frames where both grippers hold the tape (the 10 frames before A lets go): (-1.9, 24.3, 0.3) cm, same
  heading; the XLeRobot model `xlerobot.urdf` says 26.6 cm apart, same height and heading. The two tips are then a median 4.9 cm
  apart (the two grippers hold opposite sides of the tape); flipping shoulder_lift / elbow_flex / wrist_flex gives 18 cm, so the signs
  are right.
- View: A's wrist camera is approximated as a cone from the wrist along the gripper axis (about 4.6° off the end-effector z axis); only
  B's gripper tip is tested. In the wrist videos of v7 ep3, ep20 and ep33 B appears at 25–32°; it was 30° first, but in v8 ep26 and ep38
  B only came to 33.5–34° before closing and AB对位 was skipped, so 35° (AB对位 starts a median 0.2 s earlier elsewhere, 90% within 1 s).
  The camera sits above the gripper and looks down at an angle, so this is narrower than its real field of view.

## Thresholds and accuracy

2026-11-03, git commit 12ec265d + uncommitted changes.

**Tape diameter in A's wrist camera when A starts closing (E1 onset)** (fraction of the frame width): `v8-handover2` 45 episodes, median
0.362, p10 0.306, p90 0.398; `v7-handover-clean` 53 episodes, median 0.350, p10 0.281, p90 0.388. Threshold = 0.5 × the v8 median = 0.18
(set by the user on 2026-11-03; it was 2/3 × the v7 median = 0.23, which made alignment too short in episodes with a small grasp-time
diameter, e.g. v8 ep30 only 7 frames).

**A stillness threshold**: only the speed along the gripper axis, 2 cm/s (the user asked for a strict one for A, axial only: in v8 ep32 A
moves < 1 cm/s along the axis while closing but slides 3–5 cm/s along the table; total speed would start A抓住 at 156, axial at 139). **B opening threshold**: before a real grasp B's command peaks
at 17.7 or more (v7, v8); 15 is used.

**`fine9` against `std9`** (`--compare`):

| | v8-handover2-stagelabel | v7-handover-clean |
|---|---|---|
| episodes | 45 | 53 |
| frame agreement median / mean / lowest | 93% / 93% / 82% | 93% / 93% / 82% |
| reach 复位 | 44 (ep22 trimmed) | 52 (ep38 trimmed) |
| going down | none | none |
| `std9` A对位 duration median / longest | 2.40 / 8.07 s (ep31) | 2.30 / 8.07 s |
| `std9` A抓住 duration median / longest | 0.50 / 5.40 s (ep29) | 0.53 / 5.40 s |
| `std9` B抓住 duration median / shortest / longest | 0.57 / 0.23 / 2.43 s | 0.67 / 0.23 / 8.70 s (ep45, B retries) |
| `std9` B放下 duration median / shortest / longest | 1.00 / 0.20 / 2.37 s | 1.07 / 0.47 / 3.23 s |
| skip A对位 (the gripper closes right after opening, never "fully open") | ep29 (both rules), ep25 (`std9` only) | ep23 (both rules), ep21 (`std9` only) |
| `std9` also squeezes A抓住 to 0 frames (already holding when the tip stops) | ep12 | ep9, ep48, ep51 |

Frames `fine9` enters each stage after `std9` (median / largest, 30 frames = 1 s):

| Stage | v8 | v7 |
|---|---|---|
| A前往 | 0 / 0 | 0 / 0 |
| A对位 | 4 / 4 | 4 / 4 |
| A抓住 | 2 / 68 | 2 / 68 |
| A抓着 | 9 / 9 | 9 / 9 |
| AB对位 | 4 / 4 | 4 / 4 |
| B抓住 | 1 / 6 | 1 / 6 |
| B抓着 | 9 / 9 | 9 / 9 |
| B放下 | 1 / 3 | 1 / 41 |
| 复位 | 4 / 11 | 4 / 11 |

Holding (A抓着, B抓着) is 9 frames late: the reading is averaged over 5 frames, then checked for 5 frames. A抓住's occasional large lag
comes from slow closing (confirmed 5 below the peak).

## Writing them into a dataset

Training reads only the dataset's own files, so the stage has to be a column there:
[`examples/xlerobot/add_stage_labels.py`](../../../examples/xlerobot/add_stage_labels.py) writes the 9-float one-hot of `fine9` as
`observation.environment_state` (**in place**: only `data/*.parquet`, `meta/info.json` and `meta/stats.json` change, videos are untouched;
`data/` and `meta/` are backed up to `<owner>/_trash_<date>/` first), then renames the dataset `<name>-stagelabel`. Running it again on a
dataset that has the column recomputes it with the current rules (no rename). Diffusion does not normalise ENV inputs, so the one-hot
reaches the model as is; with the column in the dataset, diffusion training uses it as an input automatically.

```bash
PYTHONPATH=src python examples/xlerobot/add_stage_labels.py --dataset v8-handover2              # add the column, rename to v8-handover2-stagelabel
PYTHONPATH=src python examples/xlerobot/add_stage_labels.py --dataset v8-handover2-stagelabel   # recompute after a rule change
```

2026-11-03: `v8-handover2` (45 episodes, 23328 frames, built locally, not on HF) → `v8-handover2-stagelabel`, recomputed the same day with
the rules in this document (last after 对位 required a fully open gripper and 复位 became B's arm heading back to its start pose, the view widened to 35°, A抓住 required a still tip and B抓着 started at B's hold, judged on a 5-frame average). Frames per stage: A前往 3813, A对位 3352, A抓住 1447, A抓着 3539, AB对位 2276, B抓住 1274, B抓着 2304,
B放下 1572, 复位 3751.

## Eval

In real-robot eval the model needs the stage of every frame. `lerobot_record_eval.py` and `examples/xlerobot/eval_remote_trials.py` handle it
**automatically**: when the model's inputs include `observation.environment_state` (a model trained on a `*-stagelabel` dataset), each frame
`stage_labels.LiveStage` (joint readings + the wrist images `left_wrist` / `right_wrist` + the command sent last) computes the `fine9` stage
and feeds it to the model; startup prints `Stage input on: ...`. It keeps updating during a human takeover, so the stage is current when the
policy resumes. The stage only goes to the model and is not written into the recorded test set (the same rule recomputes it offline).
No extra flag; no `Stage input on` line means the model has no such input.

## Command line

```bash
PYTHONPATH=src python examples/xlerobot/stage_labels.py --dataset v8-handover2-stagelabel [--rule fine9|std9|eval5] [--episodes 0 1] [--compare]
```

`--compare`: compares `--rule` with `std9` episode by episode and prints frame agreement and how many frames later each stage is entered.
The first time, the wrist-camera tape diameters need both wrist videos decoded (about 6 s per episode); they are cached in
`outputs/dataset_viewer/cache/tape/` and shared with the dataset viewer.
