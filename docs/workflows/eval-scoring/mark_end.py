"""Per episode of a dataset (eval or demo): where the task really ended and the furthest stage it got to.
Writes <dir>/episode_end.csv (episode, end_frame, frames, how, top_stage). Nothing is cut. report.py scores with it.

Stages are fine9 (examples/xlerobot/stage_labels.py), always recomputed offline from the wrist videos with the current rule
(stages.csv, the stage the model saw live, keeps the rule of the eval day: 2026-11-17 the rule changed, see stage-labels).
top_stage = the highest stage any frame is in (0-8), the rule alone, no human label.

The operator stops a trial by hand, a few seconds early or late, so the recorded length is not the task's length.
still = the frame from which every arm joint (command and follower) stays within RESET_TOL of its last pose (the tail rule
of trim_stagnant_frames.keep_range): the robot has stopped and the rest is waiting for the operator.
Episodes that reached 复位 (stage 8) end where it starts (user, 2026-11-06): many were stopped mid-reset, so time up to the
reset is the one all of them share.
- how=reset: end = the first 复位 frame.
Other episodes:
- how=still: end = still.
- how=human: never reached 复位 and moved until the last 0.5 s; end = the recorded length.
The dataset viewer draws this end on the "阶段 / 进度" page and greys the rest; the video and data stay whole.
Run it again after the stage rule or a hand-set stage start (stage_edits.json) changes: the scores read this file.

usage: PYTHONPATH=src python docs/workflows/eval-scoring/mark_end.py <dataset dir> ...
"""
import csv
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score import FPS, locate  # noqa: E402

RESET_TOL = 3.0  # ponytail: one tolerance for all joints (recorded units); raise it if a jittering policy never reads "still"
RESET = 8  # stage_labels.STAGES.index("复位")
COLS = ["episode", "end_frame", "frames", "how", "top_stage"]


def still_from(st, ac):
    """First frame from which every arm joint (command and follower) stays within RESET_TOL of its last pose (the tail
    rule of trim_stagnant_frames.keep_range, +1 frame as there); 0 for a trial that never moved."""
    p = np.c_[ac[:, :12], st[:, :12]]
    away = np.flatnonzero(np.abs(p - p[-1]).max(1) >= RESET_TOL)
    return min(int(away[-1]) + 2, len(p)) if len(away) else 0


def task_end(st, ac, reset_start):
    """(end frame, how) of one trial; st / ac: (T, 12) follower state / command; reset_start = first 复位 frame or None."""
    if reset_start is not None:
        return reset_start, "reset"
    end = still_from(st, ac)
    if end >= len(st) - FPS // 2:  # still only in the last 0.5 s: the operator stopped it while it moved
        return len(st), "human"
    return end, "still"


def reset_start(lab):
    """First frame of 复位 in a per-frame stage array (None if never reached)."""
    hit = np.flatnonzero(np.asarray(lab) == RESET)
    return int(hit[0]) if len(hit) else None


def end_file(run):
    """episode_end.csv of a dataset dir: next to success.csv for an eval, in the data dir for a demo dataset."""
    base, csv_path = locate(run.rstrip("/"))
    return os.path.join(os.path.dirname(csv_path) if csv_path else base, "episode_end.csv")


def viewer_fine9(base):
    """ep -> per-frame fine9 labels through the dataset viewer (decodes the wrist videos once and caches the tape sizes)."""
    sys.path.insert(0, os.path.join(HERE, "../dataset-viewer"))
    import server  # noqa: E402  (imports this module: import here, not at the top)
    ds = server.ds_id_of(dict(root=os.path.abspath(base)))  # the viewer's id, so hand-set stage starts and caches apply
    return lambda ep: np.concatenate([[g] * (b - a) for a, b, g in server.stages(ds, int(ep), "fine9")["segs"]])


def mark(run, fine9=None, quiet=False):
    """Write episode_end.csv of one dataset dir. fine9(ep) -> per-frame stage labels (default: through the dataset viewer)."""
    base, csv_path = locate(run.rstrip("/"))
    d = pd.concat(pd.read_parquet(f, columns=["episode_index", "action", "observation.state"])
                  for f in sorted(glob.glob(f"{base}/data/*/*.parquet")))
    fine9 = fine9 or viewer_fine9(base)
    rows = []
    for ep, g in d.groupby("episode_index"):
        st, ac = np.stack(g["observation.state"].values)[:, :12], np.stack(g["action"].values)[:, :12]
        lab = np.asarray(fine9(ep))
        end, how = task_end(st, ac, reset_start(lab))
        rows.append([int(ep), end, len(st), how, int(lab.max())])
    out = end_file(run)
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows([COLS, *rows])
    if not quiet:
        print(out)
        for ep, end, n, how, top in rows:
            print(f"  ep{ep:<3} {how:6}  end {end / FPS:5.1f} s / recorded {n / FPS:5.1f} s  top stage {top}")
    return out


def load(run, fine9=None, episodes=None):
    """episode_end.csv of a dataset dir, indexed by episode. Written first when it is missing, older than top_stage, or
    does not cover `episodes` (trials were added since)."""
    path = end_file(run)
    ok = os.path.exists(path)
    if ok:
        df = pd.read_csv(path).set_index("episode")
        ok = "top_stage" in df and (episodes is None or set(episodes) <= set(df.index))
    if not ok:
        df = pd.read_csv(mark(run, fine9, quiet=True)).set_index("episode")
    return df


def selfcheck():
    st = np.zeros((100, 12))
    st[:40, 0] = np.linspace(0, 50, 40)
    st[40:, 0] = 50  # stops at frame 40, waits 2 s
    assert task_end(st, st, None) == (38, "still"), task_end(st, st, None)  # within 3 of 50 from frame 37 on
    assert task_end(st, st, 30) == (30, "reset")
    assert reset_start([0, 7, 8, 8, 0, 8]) == 2 and reset_start([0, 1]) is None
    st[:, 0] = np.linspace(0, 50, 100)  # moves to the end
    assert task_end(st, st, None) == (100, "human")
    assert task_end(np.zeros((100, 12)), np.zeros((100, 12)), None) == (0, "still")  # never moved


if __name__ == "__main__":
    selfcheck()
    for r in sys.argv[1:]:
        mark(r)
