"""Mark where each eval trial really ended: <eval dir>/episode_end.csv (episode, end_frame, frames, how). Nothing is cut.

The operator stops a trial by hand, a few seconds early or late, so the recorded length is not the task's length.
still = the frame from which every arm joint (command and follower) stays within RESET_TOL of its last pose (the tail rule
of trim_stagnant_frames.keep_range): the robot has stopped and the rest is waiting for the operator.
- how=reset: the trial reached 复位 (fine9 stage 8: the live stage from stages.csv when the eval wrote it, else recomputed
  offline): end = the later of still and the first 复位 frame + 1, so the end is always inside 复位 (user, 2026-11-05).
- how=still: never reached 复位: end = still.
- how=human: the robot moved until the last 0.5 s; end = the recorded length.
The dataset viewer draws this end on the "阶段 / 进度" page and greys the rest; the video and data stay whole.

usage: PYTHONPATH=src python docs/workflows/eval-scoring/mark_end.py <eval dir> ...
"""
import csv
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "../../../examples/xlerobot"))
from score import FPS, locate  # noqa: E402
from trim_stagnant_frames import keep_range  # noqa: E402

RESET_TOL = 3.0  # ponytail: one tolerance for all joints (recorded units); raise it if a jittering policy never reads "still"


RESET = 8  # stage_labels.STAGES.index("复位")


def task_end(st, ac, reset_start):
    """(end frame, how) of one trial; st / ac: (T, 12) follower state / command; reset_start = first 复位 frame or None."""
    end = keep_range(ac, st, list(range(12)), 0.2, 2.0, RESET_TOL)[1]
    how = "still"
    if reset_start is not None:
        end, how = max(end, reset_start + 1), "reset"
    return (end, how) if end < len(st) - FPS // 2 else (len(st), "human")  # still < 0.5 s: stopped while moving


def reset_start(lab):
    """First frame of 复位 in a per-frame stage array (None if never reached)."""
    hit = np.flatnonzero(np.asarray(lab) == RESET)
    return int(hit[0]) if len(hit) else None


def mark(run):
    base, csv_path = locate(run.rstrip("/"))
    assert csv_path, f"{run}: no success.csv"
    d = pd.concat(pd.read_parquet(f, columns=["episode_index", "action", "observation.state"])
                  for f in sorted(glob.glob(f"{base}/data/*/*.parquet")))
    log = os.path.join(os.path.dirname(csv_path), "stages.csv")
    log = pd.read_csv(log) if os.path.exists(log) else None
    rows = []
    for ep, g in d.groupby("episode_index"):
        st, ac = np.stack(g["observation.state"].values)[:, :12], np.stack(g["action"].values)[:, :12]
        live = None if log is None else log[log.episode == ep]
        if live is not None and (live.rule >= 0).any():  # the stage the model saw during the eval
            lab = np.full(len(st), -1)
            lab[live.frame.values.clip(0, len(st) - 1)] = live.rule.values
        else:  # recompute fine9 offline (decodes the wrist videos; cached with the dataset viewer)
            sys.path.insert(0, os.path.join(HERE, "../dataset-viewer"))
            import server  # noqa: E402  (imports this module: import here, not at the top)
            lab = np.concatenate([[g_] * (b - a) for a, b, g_ in server.stages(os.path.abspath(base), int(ep), "fine9")["segs"]])
        end, how = task_end(st, ac, reset_start(lab))
        rows.append([int(ep), end, len(st), how])
    out = os.path.join(os.path.dirname(csv_path), "episode_end.csv")
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows([["episode", "end_frame", "frames", "how"], *rows])
    print(out)
    for ep, end, n, how in rows:
        print(f"  ep{ep:<3} {how:6}  end {end / FPS:5.1f} s / recorded {n / FPS:5.1f} s")


def selfcheck():
    st = np.zeros((100, 12))
    st[:40, 0] = np.linspace(0, 50, 40)
    st[40:, 0] = 50  # stops at frame 40, waits 2 s
    assert task_end(st, st, None) == (38, "still"), task_end(st, st, None)  # within 3 of 50 from frame 37 on
    assert task_end(st, st, 30) == (38, "reset") and task_end(st, st, 50) == (51, "reset")  # never before 复位 starts
    assert reset_start([0, 7, 8, 8, 0, 8]) == 2 and reset_start([0, 1]) is None
    st[:, 0] = np.linspace(0, 50, 100)  # moves to the end
    assert task_end(st, st, None) == (100, "human")


if __name__ == "__main__":
    selfcheck()
    for r in sys.argv[1:]:
        mark(r)
