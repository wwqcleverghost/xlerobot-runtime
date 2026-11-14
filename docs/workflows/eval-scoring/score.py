"""Quality inputs of a handover episode from proprioception only (no vision): when the arms first move, how long they
stand still, tape holds. Nothing here judges stages: completion and the task end come from the fine9 stage labels
(report.py, mark_end.py; examples/xlerobot/stage_labels.py).
hold = gripper commanded closed but stalled open by the tape (state - action > GAP)."""
import os

import numpy as np

FPS, GAP, OPEN, MIN = 30, 2.0, 10.0, 15   # MIN frames = 0.5 s
ARM = {'L': slice(0, 5), 'R': slice(6, 11)}; GR = {'L': 5, 'R': 11}
J = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]   # arm joints without the grippers


def runs(mask):
    """[(start, end)] of True runs with length >= MIN."""
    out, s = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and s is None: s = i
        if not m and s is not None:
            if i - s >= MIN: out.append((s, i))
            s = None
    return out


def locate(run):
    """(data dir, success.csv path or None). Some eval uploads nest the dataset in dataset/; others (v6) keep data at
    the top and put only success.csv in an otherwise empty dataset/."""
    base = f'{run}/dataset' if os.path.isdir(f'{run}/dataset/data') else run
    csv = next((c for c in (f'{base}/success.csv', f'{run}/success.csv', f'{run}/dataset/success.csv') if os.path.exists(c)), None)
    return base, csv


def holds(st, ac):
    """{'L'/'R': [(start, end)]} hold runs per arm."""
    return {k: runs((ac[:, g] < st[:, g] - GAP) & (st[:, g] > 4)) for k, g in GR.items()}


def score(st, ac, end=None):
    """start: seconds until the arms first move (> 5 units/s). idle: seconds standing still (whole 0.5 s windows) between
    the start and frame `end` = the task end (default: the last frame). dur: recorded seconds. grasps: tape holds, both arms."""
    move = np.abs(np.diff(st[:, J], axis=0)).max(1) * FPS > 5
    start = int(np.argmax(move)) if move.any() else len(st)
    end = len(st) - 1 if end is None else min(int(end), len(st) - 1)
    seg = ~move[start:end]
    still = np.convolve(seg, np.ones(MIN), 'valid') >= MIN if len(seg) >= MIN else np.zeros(0)
    return dict(start=start / FPS, idle=round(still.sum() / FPS, 1), dur=len(st) / FPS,
                grasps=sum(len(v) for v in holds(st, ac).values()))


if __name__ == '__main__':
    st = np.zeros((120, 12)); st[30:60, 0] = np.linspace(0, 60, 30); st[60:, 0] = 60   # moves from 1 s to 2 s, then waits
    r = score(st, st)
    assert r['start'] == 1.0 and r['dur'] == 4.0 and r['idle'] > 1.0 and r['grasps'] == 0, r
    assert score(st, st, end=60)['idle'] == 0.0   # the wait after the task end does not count
    print('score.py ok')
