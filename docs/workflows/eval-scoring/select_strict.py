"""Very strict pick of eval trials to train on: only trials that are as good as the human demos on every measure.
A trial is kept only if ALL hold:
  1. success: human label 1 AND all 5 auto stages (score.py) pass
  2. no regrasp: at most 2 gripper closes per arm (the hold count is not used: it splits one grasp in two often,
     also in the human demos)
  3. report.py quality sub-scores all 1.0: jitter (reversals/s) and task time at or below the human median, no idle
  4. curation checklist (handover-dataset-curation/README.md): tape visible on the table in the last head frame,
     first holding arm follows the midline rule (L x<=0.49, R x>=0.51)
Tape standing at the start cannot be told reliably from one frame: check the kept trials by eye (sheets.py).

usage: PYTHONPATH=src:docs/workflows/eval-scoring:docs/workflows/handover-dataset-curation:examples/xlerobot \
  python select_strict.py <out_prefix> <ref_demo_dir> <eval_dir> ...   -> <out_prefix>.csv, <out_prefix>_spec.json
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from report import trials, quality
from score import score, runs, GAP
from ep_score import yellow, closes, HEAD

MIN_TAPE_PX = 500  # last head frame; the basket episodes of handover2 have 0


def check(path, ref):
    base = f"{path}/dataset" if os.path.isdir(f"{path}/dataset") else path
    df, _ = trials(path)
    _, sub = quality(df, ref)
    d = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/data/*/*.parquet")))
    ds, rows = None, []
    for (ep, g), (_, t), (_, q) in zip(d.groupby("episode_index"), df.iterrows(), sub.iterrows()):
        st, ac = np.stack(g["observation.state"].values), np.stack(g["action"].values)
        r = score(st, ac)
        row = dict(run=os.path.basename(path), ep=ep, human=t.success, stages="".join(map(str, r["stage"])),
                   holds=r["grasps"], closesL=closes(ac[:, 5]), closesR=closes(ac[:, 11]),
                   smooth=round(q.smooth, 2), speed=round(q.speed, 2), idle_frac=round(t.idle_frac, 3),
                   rev_rate=round(t.rev_rate, 2), task_s=round(t.task_s, 1))
        row["ok_success"] = bool(t.success and all(r["stage"]))
        row["ok_grasp"] = row["closesL"] <= 2 and row["closesR"] <= 2
        row["ok_quality"] = bool(q.smooth == 1 and q.speed == 1 and t.idle_frac == 0)
        row["ok_vision"] = False
        if row["ok_success"] and row["ok_grasp"] and row["ok_quality"]:  # decode video only for the survivors
            ds = ds or LeRobotDataset(os.path.basename(base), root=base)
            lo, hi = int(ds.meta.episodes["dataset_from_index"][ep]), int(ds.meta.episodes["dataset_to_index"][ep])
            img = lambda i: (ds[i][HEAD].permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # noqa: E731
            px_end, x0 = yellow(img(hi - 1))[0], yellow(img(lo))[1] / 640
            hold = {k: runs((ac[:, gi] < st[:, gi] - GAP) & (st[:, gi] > 4)) for k, gi in (("L", 5), ("R", 11))}
            arm = min((v[0][0], k) for k, v in hold.items() if v)[1]
            row.update(tape_px_end=px_end, tape_x_start=round(x0, 2), first_arm=arm)
            row["ok_vision"] = px_end >= MIN_TAPE_PX and ((arm == "L" and 0 <= x0 <= 0.49) or (arm == "R" and x0 >= 0.51))
        row["keep"] = row["ok_success"] and row["ok_grasp"] and row["ok_quality"] and row["ok_vision"]
        rows.append(row)
    return pd.DataFrame(rows), base


if __name__ == "__main__":
    out, ref_dir, dirs = sys.argv[1], sys.argv[2].rstrip("/"), [d.rstrip("/") for d in sys.argv[3:]]
    ref_df, _ = trials(ref_dir)
    ref = {"rev_rate": ref_df.rev_rate.median(), "task_s": ref_df.task_s.median()}
    print(f"reference {os.path.basename(ref_dir)}: {ref['rev_rate']:.2f} reversals/s, task {ref['task_s']:.1f} s")
    res, spec = [], []
    for p in dirs:
        df, base = check(p, ref)
        res.append(df)
        if df.keep.any():
            spec.append({"repo": f"local/{os.path.basename(p)}", "root": base, "note": "select_strict.py",
                         "drop": [int(e) for e in df.ep[~df.keep]]})
    res = pd.concat(res)
    res.to_csv(f"{out}.csv", index=False)
    json.dump(spec, open(f"{out}_spec.json", "w"), indent=2)
    s = res[res.human == 1]
    print(f"successful trials {len(s)}; + all auto stages {s.ok_success.sum()}, + no regrasp "
          f"{(s.ok_success & s.ok_grasp).sum()}, + human-level quality {(s.ok_success & s.ok_grasp & s.ok_quality).sum()}, "
          f"+ placement/midline {s.keep.sum()}")
    print(res[res.keep][["run", "ep", "rev_rate", "task_s", "tape_px_end", "tape_x_start", "first_arm"]].to_string(index=False))
