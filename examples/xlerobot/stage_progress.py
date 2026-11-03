"""Write the RA-BC progress table of a *-stagelabel dataset: <root>/sarm_progress.parquet (docs/workflows/stage-labels).

Progress is the SARM label built from our stage column instead of a SARM model: stage k covers [b_k, b_k + p_k) of 0..1, where
p_k = mean over episodes of (frames in stage k / episode frames), and rises linearly through each run of the stage. A slow run
rises slowly, so RA-BC (lerobot_train --use_rabc) gives it a small weight; a drop back to A前往 goes backwards and gets weight 0.
The stage column is fine9 with the hand-set starts, read from the dataset itself.

usage: python examples/xlerobot/stage_progress.py --root ~/wwq/data/v6-handover-curated-stagelabel [--horizon 16]
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from stage_labels import STAGES, segments

KEY = "observation.environment_state"


def progress(labs):
    """({episode: per-frame progress}, share of 0..1 per stage) from {episode: per-frame stage index}"""
    share = np.mean([np.bincount(lab, minlength=len(STAGES)) / len(lab) for lab in labs.values()], axis=0)
    start = np.r_[0, np.cumsum(share)[:-1]]
    out = {}
    for ep, lab in labs.items():
        p = np.empty(len(lab))
        for a, b, g in segments(lab):
            p[a:b] = start[g] + share[g] * np.arange(b - a) / (b - a)
        out[ep] = p
    return out, share


def selfcheck():
    labs = {0: np.repeat([0, 1, 2], [10, 10, 10]), 1: np.repeat([0, 1, 2, 0, 1, 2], [5, 5, 5, 5, 20, 20])}
    p, share = progress(labs)
    assert np.allclose(share[:3], [(1 / 3 + 10 / 60) / 2, (1 / 3 + 25 / 60) / 2, (1 / 3 + 25 / 60) / 2])
    assert np.all(np.diff(p[0]) > 0) and p[0][0] == 0 and p[0][-1] < 1
    assert p[1][15] == 0 and p[1][14] > 0.5  # the drop back to stage 0 starts over
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path)
    ap.add_argument("--horizon", type=int, default=16, help="policy horizon, only for the printed delta")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        raise SystemExit
    files = sorted(glob.glob(f"{args.root}/data/*/*.parquet"))
    df = pd.concat(pd.read_parquet(f, columns=["index", "episode_index", KEY]) for f in files).sort_values("index")
    labs = {ep: np.stack(g[KEY].to_numpy()).argmax(1) for ep, g in df.groupby("episode_index")}
    p, share = progress(labs)
    df["progress_sparse"] = np.concatenate([p[ep] for ep in sorted(labs)])
    df[["index", "episode_index", "progress_sparse"]].to_parquet(args.root / "sarm_progress.parquet", index=False)
    print("share per stage:", " ".join(f"{s} {x:.3f}" for s, x in zip(STAGES, share)))
    # A run at the average pace gains horizon / (mean episode frames) per chunk; --rabc_kappa at this value puts every
    # slower-than-average chunk below weight 1.
    delta = args.horizon / df.groupby("episode_index").size().mean()
    print(f"{len(labs)} episodes -> {args.root / 'sarm_progress.parquet'}; average-pace delta {delta:.4f}")
