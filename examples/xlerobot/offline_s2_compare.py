"""Offline closed-loop s2 check: replay demos of v6-handover-curated-cumprog through v6-flow_h16_s8_progself exactly as the
live eval runs it (lerobot_record_eval.ProgressMemory + predict_action; s starts at 0 and is fed back), with one progress
check (PROGRESS_S2) per run, and compare the model's memory s with the true cumulative progress of the demo
(observation.environment_state). Demos are training data (seen), so this measures drift of the feedback loop, not
generalization.
--trials <csv with ds, ep>: replay eval recordings instead (eval-<ds> in the HF cache, no true progress: true = NaN);
e.g. failed trials where the tape never moved, whose real progress cannot pass A抓住 (sum s <= 3).
Writes <out>.csv: [ds,] episode, frame, true (sum of true s, 0-9), s (sum of the memory s the model saw at that frame), and at
each new plan s1 (the previous plan's guess) / s2 (the check's correction), NaN elsewhere.
usage: PYTHONPATH=src:examples/xlerobot python examples/xlerobot/offline_s2_compare.py <out> --s2 none|head|<estimator dir>
       [--eps 3,20] [--max_eps N] [--trials list.csv]"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.scripts.lerobot_record_eval import ProgressMemory, clip_state
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from lerobot.utils.control_utils import predict_action
from offline_stage_eval import frames, load

CACHE = Path.home() / ".cache/huggingface/lerobot/wwqcleverghost"
DATA = CACHE / "v6-handover-curated-cumprog"
POLICY = "v6-flow_h16_s8_progself"


def eval_root(ds):
    r = CACHE / f"eval-{ds}"
    return r / "dataset" if (r / "dataset").exists() else r  # some eval sets keep the dataset one level down


def episode(root, meta, ep, pol, pre, post, rng, s2):
    d = pd.read_parquet(root / "data", filters=[("episode_index", "==", ep)]).sort_values("frame_index")
    st, n, e = np.stack(d[OBS_STATE].to_numpy()), len(d), meta.episodes[ep]
    true = np.stack(d[OBS_ENV_STATE].to_numpy()).sum(1) if OBS_ENV_STATE in d else np.full(n, np.nan)
    vids = {k: frames(root / meta.get_video_file_path(ep, k), e[f"videos/{k}/from_timestamp"], n) for k in meta.video_keys}
    torch.manual_seed(ep)  # same sampling noise for every s2 variant
    pol.reset(), pre.reset(), post.reset()
    mem = ProgressMemory(pol)
    if s2 != "none":
        mem.attach_check(pol, s2)
    s, s1, s2v = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
    for t in range(n):
        obs = clip_state({OBS_STATE: st[t].astype(np.float32), **{k: next(vids[k]) for k in meta.video_keys}}, rng)
        corrected = s2 != "none" and len(pol._queues[ACTION]) == 0 and mem.s1 is not None
        if corrected:
            s1[t] = mem.s1.sum()
        predict_action({**obs, OBS_ENV_STATE: mem()}, pol, torch.device("cuda"), pre, post, pol.config.use_amp)
        s[t] = mem.s.sum()  # correct() (inside predict_action) set mem.s = s2, the value the plan saw
        if corrected:
            s2v[t] = s[t]
        mem.update(pol)
    return pd.DataFrame({"episode": ep, "frame": np.arange(n), "true": true, "s": s, "s1": s1, "s2": s2v})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--s2", default="none")
    ap.add_argument("--eps", default=None)
    ap.add_argument("--max_eps", type=int, default=None)
    ap.add_argument("--trials", default=None)
    a = ap.parse_args()
    pol, pre, post, rng = load(POLICY, torch.device("cuda"))
    if a.trials:
        jobs = [(ds, eval_root(ds), int(ep)) for ds, ep in pd.read_csv(a.trials)[["ds", "ep"]].itertuples(index=False)]
    else:
        eps = list(map(int, a.eps.split(","))) if a.eps else range(LeRobotDatasetMetadata("x", root=DATA).total_episodes)[: a.max_eps]
        jobs = [(None, DATA, ep) for ep in eps]
    metas, rows = {}, []
    for ds, root, ep in jobs:
        t0 = time.time()
        if root not in metas:
            metas[root] = LeRobotDatasetMetadata("wwqcleverghost/" + root.name, root=root)
        df = episode(root, metas[root], ep, pol, pre, post, rng, a.s2)
        if ds:
            df.insert(0, "ds", ds)
        rows.append(df)
        print(ep, len(df), f"{time.time() - t0:.0f}s", f"mae {np.abs(df.s - df.true).mean():.2f}",
              f"end {df.s.iat[-1]:.2f}/{df.true.iat[-1]:.2f}", flush=True)
        pd.concat(rows).to_csv(a.out + ".csv", index=False)
