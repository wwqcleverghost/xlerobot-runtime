"""Copy a *-stagelabel dataset with the stage appended to the action, so the policy also predicts it (docs/workflows/stage-labels).

--kind stage:    action += fine9 stage one-hot (9 dims)          -> <name>-stagepred
--kind progress: action += stage progress 0..1 (1 dim, the RA-BC progress of stage_progress.py) -> <name>-progpred
The extra dims come last, so live eval still sends only the robot's joints (make_robot_action maps by the robot's names) and
the model needs no change; train with --dataset.drop_names=[observation.environment_state] so the stage is a target, not an input.
data/ and meta/ are rewritten copies, videos/ are hard links.

usage: PYTHONPATH=src python examples/xlerobot/add_stage_target.py --dataset v6-handover-curated-stagelabel --kind progress
"""
import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from stage_labels import STAGES  # noqa: E402
from stage_progress import progress  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.constants import HF_LEROBOT_HOME  # noqa: E402

KEY = "observation.environment_state"
OWNER = "wwqcleverghost"


def main(src, kind):
    root = HF_LEROBOT_HOME / OWNER / src
    dst = src.removesuffix("-stagelabel") + {"stage": "-stagepred", "progress": "-progpred"}[kind]
    new_root = root.parent / dst
    assert not new_root.exists(), f"{dst} already exists"
    files = sorted(glob.glob(str(root / "data/*/*.parquet")))
    dfs = [pd.read_parquet(f) for f in files]
    allf = pd.concat(dfs)
    labs = {ep: np.stack(g.sort_values("frame_index")[KEY].to_numpy()).argmax(1) for ep, g in allf.groupby("episode_index")}
    if kind == "stage":
        extra, names = {ep: np.eye(len(STAGES), dtype=np.float32)[lab] for ep, lab in labs.items()}, [f"stage.{s}" for s in STAGES]
    else:
        extra, names = {ep: p[:, None].astype(np.float32) for ep, p in progress(labs)[0].items()}, ["stage.progress"]
    shutil.copytree(root / "videos", new_root / "videos", copy_function=os.link)
    shutil.copytree(root / "meta", new_root / "meta")
    for f, df in zip(files, dfs):
        df["action"] = [np.r_[a, extra[int(e)][int(i)]].astype(np.float32) for a, e, i in zip(df["action"], df["episode_index"], df["frame_index"])]
        out = new_root / Path(f).relative_to(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    info = json.loads((root / "meta/info.json").read_text())
    n_act = info["features"]["action"]["shape"][0]
    info["features"]["action"]["shape"] = [n_act + len(names)]
    info["features"]["action"]["names"] += names
    (new_root / "meta/info.json").write_text(json.dumps(info, indent=4))
    allv = np.concatenate([extra[ep] for ep in sorted(extra)])
    stats = json.loads((root / "meta/stats.json").read_text())
    for stat, value in stats["action"].items():
        if stat == "count":
            continue
        q = {"min": allv.min(0), "max": allv.max(0), "mean": allv.mean(0), "std": allv.std(0)}.get(stat)
        if q is None:  # q01 .. q99
            q = np.quantile(allv, int(stat[1:]) / 100, axis=0)
        stats["action"][stat] = list(value) + q.tolist()
    (new_root / "meta/stats.json").write_text(json.dumps(stats, indent=4))
    # check: the action keeps its joints and carries the extra dims, a frame (with video) still loads
    new = LeRobotDataset(f"{OWNER}/{dst}", root=new_root)
    act = np.stack(new.hf_dataset["action"])
    old = np.stack(LeRobotDataset(f"{OWNER}/{src}", root=root).hf_dataset["action"])
    want = np.stack([extra[int(e)][int(i)] for e, i in zip(new.hf_dataset["episode_index"], new.hf_dataset["frame_index"])])
    assert np.array_equal(act[:, :n_act], old) and np.array_equal(act[:, n_act:], want)
    assert new[len(new) // 2]["action"].shape[-1] == n_act + len(names)
    print(f"{src} -> {dst}: {new.meta.total_episodes} episodes, {new.meta.total_frames} frames, action {n_act} + {names}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--kind", choices=["stage", "progress"], required=True)
    x = ap.parse_args()
    main(x.dataset, x.kind)
