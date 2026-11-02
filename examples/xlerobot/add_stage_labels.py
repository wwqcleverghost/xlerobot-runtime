"""Add per-frame stage labels to a local dataset as observation.environment_state, in place, and rename it <name>-stagelabel
(definition: docs/workflows/stage-labels/README.md).

Training reads model inputs only from the dataset's own files, so the label has to be a column there. The label is the causal
rule fine9 (what the robot computes live), one-hot over stage_labels.STAGES (9 floats). Only the data parquet files and
meta/info.json + meta/stats.json change (videos are untouched); they are backed up first to <owner>/_trash_<date>/.
Diffusion does not normalise ENV features, so the one-hot reaches the model as is; stats are written anyway for tools that need them.

usage: PYTHONPATH=src python examples/xlerobot/add_stage_labels.py --dataset v8-handover2 [--dst v8-handover2-stagelabel]
       PYTHONPATH=src python examples/xlerobot/add_stage_labels.py --dataset v8-handover2-stagelabel   # relabel after a rule change
"""
import argparse
import glob
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "docs/workflows/dataset-viewer"), str(Path(__file__).parent)]
import server  # noqa: E402
import stage_labels  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

KEY = "observation.environment_state"


def labels(ds_id):
    """{episode: (n, 9) float32 one-hot of fine9}"""
    o, out = server.load(ds_id), {}
    for ep in o["eps"].index:
        a, s = server.ep_arrays(o, ep)
        out[int(ep)] = np.eye(len(stage_labels.STAGES), dtype=np.float32)[stage_labels.fine9(s, a, server.tape_sizes(ds_id, ep))]
    return out


def main(src, dst):
    """Add the column and rename src -> dst; when src already has it (a rule changed), recompute it in place (dst = src)."""
    root = server.OWNER / src
    info = json.loads((root / "meta/info.json").read_text())
    dst = src if KEY in info["features"] else dst
    new_root = server.OWNER / dst
    assert dst == src or not new_root.exists(), f"{dst} already exists"
    lab = labels(src)
    names = list(stage_labels.STAGES)
    backup = server.OWNER / f"_trash_{time.strftime('%Y%m%d')}" / f"{src}.before-stagelabel-{time.strftime('%H%M%S')}"
    for d in ("data", "meta"):
        shutil.copytree(root / d, backup / d)
    for f in sorted(glob.glob(str(root / "data/*/*.parquet"))):
        df = pd.read_parquet(f)
        df[KEY] = [lab[int(e)][int(i)] for e, i in zip(df["episode_index"], df["frame_index"])]
        df.to_parquet(f, index=False)
    info["features"][KEY] = {"dtype": "float32", "shape": [len(names)], "names": names}
    (root / "meta/info.json").write_text(json.dumps(info, indent=4))
    allv = np.concatenate([lab[ep] for ep in sorted(lab)])
    stats = json.loads((root / "meta/stats.json").read_text())
    stats[KEY] = {"min": allv.min(0).tolist(), "max": allv.max(0).tolist(), "mean": allv.mean(0).tolist(),
                  "std": allv.std(0).tolist(), "count": [len(allv)]}
    (root / "meta/stats.json").write_text(json.dumps(stats, indent=4))
    if new_root != root:
        root.rename(new_root)
    # check: same frames, every frame carries exactly the label computed above, a frame (with video) still loads
    new = LeRobotDataset(f"wwqcleverghost/{dst}", root=new_root)
    assert new.meta.total_frames == len(allv) and KEY in new.meta.features
    col = np.stack(new.hf_dataset[KEY])
    want = np.stack([lab[int(e)][int(i)] for e, i in zip(new.hf_dataset["episode_index"], new.hf_dataset["frame_index"])])
    assert np.array_equal(col, want) and (col.sum(1) == 1).all()
    item = new[len(new) // 2]
    assert item[KEY].shape[-1] == len(names) and any(k.startswith("observation.images.") for k in item)
    print(f"{src} -> {dst} (fine9 at git {server.git_info()}): {new.meta.total_episodes} episodes, {new.meta.total_frames} frames, {KEY} = one-hot of {names}")
    print("frames per stage:", dict(zip(names, allv.sum(0).astype(int).tolist())))
    print("backup of data/ and meta/:", backup)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dst")
    x = ap.parse_args()
    main(x.dataset, x.dst or f"{x.dataset}-stagelabel")
