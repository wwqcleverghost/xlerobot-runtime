"""Offline stage check: replay recorded eval episodes through the stage models exactly as the live eval feeds them
(clipped state + 3 cameras, first frame copied into the 2-step queue, a new chunk every n_action_steps frames) and compare
each model's own stage guess with fine9 recomputed offline from the same recording (the rule, no hand edits).
Writes <out>.csv: dataset, episode, frame, fine9, and one column per model (its stage, -1 = no guess yet).
--viewer: also <eval dataset dir>/stages_offline.csv (episode, frame, rule = fine9, pred = the first model) for the dataset
viewer's 阶段 / 进度 page (source "同一模型离线重放").
usage: PYTHONPATH=src:examples/xlerobot python examples/xlerobot/offline_stage_eval.py <out_prefix> <eval dataset dir> ...
       [--models stagepred] [--first_ep N] [--max_eps N] [--eps 2,3] [--duty 0.5] [--viewer]"""
import argparse
import time
from collections import deque
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.scripts.lerobot_record_eval import clip_state, train_state_range
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
import stage_labels

MODELS = Path("/home/marmotlab/wwq/lerobot/outputs/hf_models")
NAMES = {"head": "v6-flow_h16_s8_stagelabel_stagehead", "stagepred": "v6-flow_h16_s8_stagepred", "progpred": "v6-flow_h16_s8_progpred"}


def load(name, dev):
    p = MODELS / name
    pol = DiffusionPolicy.from_pretrained(p).to(dev).eval()
    pre, post = make_pre_post_processors(pol.config, pretrained_path=str(p),
                                         preprocessor_overrides={"device_processor": {"device": str(dev)}})
    return pol, pre, post, train_state_range(pre)


def frames(path, t0, n):
    """n RGB frames (HxWx3 uint8) of an episode starting at t0 s in an mp4; the last one repeated if the file is short."""
    last, k = None, 0
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(max(int((t0 - 1) / s.time_base), 0), stream=s)
        for fr in c.decode(s):
            if fr.time < t0 - 0.5 / 30:
                continue
            last = fr.to_ndarray(format="rgb24")
            yield last
            k += 1
            if k == n:
                return
    for _ in range(n - k):
        yield last


@torch.inference_mode()
def guess(kind, pol, post, q):
    """Stage guesses for the next n_action_steps frames from the 2-step queue q of preprocessed frames."""
    b = {OBS_STATE: torch.stack([f[OBS_STATE] for f in q], 1),
         OBS_IMAGES: torch.stack([torch.stack([f[k] for k in pol.config.image_features], -4) for f in q], 1)}
    with torch.autocast("cuda"):
        if kind == "head":
            b[OBS_ENV_STATE] = torch.zeros(1, 2, 9, device=b[OBS_STATE].device)  # not seen by the head
            _, feats = pol.diffusion._prepare_global_conditioning(b, obs=True)
            return [int(pol.diffusion.stage_head(feats[:, -1]).argmax(-1))] * pol.config.n_action_steps
        act = pol.diffusion.generate_actions(b)[0].float()
    extra = post(act)[:, 12:].cpu().numpy()  # (n_action_steps, extra), unnormalized
    if extra.shape[1] == 9:
        return [int(x) for x in extra.argmax(1)]
    B = stage_labels.PROG_STARTS
    return [min(max(int(np.searchsorted(B, p, side="right")) - 1, 0), 8) for p in extra[:, 0]]


def episode(root, meta, ep, models, duty):
    d = pd.read_parquet(root / "data", filters=[("episode_index", "==", ep)]).sort_values("frame_index")
    st, ac = np.stack(d[OBS_STATE].to_numpy())[:, :12], np.stack(d[ACTION].to_numpy())  # 17-dim sets: arms only
    n, e = len(d), meta.episodes[ep]
    t0 = lambda k: e[f"videos/{k}/from_timestamp"]
    vids = {k: frames(root / meta.get_video_file_path(ep, k), t0(k), n) for k in meta.video_keys}
    sizes = {s: stage_labels.video_sizes(root / meta.get_video_file_path(ep, f"observation.images.{c}"),
                                         t0(f"observation.images.{c}"), n, 30) for s, c in stage_labels.CAM.items()}
    out = {"fine9": stage_labels.fine9(st, ac, sizes)}
    pred = {k: np.full(n, -1) for k in models}
    prev = None
    for t in range(n):
        imgs = {k: next(vids[k]) for k in meta.video_keys}
        raw = (t, imgs)
        for k, (pol, pre, post, rng) in models.items():
            if t % pol.config.n_action_steps:
                continue
            # the live queue at a chunk start holds frames t-1 and t (frame 0 twice, like populate_queues)
            q = [pre(prepare_observation_for_inference(clip_state({OBS_STATE: st[i].astype(np.float32), **im}, rng),
                                                       torch.device("cuda"))) for i, im in (prev or raw, raw)]
            t1 = time.time()
            g = guess(k, pol, post, q)
            pred[k][t:t + len(g)] = g[: n - t]
            time.sleep((time.time() - t1) * (1 / duty - 1))  # leave the GPU to the live eval
        prev = raw
    out.update(pred)
    return pd.DataFrame({"episode": ep, "frame": np.arange(n), **out})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--max_eps", type=int, default=None)
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--models", default=",".join(NAMES), help="comma list of NAMES keys")
    ap.add_argument("--first_ep", type=int, default=0)
    ap.add_argument("--eps", default=None, help="comma list of episodes (one dir), instead of --first_ep/--max_eps")
    ap.add_argument("--viewer", action="store_true", help="write <dir>/stages_offline.csv")
    a = ap.parse_args()
    NAMES = {k: NAMES[k] for k in a.models.split(",")}
    models = {k: load(v, torch.device("cuda")) for k, v in NAMES.items()}
    rows = []
    for d in map(Path, a.dirs):
        meta = LeRobotDatasetMetadata("wwqcleverghost/" + d.name, root=d)
        eps = list(map(int, a.eps.split(","))) if a.eps else range(meta.total_episodes)[a.first_ep:][: a.max_eps]
        for ep in eps:
            t0 = time.time()
            df = episode(d, meta, ep, models, a.duty)
            df.insert(0, "dataset", d.name)
            rows.append(df)
            print(d.name, ep, len(df), f"{time.time() - t0:.0f}s",
                  {k: f"{(df[k] == df.fine9).mean():.2f}" for k in NAMES}, flush=True)
            pd.concat(rows).to_csv(a.out + ".csv", index=False)
            if a.viewer:  # rewritten per episode: the viewer sees what is done so far
                k = next(iter(NAMES))
                v = pd.concat(r for r in rows if r.dataset.iat[0] == d.name)
                v = v[["episode", "frame"]].assign(rule=v.fine9, pred=v[k])
                f = d / "stages_offline.csv"
                if f.exists():  # keep episodes of earlier runs (--first_ep)
                    old = pd.read_csv(f)
                    v = pd.concat([old[~old.episode.isin(v.episode)], v]).sort_values(["episode", "frame"])
                v.to_csv(f, index=False)
