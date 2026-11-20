"""Offline checks of v9 models (and the v6 ones for reference) on the eval trials held out of the v9 training set
(build_v9.py: episode % 5 == 4 of each eval set; labels in <v9 set>/v9_labels_holdout.npz). docs/workflows/v9.

1. stuck   frames of held-out FAILED trials that sit in A对位 / A抓住 (label stage 1-2) without moving on (label adv 0):
           of N sampled chunks, the share that closes the A gripper (command 8 deg below its reading within the 8 executed
           steps), whether any does, and the share when the chunk is picked by its own predicted progress gain (best of N).
           The bottleneck: the robot hovers there with the gripper open.
2. honest  every 20th frame of all held-out trials: the model's own progress at the planning frame (mean of N samples) vs the
           label v: mean abs error (stages); and how well its predicted gain over the chunk separates frames that move on
           (adv 1) from those that do not (AUC).
Progress models of v6 (progress_mode) get the label v as their input s. adv models are run at each --guidance.

usage: PYTHONPATH=src python examples/xlerobot/offline_v9.py --models <model dir or HF name> ... [--guidance 1 1.5 2 3]
       [--n 16] [--v9 <dir of v9-handover-curated-rollouts>] [--out outputs/v9/offline.csv]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE  # noqa: E402

CLOSE = 8.0  # deg below the gripper reading = the chunk closes the gripper
CAMS = ["observation.images.left_wrist", "observation.images.right_wrist", "observation.images.head(RGDB)"]


def auc(score, label):
    """P(score of a positive > score of a negative), ties count half."""
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_policy(name, dev):
    from huggingface_hub import snapshot_download

    path = Path(name) if Path(name).exists() else Path(snapshot_download(f"wwqcleverghost/{name}"))
    pol = DiffusionPolicy.from_pretrained(path).to(dev).eval()
    pre, post = make_pre_post_processors(pol.config, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": str(dev)}})
    return pol, pre, post


@torch.inference_mode()
def sample(pol, pre, post, items, v_label, n, dev):
    """items: dataset items (2 obs steps). Returns joints (B, n, 8, 12) deg, progress now / next (B, n) in stages (nan: none)."""
    keys = list(pol.config.image_features)
    raw = {k: torch.stack([it[k] for it in items]).repeat_interleave(n, 0) for k in [OBS_STATE, *keys]}
    raw = {k: v.float() / 255 if v.dtype == torch.uint8 else v for k, v in raw.items()}
    p = pre(raw)
    b = {OBS_STATE: p[OBS_STATE], OBS_IMAGES: torch.stack([p[k] for k in keys], -4)}
    c = pol.config
    if c.env_state_feature is not None:  # v6 progress models: the label as their memory s
        v = torch.as_tensor(np.repeat(v_label, n), device=dev, dtype=torch.float32)
        b[OBS_ENV_STATE] = pol.diffusion.cumulative(v, 9)[:, None].expand(-1, 2, -1)
    with torch.autocast(dev.type):
        act = pol.diffusion.generate_actions(b).float()
    joints = post(act)[..., :12].reshape(len(items), n, -1, 12).cpu().numpy()
    if c.progress_pred_dims or c.progress_mode:
        now, nxt = pol.diffusion.progress_now.float(), pol.diffusion.progress_next.float()
        if c.progress_mode in ("delta", "deltainc"):  # increments over the input s (= the label here)
            s = b[OBS_ENV_STATE][:, -1]
            now, nxt = now + s, nxt + s
        f = lambda x: x.clamp(0, 1).sum(-1).reshape(len(items), n).cpu().numpy()  # noqa: E731
        return joints, f(now), f(nxt)
    nan = np.full((len(items), n), np.nan)
    return joints, nan, nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--guidance", nargs="+", type=float, default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--per", type=int, default=8, help="stuck frames per failed trial")
    ap.add_argument("--every", type=int, default=20, help="progress frames: every n-th frame of each held-out trial")
    ap.add_argument("--v9", default=str(Path(HF_LEROBOT_HOME) / "wwqcleverghost/v9-handover-curated-rollouts"))
    ap.add_argument("--out", default="outputs/v9/offline.csv")
    args = ap.parse_args()
    dev = torch.device("cuda")
    lab = np.load(Path(args.v9) / "v9_labels_holdout.npz")
    trials = sorted({tuple(k.split("|")[:2]) for k in lab.files if k.count("|") == 2})
    rng = np.random.default_rng(0)
    picks = []  # (ds, ep, frame, kind)
    for ds, ep in trials:
        v, a, lb = lab[f"{ds}|{ep}|v"], lab[f"{ds}|{ep}|adv"], lab[f"{ds}|{ep}|lab"]
        ok = bool(lab[f"{ds}|{ep}|meta"][1])
        if not ok:
            stuck = np.flatnonzero(((lb == 1) | (lb == 2)) & (a == 0) & (np.arange(len(v)) > 30))
            for f in (rng.choice(stuck, min(args.per, len(stuck)), replace=False) if len(stuck) else []):
                picks.append((ds, int(ep), int(f), "stuck"))
        for f in range(15, len(v), args.every):
            picks.append((ds, int(ep), int(f), "honest"))
    print(f"{len(trials)} held-out trials; {sum(p[3] == 'stuck' for p in picks)} stuck frames, "
          f"{sum(p[3] == 'honest' for p in picks)} progress frames", flush=True)

    sets = {}

    def item(ds, ep, f):
        if ds not in sets:
            root = Path(HF_LEROBOT_HOME) / "wwqcleverghost" / ds
            root = root / "dataset" if (root / "dataset/meta/info.json").exists() else root
            d = LeRobotDataset(f"wwqcleverghost/{ds}", root=root, delta_timestamps={k: [-1 / 30, 0] for k in [OBS_STATE, *CAMS]})
            sets[ds] = (d, d.meta.episodes["dataset_from_index"])
        d, lo = sets[ds]
        it = d[int(lo[ep]) + f]
        # kept for the next models: images as uint8 (float32 full frames of ~3000 items would take ~60 GB)
        return {k: (it[k] * 255).round().to(torch.uint8) if k in CAMS else it[k] for k in [OBS_STATE, *CAMS]}

    cache = {}  # the dataset items, decoded once for all models
    rows = []
    for name in args.models:
        pol, pre, post = load_policy(name, dev)
        for g in (args.guidance if pol.config.adv_cond else [None]):
            if g is not None:
                pol.config.adv_guidance = g
            label = Path(name).name + ("" if g is None else f"@w{g:g}")
            for i in range(0, len(picks), 8):
                chunk = picks[i:i + 8]
                items = [cache.setdefault(p[:3], item(*p[:3])) for p in chunk]
                vl = np.array([lab[f"{ds}|{ep}|v"][f] for ds, ep, f, _ in chunk])
                joints, now, nxt = sample(pol, pre, post, items, vl, args.n, dev)
                for j, (ds, ep, f, kind) in enumerate(chunk):
                    st = items[j][OBS_STATE][-1].numpy()
                    A = 5 if st[5] > st[11] else 11
                    close = joints[j, :, :, A].min(1) < st[A] - CLOSE
                    gain = nxt[j] - now[j]
                    best = int(np.nanargmax(gain)) if np.isfinite(gain).any() else 0
                    rows.append(dict(model=label, ds=ds, ep=ep, frame=f, kind=kind, v=float(vl[j]),
                                     adv=int(lab[f"{ds}|{ep}|adv"][f]), close=float(close.mean()), any_close=bool(close.any()),
                                     best_close=bool(close[best]), prog=float(np.nanmean(now[j])) if np.isfinite(now[j]).any() else np.nan,
                                     gain=float(np.nanmean(gain)) if np.isfinite(gain).any() else np.nan))
                if i % 800 == 0:
                    print(label, i, "/", len(picks), flush=True)
        del pol
        torch.cuda.empty_cache()
    r = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(args.out, index=False)
    out = []
    for m, g in r.groupby("model", sort=False):
        s, h = g[g.kind == "stuck"], g[g.kind == "honest"]
        out.append(dict(model=m, stuck_close=s.close.mean(), stuck_any=s.any_close.mean(), stuck_best=s.best_close.mean(),
                        prog_mae=(h.prog - h.v).abs().mean(), gain_auc=auc(h.gain.to_numpy(), h.adv.to_numpy())))
    t = pd.DataFrame(out).round(3)
    print(t.to_string(index=False))
    Path(args.out).with_suffix(".txt").write_text(t.to_string(index=False) + "\n" + json.dumps(vars(args)) + "\n")


if __name__ == "__main__":
    main()
