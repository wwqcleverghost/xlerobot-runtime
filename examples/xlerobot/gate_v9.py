"""Intervention gates of the v9 models (docs/workflows/v9, docs/notes/v9_design_2026-11-20): does a conditioning signal really
steer the action? Same observation, same noise, only the condition changed. Lesson of the first progress-feedback batch: its
progress input moved the joints no more than a new noise seed did (0.1-0.6 deg), so it was ignored.

Frames: the held-out failed trials (build_v9.py: episode % 5 == 4) where the arm sits in A对位 / A抓住 without moving on (label adv 0).
Thresholds are fixed here BEFORE the results (pre-registered); do not tune them after looking.

  G1 adv     good vs bad indicator: effect / noise >= 3, close(good) - close(bad) >= 0.10, close(good, w=2) >= 0.22 (2 x the v6
             baseline's 0.112 on the same frames)
  G2 back    back vs good: effect / noise >= 3, and the A arm moves toward its start pose within the 8 executed steps in >= 70%
             of the samples with back
  G5 best-of-N  within a frame, corr(predicted progress gain, lowest A-gripper command) <= -0.2 (more gain goes with closing), and
             best-of-16 by gain closes >= 1.5 x as often as a random sample
effect = mean |joint command difference| (deg, 8 executed steps x 12 joints) between the two conditions with the same noise;
noise = the same between two noise draws with the same condition. G3 (negmask) and G4 (progress honesty) come from offline_v9.py.

usage: PYTHONPATH=src python examples/xlerobot/gate_v9.py --model <dir or HF name> [--v9 <set dir>] [--out outputs/v9/gate_<name>.json]
       PYTHONPATH=src python examples/xlerobot/gate_v9.py --selfcheck
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

EFFECT_RATIO = 3.0
G1_CLOSE_GAP, G1_CLOSE_W2 = 0.10, 0.22
G2_TOWARD = 0.70
G5_CORR, G5_LIFT = -0.2, 1.5
N = 16
CLOSE = 8.0


def effect_ratio(a, b, a2):
    """a, b: (F, N, 8, 12) joints under two conditions, same noise; a2: condition of a with other noise. -> (effect, noise, ratio)"""
    eff, noi = float(np.abs(a - b).mean()), float(np.abs(a - a2).mean())
    return eff, noi, eff / max(noi, 1e-9)


def a_grip(st):
    """index of the A gripper (the more open one now) per frame."""
    return np.where(st[:, 5] > st[:, 11], 5, 11)


def closes(j, st):
    """(F, N) bool: the A gripper is commanded CLOSE deg below its reading within the chunk."""
    A = a_grip(st)
    g = j[np.arange(len(j)), :, :, A].min(-1)  # (F, N)
    return g < st[np.arange(len(st)), A][:, None] - CLOSE


def toward_start(j, st, start):
    """(F, N) bool: the A arm's 5 joints (no gripper) move toward the trial's start pose over the 8 steps."""
    idx = (a_grip(st) - 5)[:, None] + np.arange(5)[None]  # 0..4 (left) or 6..10 (right)
    now = np.take_along_axis(st, idx, 1)
    goal = np.take_along_axis(start, idx, 1) - now  # (F, 5)
    end = j[:, :, -1]  # (F, N, 12) last executed command
    mv = np.take_along_axis(end, np.broadcast_to(idx[:, None], (len(j), j.shape[1], 5)), 2) - now[:, None]
    return (mv * goal[:, None]).sum(-1) > 0


@torch.inference_mode()
def run(pol, pre, post, items, cmd, w, noise):
    """joints (F, N, 8, 12) and progress gain (F, N) for indicator cmd (None: no adv) and guidance w, fixed noise (F*N, H, D)."""
    from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

    keys = list(pol.config.image_features)
    raw = {k: torch.stack([it[k] for it in items]).repeat_interleave(N, 0) for k in [OBS_STATE, *keys]}
    raw = {k: (v.float() / 255 if v.dtype == torch.uint8 else v).to("cuda") for k, v in raw.items()}
    p = pre(raw)
    b = {OBS_STATE: p[OBS_STATE], OBS_IMAGES: torch.stack([p[k] for k in keys], -4)}
    if pol.config.env_state_feature is not None:  # v6 progress models: the label progress v as their memory s
        v = torch.tensor([float(it["v"]) for it in items], device="cuda").repeat_interleave(N)
        b[OBS_ENV_STATE] = pol.diffusion.cumulative(v, 9)[:, None].expand(-1, 2, -1)
    if cmd is not None:
        pol.diffusion.adv_command, pol.config.adv_guidance = cmd, w
    with torch.autocast("cuda"):
        act = pol.diffusion.generate_actions(b, noise=noise).float()
    j = post(act)[..., :12].reshape(len(items), N, -1, 12).cpu().numpy()
    gain = np.full((len(items), N), np.nan)
    if pol.config.progress_pred_dims or pol.config.progress_mode:
        d = pol.diffusion
        gain = (d.progress_next.float().clamp(0, 1).sum(-1) - d.progress_now.float().clamp(0, 1).sum(-1)).reshape(len(items), N).cpu().numpy()
    return j, gain


def gates(res):
    """pass / fail per gate from the measured numbers (also used by the selfcheck)."""
    out = {}
    if "g1_ratio" in res:
        out["G1"] = bool(res["g1_ratio"] >= EFFECT_RATIO and res["close_good"] - res["close_bad"] >= G1_CLOSE_GAP
                         and res["close_good_w2"] >= G1_CLOSE_W2)
    if "g2_ratio" in res:
        out["G2"] = bool(res["g2_ratio"] >= EFFECT_RATIO and res["toward_back"] >= G2_TOWARD)
    if "g5_corr" in res:
        out["G5"] = bool(res["g5_corr"] <= G5_CORR and res["best_close"] >= G5_LIFT * max(res["random_close"], 1e-9))
    return out


def measure(pol, pre, post, items, start, seed=0):
    from lerobot.utils.constants import OBS_STATE

    c = pol.config
    st = np.stack([it[OBS_STATE][-1].numpy() for it in items])
    g = torch.Generator("cuda").manual_seed(seed)
    shape = (len(items) * N, c.horizon, c.action_feature.shape[0])
    n1 = torch.randn(shape, device="cuda", generator=g)
    n2 = torch.randn(shape, device="cuda", generator=g)
    res = {}
    if c.adv_cond:
        good, gg = run(pol, pre, post, items, 0, 1.0, n1)
        good2, _ = run(pol, pre, post, items, 0, 1.0, n2)
        bad, _ = run(pol, pre, post, items, 1, 1.0, n1)
        w2, _ = run(pol, pre, post, items, 0, 2.0, n1)
        res["g1_effect"], res["g1_noise"], res["g1_ratio"] = effect_ratio(good, bad, good2)
        res["close_good"], res["close_bad"], res["close_good_w2"] = (float(closes(x, st).mean()) for x in (good, bad, w2))
        res["toward_good"] = float(toward_start(good, st, start).mean())
        if c.adv_back:
            back, _ = run(pol, pre, post, items, 2, 1.0, n1)
            res["g2_effect"], res["g2_noise"], res["g2_ratio"] = effect_ratio(back, good, good2)
            res["toward_back"] = float(toward_start(back, st, start).mean())
        pol.diffusion.adv_command, pol.config.adv_guidance = 0, 1.0
        base, gain = good, gg
    else:
        base, gain = run(pol, pre, post, items, None, 1.0, n1)
        base2, _ = run(pol, pre, post, items, None, 1.0, n2)
        res["seed_noise"] = float(np.abs(base - base2).mean())
        res["close_plain"] = float(closes(base, st).mean())
        res["toward_plain"] = float(toward_start(base, st, start).mean())
    if np.isfinite(gain).all():
        cl = closes(base, st)
        gmin = base[np.arange(len(base)), :, :, a_grip(st)].min(-1)
        cs = [np.corrcoef(gain[f], gmin[f])[0, 1] for f in range(len(gain)) if gain[f].std() > 0 and gmin[f].std() > 0]
        res["g5_corr"] = float(np.mean(cs)) if cs else float("nan")
        res["best_close"] = float(cl[np.arange(len(cl)), gain.argmax(1)].mean())
        res["random_close"] = float(cl.mean())
    return res


def main(args):
    import offline_v9
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME, OBS_STATE

    lab = np.load(Path(args.v9) / "v9_labels_holdout.npz")
    trials = sorted({tuple(k.split("|")[:2]) for k in lab.files if k.count("|") == 2})
    rng = np.random.default_rng(0)
    picks = []
    for ds, ep in trials:
        if bool(lab[f"{ds}|{ep}|meta"][1]):
            continue
        lb, a = lab[f"{ds}|{ep}|lab"], lab[f"{ds}|{ep}|adv"]
        stuck = np.flatnonzero(((lb == 1) | (lb == 2)) & (a == 0) & (np.arange(len(lb)) > 30))
        picks += [(ds, int(ep), int(f)) for f in (rng.choice(stuck, min(8, len(stuck)), replace=False) if len(stuck) else [])]
    sets, items, start = {}, [], []
    for ds, ep, f in picks:
        if ds not in sets:
            root = Path(HF_LEROBOT_HOME) / "wwqcleverghost" / ds
            root = root / "dataset" if (root / "dataset/meta/info.json").exists() else root
            d = LeRobotDataset(f"wwqcleverghost/{ds}", root=root, delta_timestamps={k: [-1 / 30, 0] for k in [OBS_STATE, *offline_v9.CAMS]})
            sets[ds] = (d, d.meta.episodes["dataset_from_index"])
        d, lo = sets[ds]
        it = d[int(lo[ep]) + f]
        items.append({**{k: (it[k] * 255).round().to(torch.uint8) if k in offline_v9.CAMS else it[k] for k in [OBS_STATE, *offline_v9.CAMS]},
                      "v": float(lab[f"{ds}|{ep}|v"][f])})
        start.append(d.hf_dataset[int(lo[ep])][OBS_STATE].numpy())
    start = np.stack(start)
    print(f"{len(items)} stuck frames from {len({(d, e) for d, e, _ in picks})} held-out failed trials", flush=True)
    pol, pre, post = offline_v9.load_policy(args.model, torch.device("cuda"))
    res = {}
    for i in range(0, len(items), args.batch):  # per batch of frames (x N samples), then averaged (frames weigh equally)
        r = measure(pol, pre, post, items[i:i + args.batch], start[i:i + args.batch], seed=i)
        for k, v in r.items():
            res.setdefault(k, []).append(v)
    res = {k: float(np.nanmean(v)) for k, v in res.items()}
    res = {**{k: round(v, 4) for k, v in res.items()}, "frames": len(items), "gates": gates(res), "model": str(args.model)}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    out = Path(args.out or f"outputs/v9/gate_{Path(args.model).name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))


def selfcheck():
    """A model that ignores its indicator must fail G1 / G2; the same model with the indicator wired in must show an effect."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE

    torch.manual_seed(0)
    cfg = DiffusionConfig(input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (12,)),
                                          "observation.images.cam": PolicyFeature(FeatureType.VISUAL, (3, 32, 32))},
                          output_features={ACTION: PolicyFeature(FeatureType.ACTION, (21,))}, crop_shape=None,
                          pretrained_backbone_weights=None, horizon=16, n_action_steps=8, noise_scheduler_type="FLOW",
                          down_dims=(32, 64), progress_pred_dims=9, adv_cond=True, adv_back=True, device="cuda")
    ident = lambda x: x  # noqa: E731
    items = [{OBS_STATE: torch.rand(2, 12) * 30, "observation.images.cam": torch.rand(2, 3, 32, 32)} for _ in range(4)]
    start = np.zeros((4, 12))
    pol = DiffusionPolicy(cfg).cuda().eval()
    blocks = [m for m in pol.diffusion.unet.modules() if hasattr(m, "cond_encoder")]
    for blk in blocks:  # make the indicator count: large weights on its 3 inputs (the last columns of the FiLM layers)
        blk.cond_encoder[1].weight.data[:, -3:] *= 200
    live = measure(pol, ident, ident, items, start)
    for blk in blocks:  # the same model blind to its indicator
        blk.cond_encoder[1].weight.data[:, -3:] = 0
    blind = measure(pol, ident, ident, items, start)
    assert blind["g1_ratio"] < 0.01 and blind["g2_ratio"] < 0.01, blind
    assert not gates(blind)["G1"] and not gates(blind)["G2"], gates(blind)
    assert live["g1_ratio"] > 10 * max(blind["g1_ratio"], 1e-6), (live["g1_ratio"], blind["g1_ratio"])
    # toward_start: a command that steps the A arm to its start pose counts, one that steps away does not
    st = np.c_[np.full((1, 5), 10.0), [[30.0]], np.zeros((1, 6))]
    j = np.zeros((1, 2, 8, 12))
    j[0, 0, :, :5], j[0, 1, :, :5] = 9.0, 11.0
    assert toward_start(j, st, np.zeros((1, 12))).tolist() == [[True, False]]
    print(f"selfcheck ok: wired effect/noise {live['g1_ratio']:.2f}, blind {blind['g1_ratio']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--v9", default=None)
    ap.add_argument("--out")
    ap.add_argument("--batch", type=int, default=2, help="frames per forward pass (x 16 samples); 2 fits next to a training run")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
    else:
        if a.v9 is None:
            from lerobot.utils.constants import HF_LEROBOT_HOME

            a.v9 = str(Path(HF_LEROBOT_HOME) / "wwqcleverghost/v9-handover-curated-rollouts")
        main(a)
