"""Build the v9 training set: the 58 v6-handover-curated demos + real-robot eval rollouts of the v6 models, with an
honest per-frame progress and an advantage label per frame (docs/workflows/v9).

Per source frame (step --label):
  stage   fine9 (stage_labels.py, hand-set starts for the demos); a failed trial whose note says "<stage>失败" never goes
          past that stage (the rule is too lenient on evals: it saw A抓住 where the reviewer saw A对位失败)
  p       within-stage progress: demos = the road measure of stage_progress_cum.py; rollouts = the mean road progress of the
          K nearest demo frames of the same stage (A arm first, joints / demo std, 5-frame smoothed). A rollout that hovers
          next to the tape keeps the progress of the demo frames it looks like, so its progress stays flat.
  v       stage + p (0..9)
  gain    best v within the next H frames - v
  adv     1 = gain >= eps[stage], eps = the 30% quantile of the demos' gain in that stage (RECAP's choice); 0 = slower
          (a hover, a stall, a failure's end). neg = rollout frame with adv 0.
Rollout frames kept: [0, task end) of episode_end.csv; a run of adv 0 longer than CAP frames keeps its first CAP frames and
its last TAIL ones (a trial's final stall only the first CAP). Every 5th trial of each eval set (episode % 5 == 4) is
held out for the offline checks.

usage (server2: HOME with .cache/huggingface/lerobot -> $HF_HOME/lerobot, PYTHONPATH=src, docs/workflows/v9):
  python examples/xlerobot/build_v9.py --download
  python examples/xlerobot/build_v9.py --label            # -> outputs/v9/{spec.json,labels.npz,holdout.json,summary.txt}
  python examples/xlerobot/trim_stagnant_frames.py --spec outputs/v9/spec.json --dst wwqcleverghost/raw-v9-build
  python examples/xlerobot/build_v9.py --finish           # -> wwqcleverghost/v9-handover-curated-rollouts
  python examples/xlerobot/build_v9.py --selfcheck
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "docs/workflows/dataset-viewer"), str(Path(__file__).parent)]

OWNER = "wwqcleverghost"
DEMOS = "v6-handover-curated"
EVALS = ["eval-v6-flow_h16_s8", "eval-v6-flow_h16_s8_progdelta", "eval-v6-flow_h16_s8_progdelta_rabc",
         "eval-v6-flow_h16_s8_progpred", "eval-v6-flow_h16_s8_progself", "eval-v6-flow_h16_s8_progself_s2head",
         "eval-v6-flow_h16_s8_progself_s2mix35", "eval-v6-flow_h16_s8_rabc", "eval-v6-flow_h16_s8_stagelabel",
         "eval-v6-flow_h16_s8_stagelabel_stagehead", "eval-v6-flow_h16_s8_stagepred"]
# notes of the baseline's first 30 trials were entered under the viewer id of that subset
NOTE_ALIAS = {"eval-v6-flow_h16_s8": "eval-v6-flow_h16_s8_first30"}
RAW, FINAL = "raw-v9-build", "v9-handover-curated-rollouts"
OUT = REPO / "outputs/v9"
K = 10       # nearest demo frames for a rollout frame's progress
H = 30       # look-ahead frames of the gain (1 s at 30 fps; evals really run ~25 fps)
Q = 0.3      # eps = this quantile of the demos' gain per stage
EPS_MIN = 0.02
CAP, TAIL = 150, 15


def lerobot_home():
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return Path(HF_LEROBOT_HOME) / OWNER


def download():
    from huggingface_hub import HfApi, snapshot_download

    api, revs = HfApi(), {}
    for name in [DEMOS, *EVALS]:
        rev = api.dataset_info(f"{OWNER}/{name}").sha
        snapshot_download(f"{OWNER}/{name}", repo_type="dataset", revision=rev, local_dir=lerobot_home() / name)
        revs[name] = rev
        print(name, rev[:8], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sources.json").write_text(json.dumps(revs, indent=1))


def a_is_left(st, lab):
    """A = the arm that opens to grasp first: more open gripper over A对位 / A抓住; else the arm B is not."""
    import stage_progress_cum

    m = (lab == 1) | (lab == 2)
    if m.sum() >= 5:
        return st[m, 5].mean() > st[m, 11].mean()
    return not stage_progress_cum.b_is_left(st, lab)


def features(st, std, a_left):
    import stage_progress_cum

    x = st / std
    if not a_left:  # order A arm, B arm
        x = np.r_["1", x[:, 6:], x[:, :6]]
    return stage_progress_cum.smooth(x)


def knn_mean(q, ref, val, k):
    """mean of val over the k nearest rows of ref for every row of q (chunked brute force)."""
    out, sq = np.empty(len(q)), (ref * ref).sum(1)
    k = min(k, len(ref))
    for i in range(0, len(q), 4096):
        d2 = (q[i:i + 4096] ** 2).sum(1)[:, None] + sq[None] - 2 * q[i:i + 4096] @ ref.T
        j = np.argpartition(d2, k - 1, 1)[:, :k]
        out[i:i + 4096] = val[j].mean(1)
    return out


def gain(v, h=H):
    """best v within the next h frames minus v (the last frame: 0)."""
    n = len(v)
    best = np.array([v[t + 1:t + 1 + h].max() if t + 1 < n else v[t] for t in range(n)])
    return best - v


def note_stage(note):
    """'A对位失败' / 'a对位失败，...' -> stage index of the failed stage, else None."""
    from stage_labels import STAGES

    note = (note or "").strip().upper()
    if "失败" not in note:
        return None
    head = note.split("失败")[0]
    return next((i for i, s in enumerate(STAGES) if s.upper() == head), None)


def runs(mask):
    """[(a, b)] runs of True."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def keep_mask(adv, cap=CAP, tail=TAIL):
    """frames kept of a rollout (already cut at the task end): long adv-0 runs shortened (see module doc)."""
    keep = np.ones(len(adv), bool)
    for a, b in runs(adv == 0):
        if b - a <= cap:
            continue
        if b == len(adv):  # the run reaches the end (a failure's final stall): keep its first cap frames
            keep[a + cap:] = False
        else:
            keep[a + cap:max(a + cap, b - tail)] = False
    return keep


def label():
    import server
    import stage_progress_cum
    from stage_labels import STAGES

    notes = json.loads((REPO / "examples/xlerobot/dataset_labels.json").read_text())
    std = np.array(json.loads((server.root_of(DEMOS) / "meta/stats.json").read_text())["observation.state"]["std"])[:12]

    # demos: fine9 + hand edits, road progress
    o, f9 = server.load(DEMOS), server.fine9_of(DEMOS)
    demo = {}
    for ep in o["eps"].index:
        _, s = server.ep_arrays(o, ep)
        lab = np.asarray(f9(ep))
        demo[int(ep)] = (features(s, std, not stage_progress_cum.b_is_left(s, lab)), lab)
    road = stage_progress_cum.progress(demo)["road"]
    ref = {g: (np.concatenate([demo[e][0][demo[e][1] == g] for e in demo]),
               np.concatenate([road[e][demo[e][1] == g] for e in demo])) for g in range(len(STAGES))}
    rows = {}
    for ep, (_, lab) in demo.items():
        v = lab + road[ep]
        rows[(DEMOS, ep)] = dict(lab=lab, p=road[ep], v=v, gain=gain(v), demo=True, success=True)
    dg = np.concatenate([r["gain"] for r in rows.values()])
    dl = np.concatenate([r["lab"] for r in rows.values()])
    eps_stage = np.array([max(EPS_MIN, np.quantile(dg[dl == g], Q)) if (dl == g).any() else EPS_MIN for g in range(len(STAGES))])

    holdout = []
    for ds in EVALS:
        o, f9, ends = server.load(ds), server.fine9_of(ds), server.ends_of(ds)
        nt = {**notes.get(NOTE_ALIAS.get(ds, ""), {}), **notes.get(ds, {})}
        for ep in o["eps"].index:
            ep = int(ep)
            if ep % 5 == 4:
                holdout.append([ds, ep])
            _, s = server.ep_arrays(o, ep)
            ok = bool(o["success"].get(ep, 0))
            end = int(ends.loc[ep, "end_frame"]) if ep in ends.index else len(s)
            end = max(min(end, len(s)), 2)
            lab = np.asarray(f9(ep))[:end].copy()
            ns = None if ok else note_stage(nt.get(str(ep), {}).get("note", ""))
            if ns is not None:
                lab = np.minimum(lab, ns)
            x = features(s[:end], std, a_is_left(s[:end], lab))
            p = np.zeros(end)
            for g in np.unique(lab):
                m = lab == g
                p[m] = knn_mean(x[m], ref[g][0], ref[g][1], K) if len(ref[g][0]) else 0.0
            v = lab + p
            rows[(ds, ep)] = dict(lab=lab, p=p, v=v, gain=gain(v), demo=False, success=ok, note_stage=ns, held=ep % 5 == 4)
    for r in rows.values():
        r["adv"] = (r["gain"] >= eps_stage[r["lab"]]).astype(np.int8)
        r["keep"] = np.ones(len(r["v"]), bool) if r["demo"] else keep_mask(r["adv"])
        r["neg"] = ((not r["demo"]) & (r["adv"] == 0)).astype(np.int8)

    # trim spec: whole demo episodes; rollouts [0, end) with the cuts of keep
    spec = [{"repo": f"{OWNER}/{DEMOS}", "root": str(server.root_of(DEMOS)), "note": "v6 demos, whole",
             "range": {str(ep): [0, len(rows[(DEMOS, ep)]["v"])] for ep in demo}}]
    order = [(DEMOS, ep) for ep in sorted(demo)]
    for ds in EVALS:
        eps = sorted(int(e) for e in server.load(ds)["eps"].index)
        rng, cut = {}, {}
        for ep in eps:
            if (ds, ep) not in rows or rows[(ds, ep)]["held"]:
                continue
            kp = rows[(ds, ep)]["keep"]
            last = int(np.flatnonzero(kp)[-1]) + 1
            rng[str(ep)] = [0, last]
            gaps = [[int(a), int(b)] for a, b in runs(~kp[:last])]
            if gaps:
                cut[str(ep)] = gaps
            order.append((ds, ep))
        spec.append({"repo": f"{OWNER}/{ds}", "root": str(server.root_of(ds)), "note": "eval rollouts",
                     "drop": [e for e in eps if str(e) not in rng], "range": rng, "cut": cut})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    (OUT / "holdout.json").write_text(json.dumps(holdout))
    arr = {}
    for d, e in order:
        r = rows[(d, e)]
        for k in ("lab", "p", "v", "adv", "neg"):
            arr[f"{d}|{e}|{k}"] = r[k][r["keep"]]
        arr[f"{d}|{e}|meta"] = np.array([r["demo"], r["success"]])
    np.savez_compressed(OUT / "labels.npz", order=np.array([f"{d}|{e}" for d, e in order]), eps_stage=eps_stage, **arr)
    # held-out trials: labels over all frames up to the task end (no cuts), for the offline checks (offline_v9.py)
    np.savez_compressed(OUT / "labels_holdout.npz", eps_stage=eps_stage, **{
        f"{d}|{e}|{k}": r[k] for (d, e), r in rows.items() if r.get("held") for k in ("lab", "p", "v", "adv")},
        **{f"{d}|{e}|meta": np.array([False, r["success"]]) for (d, e), r in rows.items() if r.get("held")})
    rows = {k: r for k, r in rows.items() if not r.get("held")}
    lines = [f"eps_stage (gain threshold per stage, stage units over {H} frames): {np.round(eps_stage, 3).tolist()}"]
    for name in [DEMOS, *EVALS]:
        rs = [r for (d, _), r in rows.items() if d == name]
        if not rs:
            continue
        kept = sum(r["keep"].sum() for r in rs)
        allf = sum(len(r["keep"]) for r in rs)
        adv = np.concatenate([r["adv"][r["keep"]] for r in rs])
        lines.append(f"{name}: {len(rs)} eps ({sum(r['success'] for r in rs)} ok), frames kept {kept}/{allf}, adv good {adv.mean():.2f}")
    tot = sum(r["keep"].sum() for r in rows.values())
    neg = sum(r["neg"][r["keep"]].sum() for r in rows.values())
    lines.append(f"total {len(rows)} eps, {tot} frames, neg {neg} ({neg / tot:.2f}); holdout {len(holdout)} trials")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def finish(raw=RAW, final=FINAL):
    """Progress dims into the built set (action 12 -> 21, identity-normalised like add_stage_target.py --kind cumprog),
    frame_labels.parquet, sarm_progress.parquet (RA-BC), episodes_*.json, sources; written as a new dataset `final`."""
    import stage_progress_cum
    from stage_labels import STAGES

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    lab = np.load(OUT / "labels.npz")
    order = [tuple(x.rsplit("|", 1)) for x in lab["order"]]
    root, new_root = lerobot_home() / raw, lerobot_home() / final
    assert not new_root.exists(), f"{final} already exists"
    files = sorted(root.glob("data/*/*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    allf = pd.concat(dfs)
    n_ep = allf.episode_index.nunique()
    assert n_ep == len(order), f"built {n_ep} episodes, labels have {len(order)}"
    extra, fl = {}, []
    for i, (d, e) in enumerate(order):
        g = allf[allf.episode_index == i].sort_values("frame_index")
        k = f"{d}|{e}"
        assert len(g) == len(lab[f"{k}|v"]), f"episode {i} ({k}): {len(g)} frames built, {len(lab[k + '|v'])} labelled"
        extra[i] = stage_progress_cum.cumulative(lab[f"{k}|lab"].astype(int), lab[f"{k}|p"]).astype(np.float32)
        demo, ok = lab[f"{k}|meta"]
        fl.append(pd.DataFrame(dict(index=g["index"].to_numpy(), episode_index=i, frame_index=g.frame_index.to_numpy(),
                                    source=d, source_episode=int(e), demo=int(demo), success=int(ok), v=lab[f"{k}|v"],
                                    adv=lab[f"{k}|adv"].astype(int), neg=lab[f"{k}|neg"].astype(int))))
    shutil.copytree(root / "videos", new_root / "videos", copy_function=os.link)
    shutil.copytree(root / "meta", new_root / "meta")
    for f, df in zip(files, dfs):
        df["action"] = [np.r_[a, extra[int(e)][int(i)]].astype(np.float32) for a, e, i in zip(df["action"], df["episode_index"], df["frame_index"])]
        out = new_root / f.relative_to(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
    names = [f"progress.{s}" for s in STAGES]
    info = json.loads((root / "meta/info.json").read_text())
    n_act = info["features"]["action"]["shape"][0]
    info["features"]["action"]["shape"] = [n_act + len(names)]
    info["features"]["action"]["names"] = list(info["features"]["action"]["names"]) + names
    (new_root / "meta/info.json").write_text(json.dumps(info, indent=4))
    stats = json.loads((root / "meta/stats.json").read_text())
    n = len(names)
    ident = {"min": -np.ones(n), "max": np.ones(n), "mean": np.zeros(n), "std": np.ones(n)}
    allv = np.concatenate([extra[i] for i in sorted(extra)])
    for stat, value in stats["action"].items():
        if stat == "count":
            continue
        q = ident.get(stat)
        if q is None:  # q01 .. q99: the real quantiles (not used by MIN_MAX)
            q = np.quantile(allv, int(stat[1:]) / 100, axis=0)
        stats["action"][stat] = list(value) + q.tolist()
    (new_root / "meta/stats.json").write_text(json.dumps(stats, indent=4))
    fl = pd.concat(fl, ignore_index=True)
    fl.to_parquet(new_root / "frame_labels.parquet", index=False)
    fl.assign(progress_sparse=fl.v / len(STAGES))[["index", "episode_index", "progress_sparse"]].to_parquet(
        new_root / "sarm_progress.parquet", index=False)
    per = fl.groupby("episode_index").agg(demo=("demo", "first"), success=("success", "first"))
    (new_root / "episodes_demo.json").write_text(json.dumps(per.index[per.demo == 1].tolist()))
    (new_root / "episodes_succ.json").write_text(json.dumps(per.index[per.success == 1].tolist()))
    for f in ("sources.json", "spec.json", "holdout.json", "summary.txt", "labels_holdout.npz"):
        if (OUT / f).exists():
            shutil.copy(OUT / f, new_root / f"v9_{f}")
    # check: joints unchanged, progress dims carried, a frame with video still loads
    new = LeRobotDataset(f"{OWNER}/{final}", root=new_root)
    act = np.stack(new.hf_dataset["action"])
    old = np.stack(LeRobotDataset(f"{OWNER}/{raw}", root=root).hf_dataset["action"])
    assert np.array_equal(act[:, :n_act], old) and act.shape[1] == n_act + n
    assert new[len(new) // 2]["action"].shape[-1] == n_act + n
    print(f"{final}: {new.meta.total_episodes} episodes, {new.meta.total_frames} frames; adv good {fl.adv.mean():.2f}, "
          f"neg {fl.neg.mean():.2f}; demo eps {int(per.demo.sum())}, successful eps {int(per.success.sum())}")


def selfcheck():
    v = np.r_[np.linspace(0, 1, 20), np.full(10, 1.0), np.linspace(1, 2, 10)]
    g = gain(v, h=5)
    assert g[0] > 0 and g[-1] == 0
    assert g[22] == 0 and g[27] > 0  # a flat stretch gains nothing until progress is within reach
    adv = np.r_[np.ones(5), np.zeros(200), np.ones(5), np.zeros(300)].astype(np.int8)
    k = keep_mask(adv, cap=150, tail=15)
    assert k[:155].all() and not k[155:190].any() and k[190:210].all()
    assert k[210:360].all() and not k[360:].any()
    assert note_stage("A对位失败") == 1 and note_stage("a对位失败，xx") == 1 and note_stage("AB对位失败") == 4
    assert note_stage("B抓住失败  ") == 5 and note_stage("") is None and note_stage("突变") is None
    assert note_stage("AB对位失败，失败时间不对") == 4
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--finish", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    x = ap.parse_args()
    if x.selfcheck:
        selfcheck()
    if x.download:
        download()
    if x.label:
        label()
    if x.finish:
        finish()
