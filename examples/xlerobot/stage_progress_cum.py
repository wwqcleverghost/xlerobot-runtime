"""Within-stage progress of a *-stagelabel dataset, as the cumulative 9-element array s (docs/workflows/stage-labels).

s[k] = 1 for stages already done, the progress of the current stage at its index, 0 after: stage 3 at 40% = [1,1,1,0.4,0,...].
Stage boundaries are the dataset's fine9 column (hand-set starts included). For each run [a, b) of a stage the target is the
state at b-1; joints (12, ordered A arm then B arm, divided by the dataset std, ~5-frame smoothed) give three progress measures:
  straight: clip(1 - d(t) / max d), d = straight-line distance to the target (the first proposal)
  eta:      1 - expected frames left / that at a: frames-left of the K nearest frames of the same stage in other episodes
  road:     1 - d_road(t) / d_road(a), d_road = shortest path to the target over a K-nearest-neighbour graph of all frames of
            the stage (all episodes, plus each episode's own frame-to-frame steps): the walking, not straight-line, distance
eta and road start at exactly 0 and end at 1; straight and road can go down when the arm moves away from the target.
Post-hoc labels for training only (input s and prediction target); not computable live.

usage: python examples/xlerobot/stage_progress_cum.py --root <dataset dir> --plot 0 48 --out outputs/stage_progress
       python examples/xlerobot/stage_progress_cum.py --selfcheck
"""
import argparse
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage_labels import STAGES, segments

KEY = "observation.environment_state"
K = 10        # neighbours per frame (graph and eta)
SMOOTH = 5    # frames of the moving average on the joints


def smooth(x, w=SMOOTH):
    pad = np.pad(x, ((w // 2, w - 1 - w // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, j], np.ones(w) / w, "valid") for j in range(x.shape[1])], 1)


def b_is_left(st, lab):
    """B is the arm that opens to take the tape in AB对位 (A keeps holding it)."""
    for g in (4, 5, 3):
        m = lab == g
        if m.any():
            return st[m, 5].mean() > st[m, 11].mean()
    return False


def knn(x, k):
    """(indices, distances) of the k nearest rows of x (self excluded), nearest first, chunked brute force."""
    sq = (x * x).sum(1)
    idx, dist = np.empty((len(x), k), np.int64), np.empty((len(x), k))
    for i in range(0, len(x), 2048):
        d2 = np.maximum(sq[i:i + 2048, None] + sq[None] - 2 * x[i:i + 2048] @ x.T, 0)
        d2[np.arange(len(d2)), np.arange(i, i + len(d2))] = np.inf
        j = np.argpartition(d2, k, 1)[:, :k]
        dj = np.take_along_axis(d2, j, 1)
        o = np.argsort(dj, 1)
        idx[i:i + 2048], dist[i:i + 2048] = np.take_along_axis(j, o, 1), np.sqrt(np.take_along_axis(dj, o, 1))
    return idx, dist


def dijkstra(adj, src):
    d = np.full(len(adj), np.inf)
    d[src], q = 0.0, [(0.0, src)]
    while q:
        du, u = heapq.heappop(q)
        if du > d[u]:
            continue
        for v, w in adj[u]:
            if du + w < d[v]:
                d[v] = du + w
                heapq.heappush(q, (d[v], v))
    return d


def progress(eps):
    """eps: {ep: (x (n, 12) normalized smoothed joints, lab (n,) stage)} -> {method: {ep: (n,) within-stage progress}}"""
    out = {m: {ep: np.zeros(len(lab)) for ep, (_, lab) in eps.items()} for m in ("straight", "eta", "road")}
    for g in range(len(STAGES)):
        runs = [(ep, a, b) for ep, (_, lab) in eps.items() for a, b, s in segments(lab) if s == g]
        if not runs:
            continue
        x = np.concatenate([eps[ep][0][a:b] for ep, a, b in runs])
        owner = np.concatenate([np.full(b - a, i) for i, (_, a, b) in enumerate(runs)])
        left = np.concatenate([np.arange(b - a)[::-1] for _, a, b in runs]).astype(float)   # frames to the run's end
        start = np.r_[0, np.cumsum([b - a for _, a, b in runs])]
        idx, dist = knn(x, min(K * 4, len(x) - 1))
        adj = [[] for _ in range(len(x))]
        for i in range(len(x)):
            for j, w in zip(idx[i, :K], dist[i, :K]):
                adj[i].append((j, w)), adj[j].append((i, w))
            if i + 1 < len(x) and owner[i + 1] == owner[i]:
                w = float(np.linalg.norm(x[i + 1] - x[i]))
                adj[i].append((i + 1, w)), adj[i + 1].append((i, w))
        for r, (ep, a, b) in enumerate(runs):
            seg = slice(start[r], start[r + 1])
            d = np.linalg.norm(x[seg] - x[seg][-1], axis=1)
            out["straight"][ep][a:b] = np.clip(1 - d / max(d.max(), 1e-9), 0, 1)
            other = owner[idx[seg]] != r      # eta: frames left of the K nearest frames from other runs
            e = np.array([left[row[ok][:K]].mean() if ok.any() else left[start[r] + t]
                          for t, (row, ok) in enumerate(zip(idx[seg], other))])
            out["eta"][ep][a:b] = np.clip((e[0] - e) / max(e[0] - e[-1], 1e-9), 0, 1)
            dr = dijkstra(adj, start[r + 1] - 1)[seg]
            out["road"][ep][a:b] = np.clip(1 - dr / max(dr[0], 1e-9), 0, 1)
            for m in ("eta", "road"):
                out[m][ep][a], out[m][ep][b - 1] = 0.0, 1.0
    return out


def cumulative(lab, p):
    """(n, 9) array s: done stages 1, current stage p, later 0."""
    s = (np.arange(len(STAGES))[None] < lab[:, None]).astype(np.float32)
    s[np.arange(len(lab)), lab] = p
    return s


def load(root):
    df = pd.concat(pd.read_parquet(f) for f in sorted(Path(root).glob("data/*/*.parquet"))).sort_values("index")
    std = np.array(json.loads((Path(root) / "meta/stats.json").read_text())["observation.state"]["std"])
    eps = {}
    for ep, g in df.groupby("episode_index"):
        st, lab = np.stack(g["observation.state"].to_numpy()), np.stack(g[KEY].to_numpy()).argmax(1)
        x = st / std
        if b_is_left(st, lab):      # order A arm, B arm
            x = np.r_["1", x[:, 6:], x[:, :6]]
        eps[int(ep)] = (smooth(x), lab)
    return eps


def svg(ep, lab, prog, path, fps=30):
    """Plot of one episode: x time, y = stage index + within-stage progress (0..9), one line per method, stage bands."""
    W, H, L, T = 1100, 380, 40, 20
    n = len(lab)
    X = lambda t: L + t / max(n - 1, 1) * (W - L - 20)
    Y = lambda v: T + (1 - v / 9) * (H - T - 40)
    col = {"straight": "#999", "eta": "#2a7", "road": "#d33"}
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Noto Sans CJK SC,sans-serif" '
         f'font-size="11"><rect width="{W}" height="{H}" fill="white"/>']
    for a, b, g in segments(lab):
        e.append(f'<rect x="{X(a):.1f}" y="{T}" width="{X(b) - X(a):.1f}" height="{H - T - 40}" '
                 f'fill="{"#f0f0f0" if g % 2 else "#fff"}"/><text x="{X(a) + 2:.1f}" y="{T + 12 + 12 * (g % 2)}">{STAGES[g]}</text>')
    for v in range(10):
        e.append(f'<line x1="{L}" x2="{W - 20}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#ddd"/>'
                 f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for t in range(0, n, fps * 5):
        e.append(f'<text x="{X(t):.1f}" y="{H - 24}" text-anchor="middle">{t // fps}s</text>')
    for i, (m, p) in enumerate(prog.items()):
        pts = " ".join(f"{X(t):.1f},{Y(lab[t] + p[t]):.1f}" for t in range(n))
        e.append(f'<polyline points="{pts}" fill="none" stroke="{col[m]}" stroke-width="1.5"/>'
                 f'<text x="{L + 120 * i}" y="{H - 6}" fill="{col[m]}">— {m}</text>')
    e.append(f'<text x="{W - 20}" y="{H - 6}" text-anchor="end">ep{ep}  y = 阶段编号 + 阶段内进度</text></svg>')
    Path(path).write_text("".join(e))


def selfcheck():
    # a U-shaped path (out, across, back toward the end): the straight distance first grows, the road distance only falls
    t = np.linspace(0, 1, 60)
    u = np.zeros((180, 12))
    u[:, 0], u[:, 1] = np.r_[t, np.ones(60), 1 - t], np.r_[np.zeros(60), t, np.ones(60)]
    eps = {i: (np.r_[np.zeros((10, 12)), u + 0.01 * i], np.r_[np.zeros(10, int), np.ones(180, int)]) for i in range(4)}
    p = progress(eps)
    r, s = p["road"][0][10:], p["straight"][0][10:]
    assert r[0] == 0 and r[-1] == 1 and (np.diff(r) >= -1e-6).all(), "road progress rises monotonically along the U"
    assert (np.diff(s[:60]) < 0).any(), "straight progress first falls on the U"
    assert p["eta"][0][10] == 0 and p["eta"][0][-1] == 1
    assert np.allclose(cumulative(np.array([3]), np.array([0.4])), [[1, 1, 1, 0.4, 0, 0, 0, 0, 0]])
    print("selfcheck ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path)
    ap.add_argument("--plot", type=int, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=Path("outputs/stage_progress"))
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
        raise SystemExit
    eps = load(args.root)
    prog = progress(eps)
    args.out.mkdir(parents=True, exist_ok=True)
    for ep in args.plot:
        svg(ep, eps[ep][1], {m: prog[m][ep] for m in prog}, args.out / f"ep{ep}.svg")
    np.savez(args.out / "progress.npz", **{f"{m}_{ep}": v for m, d in prog.items() for ep, v in d.items()})
    print(f"{len(eps)} episodes; plots in {args.out}")
