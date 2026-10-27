"""Completion + quality scores (0-1, 95% CI) for handover evals and demo datasets, plus one bar chart.

Completion = mean of the 5 stage pass rates, each rate = (passed + 1) / (reached + 2) (a trial that never got to a stage
does not count against it; the +1/+2 pulls stages with few trials toward 0.5). Quality = weighted per-trial score vs the
human reference demos: jitter and task time score 1 at the human median, 0.5 at 2x, 0 at 4x. Stage rates and the idle
score are truncated: values at or below FLOOR score 0, so differences between models are not squeezed into the top of the scale.
Total = W_COMPLETION * completion + (1 - W_COMPLETION) * quality for evals; quality only for demo datasets (no success.csv).
Also reported: plain task success rate. CIs: Wilson for stage and success rates, bootstrap over trials for completion / quality / total.
No video review: the human success label in success.csv is final for task success.

Usage: python report.py <out_prefix> <ref_demo_dir> <dataset_dir> ...   -> <out_prefix>.csv, <out_prefix>.svg
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import score  # noqa: E402
from smooth import ep_metrics  # noqa: E402

W_COMPLETION = 0.5
W_QUALITY = {"smooth": 0.4, "speed": 0.3, "idle": 0.3}
STAGES = ["reach", "grasp", "no_drop", "handover", "place"]
B = 2000
FLOOR = 0.5  # ponytail: one truncation floor for everything; tune if scores bunch at 0 or 1
rng = np.random.default_rng(0)


def trials(path):
    path = f"{path}/dataset" if os.path.isdir(f"{path}/dataset") else path  # some eval uploads nest it
    d = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{path}/data/*/*.parquet")))
    csv = f"{path}/success.csv"
    lab = pd.read_csv(csv).set_index("episode").success if os.path.exists(csv) else None
    rows = [dict(ep=ep, **trial(np.stack(g["observation.state"].values), np.stack(g["action"].values),
                                None if lab is None else lab[ep]))
            for ep, g in d.groupby("episode_index")]
    return pd.DataFrame(rows), lab is not None


def trial(st, ac, label=None):
    """One episode's stages and quality inputs; label = human success (None for demos)."""
    r = score(st, ac)
    rev, moving_s = ep_metrics(st)
    stg = list(r["stage"])
    if label is not None:  # human label overrides the auto result for task success
        stg = [1] * 5 if label else (stg[:4] + [0] if all(stg) else stg)
    ok = all(stg) if label is not None else r["done"] is not None
    end = r["done"] if r["done"] else r["dur"]
    task_s = max(end - r["start"], 1e-6)
    return dict(**dict(zip(STAGES, stg)), success=int(ok), rev_rate=rev / moving_s,
                task_s=task_s if ok and r["done"] else np.nan, idle_frac=min(r["idle"] / task_s, 1.0))


def trunc(x):
    return np.clip((x - FLOOR) / (1 - FLOOR), 0, 1)  # NaN stays NaN


def vs_human(x, ref):
    """1 at or better than the human median, 0.5 at 2x worse, 0 at 4x worse (each doubling costs half)."""
    return np.clip(1 - np.log2(np.maximum(x / ref, 1)) / 2, 0, 1)


def quality(df, ref):
    s = pd.DataFrame({"smooth": vs_human(df.rev_rate, ref["rev_rate"]),
                      "speed": vs_human(df.task_s, ref["task_s"]),  # NaN for failed trials: not counted
                      "idle": trunc(1 - df.idle_frac)})
    w = pd.Series(W_QUALITY)
    return (s * w).sum(1) / s.notna().mul(w).sum(1), s


def completion(df):
    stg = df[STAGES].values
    reached = np.c_[np.ones(len(df)), stg.cumprod(1)[:, :-1]]
    n, k = reached.sum(0), (stg * reached).sum(0)
    return trunc((k + 1) / (n + 2)).mean(), k, n


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0, c - h), min(1, c + h)


def summarize(name, df, is_eval, ref):
    q_ep, sub = quality(df, ref)

    def stats(idx):
        q = q_ep.iloc[idx].mean()
        if not is_eval:
            return q, q, q
        c = completion(df.iloc[idx])[0]
        return c, q, W_COMPLETION * c + (1 - W_COMPLETION) * q

    point = stats(np.arange(len(df)))
    boot = np.array([stats(rng.integers(0, len(df), len(df))) for _ in range(B)])
    lo, hi = np.nanpercentile(boot, 2.5, 0), np.nanpercentile(boot, 97.5, 0)
    row = dict(dataset=name, n=len(df), success=f"{df.success.sum()}/{len(df)}" if is_eval else "",
               completion=point[0] if is_eval else np.nan, completion_lo=lo[0] if is_eval else np.nan,
               completion_hi=hi[0] if is_eval else np.nan,
               quality=point[1], quality_lo=lo[1], quality_hi=hi[1], total=point[2], total_lo=lo[2], total_hi=hi[2])
    if is_eval:
        k = df.success.sum()
        row["success_rate"], (row["success_lo"], row["success_hi"]) = k / len(df), wilson(k, len(df))
        _, k, n = completion(df)
        for s, ki, ni in zip(STAGES, k, n):
            a, b = wilson(ki, ni)
            row[s] = f"{int(ki)}/{int(ni)} [{a:.2f}-{b:.2f}]"
    for c in W_QUALITY:
        row[f"q_{c}"] = sub[c].mean()
    return row


def plot(res, out):
    """One stacked bar per dataset: W_COMPLETION*completion (bottom) + rest*quality (top) = total, whisker = total 95% CI.
    Demo datasets have no completion score; a dashed box draws it as 1.0 for comparison. Plain SVG, no dependency."""
    res = res.assign(demo=res.completion.isna()).sort_values(["demo", "total"], ascending=[True, False])  # evals first
    wc, wq = W_COMPLETION, 1 - W_COMPLETION
    L, T, H, G, bw = 50, 50, 280, 64, 34  # left pad, top pad, plot height, slot width, bar width
    W = L + G * len(res) + 140  # room for the last rotated label
    y = lambda v: T + H * (1 - v)  # noqa: E731
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{T + H + 150}" font-family="sans-serif" '
         f'font-size="11"><rect width="100%" height="100%" fill="#fff"/>',
         f'<text x="{L}" y="18" font-size="13">Handover score = {wc:g} x completion + {wq:g} x quality '
         f'(whisker: 95% CI of the total)</text>']
    for v in np.linspace(0, 1, 6):
        s.append(f'<line x1="{L}" x2="{W - 10}" y1="{y(v)}" y2="{y(v)}" stroke="#e3e3e3"/>'
                 f'<text x="{L - 6}" y="{y(v) + 4}" text-anchor="end" fill="#555">{v:.1f}</text>')
    legend = [('<rect width="10" height="10" rx="2" fill="#2a78d6"/>', f"completion x {wc:g}"),
              ('<rect width="10" height="10" rx="2" fill="#eb6834"/>', f"quality x {wq:g}"),
              ('<rect width="10" height="10" fill="none" stroke="#2a78d6" stroke-dasharray="3 2"/>',
               "human demos: completion taken as 1.0")]
    for i, (mark, label) in enumerate(legend):
        s.append(f'<g transform="translate({L + 150 * i},28)">{mark}<text x="14" y="9" fill="#333">{label}</text></g>')
    for g, (_, r) in enumerate(res.iterrows()):
        x, cx = L + G * g + (G - bw) / 2, L + G * g + G / 2
        c = 1.0 if r.demo else r.completion
        lo, hi = (wc + wq * r.quality_lo, wc + wq * r.quality_hi) if r.demo else (r.total_lo, r.total_hi)
        top = wc * c + wq * r.quality
        tip = (f"{r.dataset}: total {top:.2f} [{lo:.2f}-{hi:.2f}], completion "
               f"{'n/a' if r.demo else f'{c:.2f}'}, quality {r.quality:.2f}")
        if r.demo:
            s.append(f'<rect x="{x + 0.5}" y="{y(wc)}" width="{bw - 1}" height="{H * wc}" fill="none" '
                     f'stroke="#2a78d6" stroke-dasharray="4 3"/>')
        else:
            s.append(f'<rect x="{x}" y="{y(wc * c)}" width="{bw}" height="{H * wc * c}" fill="#2a78d6">'
                     f'<title>{tip}</title></rect>')
        s.append(f'<rect x="{x}" y="{y(top)}" width="{bw}" height="{H * wq * r.quality - 2}" fill="#eb6834">'
                 f'<title>{tip}</title></rect>')
        s.append(f'<path d="M{cx} {y(lo)}V{y(hi)}M{cx - 4} {y(lo)}h8M{cx - 4} {y(hi)}h8" stroke="#333"/>'
                 f'<text x="{cx}" y="{y(hi) - 5}" text-anchor="middle" fill="#333">{top:.2f}</text>')
        label = r.dataset + ("" if r.demo else f" ({r.success})")
        s.append(f'<text transform="translate({cx - 4},{T + H + 12}) rotate(35)" fill="#333">{label}</text>')
    s.append("</svg>")
    open(out, "w").write("\n".join(s))


if __name__ == "__main__":
    out, ref_dir, dirs = sys.argv[1], sys.argv[2].rstrip("/"), [d.rstrip("/") for d in sys.argv[3:]]
    ref_df, _ = trials(ref_dir)
    ref = {"rev_rate": ref_df.rev_rate.median(), "task_s": ref_df.task_s.median()}
    print(f"reference {os.path.basename(ref_dir)}: {ref['rev_rate']:.2f} reversals/s, task {ref['task_s']:.1f} s")
    res = pd.DataFrame([summarize(os.path.basename(d), *trials(d), ref) for d in dirs])
    res.to_csv(f"{out}.csv", index=False, float_format="%.3f")
    plot(res, f"{out}.svg")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res.round(2).to_string(index=False))
