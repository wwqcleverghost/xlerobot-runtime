"""Completion + quality scores (0-1, 95% CI) for handover evals and demo datasets, plus one bar chart.

One set of stages everywhere: the 9 fine9 stages (examples/xlerobot/stage_labels.py), read per episode from episode_end.csv
(mark_end.py: the task end = the first 复位 frame, and the furthest stage reached).
Completion = mean of the 9 stage pass rates, each rate = (passed + 1) / (entered + 2) (a trial that never got to a stage
does not count against it; the +1/+2 pulls stages with few trials toward 0.5). A stage is passed when the next one is
entered; the last one (复位) when the human label says success. The human judgement is final: a success label passes all 9,
a failed trial whose note names a stage ("AB对位失败", dataset_labels.json) got to that stage and no further.
Quality = weighted per-trial score vs the human reference demos: jitter and task time (first motion to the task end) score
1 at the human median, 0.5 at 2x, 0 at 4x. Stage rates and the idle score are truncated: values at or below FLOOR score 0,
so differences between models are not squeezed into the top of the scale.
Total = W_COMPLETION * completion + (1 - W_COMPLETION) * quality for evals; quality only for demo datasets (no success.csv).
Also reported: plain task success rate. CIs: Wilson for stage and success rates, bootstrap over trials for completion / quality / total.

Usage: python report.py <out_prefix> <ref_demo_dir> <dataset_dir> ...   -> <out_prefix>.csv, <out_prefix>.svg
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
XLE = os.path.join(HERE, "../../../examples/xlerobot")
sys.path.insert(0, HERE)
sys.path.insert(0, XLE)
import mark_end  # noqa: E402
from score import FPS, locate, score  # noqa: E402
from smooth import ep_metrics  # noqa: E402
from stage_labels import STAGES  # noqa: E402

W_COMPLETION = 0.5
W_QUALITY = {"smooth": 0.4, "speed": 0.3, "idle": 0.3}
B = 2000
FLOOR = 0.5  # ponytail: one truncation floor for everything; tune if scores bunch at 0 or 1
rng = np.random.default_rng(0)


def notes(name):
    """{episode: note} of one dataset in examples/xlerobot/dataset_labels.json (the dataset viewer's notes)."""
    path = os.path.join(XLE, "dataset_labels.json")
    labels = json.load(open(path)).get(name, {}) if os.path.exists(path) else {}
    return {int(ep): v.get("note", "") for ep, v in labels.items() if ep.isdigit()}


def note_stage(note):
    """Index of the stage a note names as the failure ("AB对位失败"), the longest name that fits; -1 when it names none."""
    hit = [g for g, name in enumerate(STAGES) if name + "失败" in (note or "").upper()]  # "a对位失败" counts too
    return max(hit, key=lambda g: len(STAGES[g]), default=-1)


def stage_marks(top, success, note=""):
    """(9 stage results with 1 = passed, who decided: label / note / rule). top = the furthest stage the rule saw."""
    if success:
        return [1] * len(STAGES), "label"
    g = note_stage(note)
    return [int(i < (g if g >= 0 else top)) for i in range(len(STAGES))], "note" if g >= 0 else "rule"


def trials(path, name=None, fine9=None):
    """(one row per episode, is it an eval). name = the dataset's key in dataset_labels.json (default: the folder name);
    fine9(ep) -> per-frame stage labels, only used when episode_end.csv has to be written (default: mark_end's)."""
    name = name or os.path.basename(path.rstrip("/"))
    base, csv = locate(path)
    d = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/data/*/*.parquet")))
    lab = pd.read_csv(csv).set_index("episode").success if csv else None
    ends, note = mark_end.load(path, fine9, set(d.episode_index.unique())), notes(name)
    rows = [dict(ep=ep, **trial(np.stack(g["observation.state"].values), np.stack(g["action"].values),
                                None if lab is None else lab[ep], int(ends.end_frame[ep]), ends.how[ep],
                                int(ends.top_stage[ep]), note.get(int(ep), "")))
            for ep, g in d.groupby("episode_index")]
    return pd.DataFrame(rows), lab is not None


def trial(st, ac, label=None, end=None, how=None, top=-1, note=""):
    """One episode's stage results and quality inputs. label = human success (None for demos: done = reached 复位);
    end / how / top = its row of episode_end.csv (task end frame, how it was found, furthest stage); note = its note.
    Without them (a part of an episode) there is no task time and the idle time runs to the last frame."""
    reset = how == "reset"  # the task end is the start of 复位: only then is there a task time
    ok = reset if label is None else bool(label)
    stg, by = stage_marks(top, ok, note)
    r = score(st, ac, end)
    rev, moving_s = ep_metrics(st)
    task_s = max((len(st) if end is None else end) / FPS - r["start"], 1e-6)
    return dict(**dict(zip(STAGES, stg)), success=int(ok), top=min(sum(stg), len(STAGES) - 1), top_by=by, rule_top=top,
                rev_rate=rev / moving_s, task_s=task_s if ok and reset else np.nan, idle_frac=min(r["idle"] / task_s, 1.0))


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


def selfcheck():
    assert len(STAGES) == 9 and note_stage("AB对位失败，没接住") == 4 and note_stage("A对位失败") == 1 and note_stage("a对位失败") == 1 and note_stage("夹爪抖") == -1
    assert stage_marks(3, True) == ([1] * 9, "label")
    assert stage_marks(3, False) == ([1, 1, 1, 0, 0, 0, 0, 0, 0], "rule")  # entered stage 3 (A抓着), did not pass it
    assert stage_marks(8, False, "B抓住失败") == ([1] * 5 + [0] * 4, "note")  # the note wins over the rule
    st = np.zeros((300, 12))
    st[30:90, 0] = np.linspace(0, 60, 60)
    st[90:, 0] = 60  # moves from 1 s to 3 s, then waits 7 s
    t = trial(st, st, 1, 120, "reset", 8)  # success, 复位 starts at 4 s
    assert abs(t["task_s"] - 3.0) < 1e-9 and t["success"] == 1 and t["top"] == 8 and t["top_by"] == "label"
    assert np.isnan(trial(st, st, 1, 300, "human", 7)["task_s"])  # success, but the rule never saw 复位: no task time
    assert trial(st, st, None, 120, "reset", 8)["success"] == 1 and trial(st, st, None, 250, "still", 5)["success"] == 0
    assert trial(st, st, 0, 300, "human", 8)["idle_frac"] > trial(st, st, 0, 120, "reset", 8)["idle_frac"]
    df = pd.DataFrame([trial(st, st, s, 120, "reset", top, n) for s, top, n in [(1, 8, ""), (0, 3, ""), (0, 8, "A抓住失败")]])
    c, k, n = completion(df)
    assert list(n) == [3, 3, 3, 2, 1, 1, 1, 1, 1] and list(k) == [3, 3, 2, 1, 1, 1, 1, 1, 1], (k, n)


if __name__ == "__main__":
    selfcheck()
    out, ref_dir, dirs = sys.argv[1], sys.argv[2].rstrip("/"), [d.rstrip("/") for d in sys.argv[3:]]
    ref_df, _ = trials(ref_dir)
    ref = {"rev_rate": ref_df.rev_rate.median(), "task_s": ref_df.task_s.median()}
    print(f"reference {os.path.basename(ref_dir)}: {ref['rev_rate']:.2f} reversals/s, task {ref['task_s']:.1f} s")
    res = pd.DataFrame([summarize(os.path.basename(d), *trials(d), ref) for d in dirs])
    res.to_csv(f"{out}.csv", index=False, float_format="%.3f")
    plot(res, f"{out}.svg")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res.round(2).to_string(index=False))
