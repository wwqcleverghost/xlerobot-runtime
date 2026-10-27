"""Pick high-quality episodes: per-episode metrics, on/off rules with thresholds (presets), and what each rule drops.

The web page (#curate in server.py) and this command line use the same functions. Metrics come from the existing rule
scripts where they exist: snaps / pauses (review_page.py), small back-and-forth (smooth.ep_metrics, as report.py),
follower jerk (lerobot_dataset_quality.episode_metrics), gripper closes / tape pixels (ep_score.py), midline rule and
500 tape pixels (select_strict.py). small_snap, stall and the standing threshold are new (2026-10-27) and still need the
user's OK. Presets: docs/workflows/dataset-viewer/presets/<name>.json. Forced keep / drop: "force" in dataset_labels.json.

usage: PYTHONPATH=src python docs/workflows/dataset-viewer/curate.py --datasets raw-handover-20260929-trimmed raw-handover-20261001-trimmed \\
         [--preset default] [--set jerk.x=2 --set stall.on=1 ...] [--report out.csv] [--spec <name>] [--build <dst> [--arm_only]]
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRESETS = HERE / "presets"
J = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]  # arm joints without the grippers

# Rule order = order of the cumulative counts. "new": not agreed with the user yet (default off, marked on the page).
RULES = [
    dict(key="snap", short="突变", label="突变（从臂一帧跳 >20，指令该帧变化 <5）", params={"min": 1}, on=True,
         help="review_page.py snaps()；>= min 次就去掉"),
    dict(key="pause", short="停顿", label="停顿（指令 <3 单位/秒持续 >= s 秒）", params={"s": 1.0}, on=True,
         help="review_page.py pauses()；有一段 >= s 秒就去掉（开头的静止不算）"),
    dict(key="rev", short="抖动", label="小幅来回抖动 >= 基准中位数 × x", params={"x": 2.0, "ref": "v4-handover2-trimmed"}, on=True,
         help="report.py 的 rev_rate（smooth.py），基准 = ref 数据集各条的中位数；v6 精选时用户同意过 2 倍"),
    dict(key="jerk", short="jerk", label="从臂 jerk >= 基准中位数 × x", params={"x": 2.0, "ref": "v4-handover2-trimmed"}, on=True,
         help="observation.state 前 12 维三阶差分范数的中位数 × fps³，基准 = ref 数据集各条的中位数（裁剪后的 handover2；"
              "原始 handover2 含头尾静止段，中位数偏低）；用户已同意看从臂"),
    dict(key="small_snap", short="小卡顿", label="从臂小卡顿：指令几乎不动、从臂一帧跳 >6 的帧数 >= min", params={"min": 15}, on=False,
         new=True, help="10 个非夹爪关节 |Δaction| 都 <2 而某关节 |Δstate| >6（低于突变阈值）；新指标，待用户确认"),
    dict(key="stall", short="卡住", label="从臂卡住：指令在动、从臂不动连续 >=4 帧的次数 >= min", params={"min": 8}, on=False,
         new=True, help="max|Δaction| >1 而 max|Δstate| <0.3，连续 4 帧算一次；新指标，待用户确认"),
    dict(key="place", short="放置", label="清单：末帧桌上看得到胶带（黄色像素 >= px）", params={"px": 500}, on=False,
         help="ep_score.py yellow()，阈值同 select_strict.py；0 = 放进篮子或不在可见桌面"),
    dict(key="regrasp", short="重抓", label="清单：每只手夹爪闭合 <= n 次（多了 = 重抓）", params={"n": 2}, on=False,
         help="ep_score.py closes()；2/2 是干净的"),
    dict(key="midline", short="中线", label="清单：先抓的手臂符合中线规则（L x<=0.49，R x>=0.51）", params={}, on=False,
         help="首帧胶带位置 + 第一次抓住的手臂（select_strict.py）"),
    dict(key="standing", short="立着", label="清单：开头胶带立着（色块高/宽 >= r）", params={"r": 1.2}, on=False, new=True,
         help="ep_score.py 的 hw_start；清单说开头立着要丢，但没有定数值阈值，1.2 待用户看图确认"),
]
DEFAULT = {r["key"]: {"on": r["on"], **r["params"]} for r in RULES}


# ---------- metrics (one episode: a = action, s = observation.state, first 12 dims) ----------

def small_snaps(a, s):
    """Frames where the command barely moves (every arm joint |Δaction| < 2) but a follower joint jumps > 6."""
    ds, da = np.abs(np.diff(s[:, J], axis=0)), np.abs(np.diff(a[:, J], axis=0))
    return int(((ds > 6) & (da < 2)).any(1).sum())


def stalls(a, s):
    """Times the follower stands still (max |Δstate| < 0.3) for 4 frames in a row while the command moves (max |Δaction| > 1)."""
    ds, da = np.abs(np.diff(s[:, J], axis=0)), np.abs(np.diff(a[:, J], axis=0))
    run, n = 0, 0
    for m in (da.max(1) > 1.0) & (ds.max(1) < 0.3):
        run = run + 1 if m else 0
        n += run == 4
    return n


def episode(a, s, fps):
    from ep_score import closes
    from lerobot_dataset_quality import episode_metrics
    from review_page import pauses, snaps
    from score import GAP, runs
    from smooth import ep_metrics
    rev, moving = ep_metrics(s)
    hold = {k: runs((a[:, g] < s[:, g] - GAP) & (s[:, g] > 4)) for k, g in (("L", 5), ("R", 11))}
    return dict(n=len(a), snaps=len(snaps(a, s, 0, len(a))),
                pauses=[round((t - t0) / fps, 2) for t0, t in pauses(a, 0, len(a), min_s=0.3)],
                rev_rate=rev / max(moving, 1e-6), jerk=episode_metrics(a, fps, s)["median_jerk"],
                small_snap=small_snaps(a, s), stall=stalls(a, s), closesL=closes(a[:, 5]), closesR=closes(a[:, 11]),
                first_arm=min((v[0][0], k) for k, v in hold.items() if v)[1] if any(hold.values()) else "")


def vision(start, end):
    """start / end: the first and last head-camera frame as HxWx3 uint8."""
    from ep_score import yellow
    _, x0, _, hw0 = yellow(start)
    return dict(tape_px_end=yellow(end)[0], tape_x_start=round(x0 / start.shape[1], 3) if x0 >= 0 else None, hw_start=hw0)


# ---------- rules ----------

def failed(m, p, refs, only=None):
    """Rule key -> why this episode fails it, for the rules that are on (or just `only`)."""
    on = lambda k: (only == k) if only else p[k]["on"]  # noqa: E731
    why = {}
    if on("snap") and m["snaps"] >= p["snap"]["min"]:
        why["snap"] = f"突变 {m['snaps']} 次"
    long = [x for x in m["pauses"] if x >= p["pause"]["s"]]
    if on("pause") and long:
        why["pause"] = f"停顿 {max(long):.1f} 秒"
    if on("rev") and refs.get("rev") and m["rev_rate"] >= p["rev"]["x"] * refs["rev"]:
        why["rev"] = f"抖动 {m['rev_rate'] / refs['rev']:.1f} 倍"
    if on("jerk") and refs.get("jerk") and m["jerk"] >= p["jerk"]["x"] * refs["jerk"]:
        why["jerk"] = f"jerk {m['jerk'] / refs['jerk']:.1f} 倍"
    if on("small_snap") and m["small_snap"] >= p["small_snap"]["min"]:
        why["small_snap"] = f"小卡顿 {m['small_snap']} 帧"
    if on("stall") and m["stall"] >= p["stall"]["min"]:
        why["stall"] = f"卡住 {m['stall']} 次"
    if on("place") and (m.get("tape_px_end") or 0) < p["place"]["px"]:
        why["place"] = f"末帧胶带 {m.get('tape_px_end')} 像素"
    if on("regrasp") and max(m["closesL"], m["closesR"]) > p["regrasp"]["n"]:
        why["regrasp"] = f"闭合 L{m['closesL']}/R{m['closesR']}"
    x = m.get("tape_x_start")
    if on("midline") and not ((m["first_arm"] == "L" and x is not None and x <= 0.49) or
                              (m["first_arm"] == "R" and x is not None and x >= 0.51)):
        why["midline"] = f"{m['first_arm'] or '-'} 臂、胶带 x={x}"
    if on("standing") and (m.get("hw_start") or 0) >= p["standing"]["r"]:
        why["standing"] = f"开头色块高/宽 {m['hw_start']}"
    return why


def full(preset):
    return {k: {**DEFAULT[k], **preset.get(k, {})} for k in DEFAULT}


MANUAL = {"standing": "人工：开头立着", "basket": "人工：放进框里"}  # "ok" = looked at, no effect


def evaluate(metrics, preset, refs, force, manual=None):
    """metrics = {ds: [episode dicts with "ep"]}, force = {ds: {ep: 'keep'|'drop'}}, manual = {ds: {ep: 'standing'|
    'basket'|'ok'}} (the start / end frame review: standing and basket drop the episode). Returns one row per episode
    (kept, reasons) and, per dataset, how many remain after each rule that is on, in RULES order, then after the manual
    marks (cumulative; force keep / drop is applied to the rows only)."""
    p, manual = full(preset), manual or {}
    rows, steps = [], {}
    for ds, eps in metrics.items():
        steps[ds], gone, man = [dict(key="all", left=len(eps))], set(), manual.get(ds, {})
        for r in RULES:
            if p[r["key"]]["on"]:
                gone |= {m["ep"] for m in eps if failed(m, p, refs, only=r["key"])}
                steps[ds].append(dict(key=r["key"], left=len(eps) - len(gone)))
        if any(v in MANUAL for v in man.values()):
            gone |= {e for e, v in man.items() if v in MANUAL}
            steps[ds].append(dict(key="manual", left=len(eps) - len(gone)))
        for m in eps:
            why, f, v = failed(m, p, refs), force.get(ds, {}).get(m["ep"]), man.get(m["ep"])
            if v in MANUAL:
                why["manual"] = MANUAL[v]
            rows.append(dict(m, ds=ds, reasons=why, force=f, manual=v, kept=f == "keep" or (f != "drop" and not why)))
    return rows, steps


def merged(preset, sets=()):
    """preset + --set rule.param=value overrides."""
    p = copy.deepcopy(preset)
    for kv in sets:
        k, v = kv.split("=", 1)
        rule, param = k.split(".")
        assert rule in DEFAULT and param in DEFAULT[rule], f"unknown setting {k}"
        v = v.lower() in ("1", "true", "on") if param == "on" else v if param == "ref" else float(v)
        p.setdefault(rule, {})[param] = v
    return p


def read_preset(name):
    return json.load(open(PRESETS / f"{name}.json"))


def presets():
    return {p.stem: json.load(open(p)) for p in sorted(PRESETS.glob("*.json"))}


def selfcheck():
    a, s = np.zeros((20, 12)), np.zeros((20, 12))
    s[10:, 0] = 8  # the follower jumps 8 while the command stands still
    assert small_snaps(a, s) == 1
    a[:, 1] = np.arange(20) * 2  # the command moves, the follower stays frozen for 19 frames -> one stall
    assert stalls(a, np.zeros((20, 12))) == 1
    m = dict(ep=0, snaps=0, pauses=[1.5], rev_rate=1.0, jerk=1.0, small_snap=0, stall=0, closesL=2, closesR=2,
             first_arm="L", tape_x_start=0.3, tape_px_end=900, hw_start=0.5)
    rows, steps = evaluate({"d": [m, dict(m, ep=1, pauses=[])]}, {}, {"rev": 1.0, "jerk": 1.0}, {"d": {0: "keep"}})
    assert [r["kept"] for r in rows] == [True, True] and "pause" in rows[0]["reasons"]
    assert [x["left"] for x in steps["d"]] == [2, 2, 1, 1, 1], steps
    rows, steps = evaluate({"d": [dict(m, ep=1, pauses=[]), dict(m, ep=2, pauses=[])]}, {}, {"rev": 1.0, "jerk": 1.0}, {},
                           {"d": {1: "basket", 2: "ok"}})
    assert [r["kept"] for r in rows] == [False, True] and rows[0]["reasons"] == {"manual": "人工：放进框里"}
    assert steps["d"][-1] == {"key": "manual", "left": 1}, steps
    assert merged({}, ["jerk.x=3", "stall.on=1"]) == {"jerk": {"x": 3.0}, "stall": {"on": True}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", required=True, help="dataset ids, e.g. raw-handover-20260929-trimmed")
    ap.add_argument("--preset", default="default")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. jerk.x=2.5 or stall.on=1")
    ap.add_argument("--report", help="write one row per episode to this CSV")
    ap.add_argument("--spec", help="write the trim spec examples/xlerobot/<name>.json with the kept episodes")
    ap.add_argument("--build", help="build that spec into wwqcleverghost/<dst> in the background (needs --spec)")
    ap.add_argument("--arm_only", action="store_true", help="keep only the 12 arm joints")
    ap.add_argument("--overwrite", action="store_true", help="rename an existing <dst> to <dst>.old_<time> first")
    x = ap.parse_args()
    selfcheck()
    import server  # data loading, caches, labels and rebuild live there (the same code as the web page)
    preset = merged(read_preset(x.preset), x.set)
    res = server.curate_run(x.datasets, preset)
    names = {**{r["key"]: r["label"] for r in RULES}, "manual": "人工标记（开头立着 / 放进框里）"}
    shorts = {**{r["key"]: r["short"] for r in RULES}, "manual": "人工"}
    print(f"preset {x.preset} {x.set or ''}; 基准: 抖动 {res['refs'].get('rev') or 0:.3f}/s, jerk {res['refs'].get('jerk') or 0:.0f}")
    for ds, st in res["steps"].items():
        print(f"\n{ds}: " + " -> ".join(f"{'全部' if s['key'] == 'all' else shorts[s['key']]} {s['left']}" for s in st))
        for key in [s["key"] for s in st[1:]]:
            eps = [r["ep"] for r in res["rows"] if r["ds"] == ds and key in r["reasons"]]
            print(f"  {names[key]}: {len(eps)} 条 {eps}")
        forced = [(r["ep"], r["force"]) for r in res["rows"] if r["ds"] == ds and r["force"]]
        if forced:
            print(f"  手动强制: {forced}")
    kept = [r for r in res["rows"] if r["kept"]]
    print(f"\n保留 {len(kept)} / {len(res['rows'])} 条")
    if x.report:
        import pandas as pd
        pd.DataFrame([dict(r, reasons="; ".join(r["reasons"].values())) for r in res["rows"]]).to_csv(x.report, index=False)
        print("wrote", x.report)
    if x.spec:
        print("wrote spec", server.curate_spec(x.datasets, preset, x.spec, x.preset))
        if x.build:
            srcs = json.load(open(server.spec_path(f"{x.spec}.json")))
            jobs = server.rebuild(f"{x.spec}.json", {"reset_tol": 10, "to_target": False, "snap_rule": False},
                                  {s["repo"]: x.build for s in srcs}, x.overwrite, x.arm_only)
            print("building", [(j["dst"], j["log"]) for j in jobs])
    elif x.build:
        ap.error("--build needs --spec")


if __name__ == "__main__":
    main()
