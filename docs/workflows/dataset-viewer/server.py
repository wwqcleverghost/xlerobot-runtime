"""Local dataset viewer: trim review, training-set quality, batch playback and eval scores in one page (see README.md).

The page plays the datasets' own mp4 files (h264, seeked to each episode's from_timestamp, nothing is transcoded) and
gets everything else from a small JSON API. Rules come from the existing scripts, nothing is re-implemented here:
keep_range / reach_start (trim_stagnant_frames.py), snaps / pauses / propose (review_page.py), episode_metrics /
flag_outliers (lerobot_dataset_quality.py), trials / quality / summarize (eval-scoring/report.py).
Trim edits go into the spec JSON (examples/xlerobot/*.json), review / notes / tags into examples/xlerobot/dataset_labels.json.

usage: PYTHONPATH=src python docs/workflows/dataset-viewer/server.py [--port 8770]   then open http://localhost:8770
"""
import argparse
import glob
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in ("src", "src/lerobot/scripts", "examples/xlerobot", "docs/workflows/eval-grid-video", "docs/workflows/handover-dataset-curation",
          "docs/workflows/eval-scoring"):
    sys.path.insert(0, str(REPO / p))
from lerobot.scripts.lerobot_dataset_quality import episode_metrics, flag_outliers  # noqa: E402
import report  # noqa: E402
from make_grid import first_arm  # noqa: E402
from review_page import NAMES, pauses, propose, snaps  # noqa: E402
from score import GR, runs  # noqa: E402
from trim_stagnant_frames import keep_range, reach_start  # noqa: E402
sys.path.insert(0, str(HERE))
import curate  # noqa: E402

HOME = Path(os.path.expanduser("~/.cache/huggingface/lerobot"))
OWNER = HOME / "wwqcleverghost"
SPECS = REPO / "examples/xlerobot"
LABELS = SPECS / "dataset_labels.json"
OUT = REPO / "outputs/dataset_viewer"
REF = "v4-handover2-trimmed"  # human reference demos for the quality score (eval-scoring README)
PYTHON = sys.executable
lock = threading.RLock()  # freeze() loads datasets while holding it
_cache = {}


# ---------- datasets ----------

def scan():
    """Every LeRobot dataset under OWNER: id = path relative to OWNER without a trailing /dataset. Symlinked aliases
    (old names) are skipped, eval uploads nest the dataset in <dir>/dataset or <dir>/<set>/dataset."""
    out = []
    for top in sorted(OWNER.iterdir()):
        if top.is_symlink() or not top.is_dir():
            continue
        for info in sorted([*top.glob("meta/info.json"), *top.glob("dataset/meta/info.json"), *top.glob("*/dataset/meta/info.json")]):
            root = info.parent.parent
            rel = str(root.relative_to(OWNER))
            out.append(dict(id=re.sub(r"/dataset$", "", rel), root=root))
    return out


def is_eval(root):
    """An eval run: has success.csv. Runs with human takeovers (interventions.csv) are DAgger data, listed with the training sets."""
    return (root / "success.csv").exists() and not (root / "interventions.csv").exists()


def root_of(ds_id):
    p = Path(ds_id) if os.path.isabs(ds_id) else OWNER / ds_id
    p = p / "dataset" if (p / "dataset/meta/info.json").exists() else p
    p = p.resolve()
    assert (p / "meta/info.json").exists(), f"no dataset at {ds_id}"
    return p


def ds_id_of(source):
    """Spec source -> dataset id."""
    if source.get("root"):
        p = Path(os.path.expanduser(source["root"])).resolve()
        try:
            return re.sub(r"/dataset$", "", str(p.relative_to(OWNER.resolve())))
        except ValueError:
            return str(p)
    return source["repo"].split("/")[-1]


def building():
    """--dst of every trim_stagnant_frames.py process that is running now."""
    out = set()
    for f in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            a = open(f, "rb").read().decode(errors="ignore").split("\0")
        except OSError:
            continue
        if any(x.endswith("trim_stagnant_frames.py") for x in a) and "--dst" in a:
            out.add(a[a.index("--dst") + 1].split("/")[-1])
    return out


def revision(root):
    """HF commit the local copy was downloaded at (huggingface_hub keeps it in .cache), None for local builds."""
    for top in (root, root.parent, root.parent.parent):
        m = top / ".cache/huggingface/download" / (root / "meta/info.json").relative_to(top)
        m = m.with_name(m.name + ".metadata")
        if m.exists():
            return m.read_text().split("\n")[0].strip()
    return None


def build_record(ds_id):
    recs = sorted(glob.glob(str(OUT / "builds" / f"*_{ds_id}.json")))
    return json.load(open(recs[-1])) if recs else None


def ds_list():
    busy, out = building(), []
    for d in scan():
        is_busy = d["id"].split("/")[0] in busy
        if not is_busy and not glob.glob(str(d["root"] / "data/*/*.parquet")):
            continue  # meta only (e.g. an eval run that recorded nothing): nothing to show
        info = json.load(open(d["root"] / "meta/info.json"))
        rec = build_record(d["id"])
        out.append(dict(id=d["id"], episodes=info["total_episodes"], frames=info["total_frames"], fps=info["fps"],
                        dims=info["features"]["action"]["shape"][0], eval=is_eval(d["root"]), dagger=(d["root"] / "interventions.csv").exists(),
                        revision=revision(d["root"]), building=is_busy,
                        built=rec and dict(git=rec["git"], started=rec["started"], spec=rec["spec"])))
    return out


def load(ds_id):
    """Arrays of one dataset, cached until its data files change."""
    root = root_of(ds_id)
    files = sorted(glob.glob(str(root / "data/*/*.parquet")))
    csv = root / "success.csv"  # relabeling a trial must reload the labels too
    key = tuple((f, os.path.getmtime(f)) for f in files) + (os.path.getmtime(root / "meta/info.json"),
                                                            os.path.getmtime(csv) if csv.exists() else 0)
    with lock:
        if ds_id in _cache and _cache[ds_id]["key"] == key:
            return _cache[ds_id]
    info = json.load(open(root / "meta/info.json"))
    d = pd.concat(pd.read_parquet(f, columns=["index", "action", "observation.state"]) for f in files).sort_values("index")
    eps = pd.concat(map(pd.read_parquet, sorted(glob.glob(str(root / "meta/episodes/*/*.parquet"))))).sort_values("episode_index")
    eps = eps.set_index("episode_index")
    cams = sorted((k[len("observation.images."):] for k, v in info["features"].items() if v["dtype"] == "video"),
                  key=lambda c: (not c.startswith("head"), c))
    csv = root / "success.csv"
    obj = dict(key=key, root=root, info=info, fps=info["fps"], eps=eps, cams=cams,
               A=np.stack(d["action"].values).astype(float), S=np.stack(d["observation.state"].values).astype(float),
               success=pd.read_csv(csv).set_index("episode").success.to_dict() if csv.exists() else None)
    assert len(obj["A"]) == info["total_frames"], "data files incomplete (still being written?)"
    with lock:
        _cache[ds_id] = obj
    return obj


def ep_arrays(o, ep):
    lo, hi = int(o["eps"].loc[ep, "dataset_from_index"]), int(o["eps"].loc[ep, "dataset_to_index"])
    return o["A"][lo:hi, :12], o["S"][lo:hi, :12]


def videos(ds_id, o, ep):
    row, out = o["eps"].loc[ep], {}
    for c in o["cams"]:
        v = f"videos/observation.images.{c}"
        p = f"{v}/chunk-{int(row[f'{v}/chunk_index']):03d}/file-{int(row[f'{v}/file_index']):03d}.mp4"
        mt = int(os.path.getmtime(o["root"] / p))
        out[c] = dict(url=f"/video?id={ds_id}&p={p}&v={mt}", path=p, t0=float(row[f"{v}/from_timestamp"]),
                      t1=float(row[f"{v}/to_timestamp"]))
    return out


def episodes(ds_id):
    o = load(ds_id)
    labels = read_json(LABELS, {}).get(ds_id, {})
    out = []
    for ep in o["eps"].index:
        a, _ = ep_arrays(o, ep)
        out.append(dict(ep=int(ep), n=len(a), video=videos(ds_id, o, ep), arm=first_arm(a), label=labels.get(str(ep), {}),
                        success=None if o["success"] is None else int(o["success"].get(int(ep), 0))))
    return dict(id=ds_id, fps=o["fps"], cams=o["cams"], episodes=out)


def curves(ds_id, ep):
    a, s = ep_arrays(load(ds_id), ep)
    return dict(names=NAMES, action=np.round(a, 1).tolist(), state=np.round(s, 1).tolist())


def decode_frame(ds_id, cam, ep, f):
    """Frame f of an episode from the dataset's mp4, full size (PIL image)."""
    import av
    o = load(ds_id)
    v = videos(ds_id, o, ep)[cam]
    t = v["t0"] + (f + 0.5) / o["fps"]
    with av.open(str(o["root"] / v["path"])) as c:
        s = c.streams.video[0]
        c.seek(max(int((t - 1) / s.time_base), 0), stream=s)
        img = None
        for fr in c.decode(s):
            img = fr
            if fr.time >= t - 1 / o["fps"]:
                break
    return img.to_image()


def frame_jpg(ds_id, cam, ep, f, w=320):
    """One camera frame as a JPEG w pixels wide (posters / stills: 320, start / end review: 640), cached on disk."""
    w = 640 if w > 320 else 320
    out = OUT / "cache/frames" / ds_id.replace("/", "__") / f"{cam}_{ep}_{f}{'' if w == 320 else f'_w{w}'}.jpg"
    if out.exists():
        return out.read_bytes()
    buf = io.BytesIO()
    decode_frame(ds_id, cam, ep, f).resize((w, w * 3 // 4)).save(buf, "JPEG", quality=78 if w == 320 else 85)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buf.getvalue())
    return buf.getvalue()


# ---------- labels / specs ----------

def read_json(path, default):
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return default


def write_json(path, obj, spec=False):
    tmp = Path(str(path) + ".tmp")
    if spec:  # same layout as the hand-written specs: one source per line
        tmp.write_text("[\n" + ",\n".join(" " + json.dumps(s, ensure_ascii=False) for s in obj) + "\n]\n")
    else:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def set_label(ds_id, ep, patch):
    with lock:
        lab = read_json(LABELS, {})
        e = lab.setdefault(ds_id, {}).setdefault(str(ep), {})
        e.update(patch)
        for k in [k for k, v in e.items() if v in ("", [], None)]:
            del e[k]
        if not e:
            del lab[ds_id][str(ep)]
        if not lab[ds_id]:
            del lab[ds_id]
        write_json(LABELS, lab)
        return e


def spec_path(name):
    p = (SPECS / name).resolve()
    assert p.parent == SPECS.resolve() and p.suffix == ".json" and p != LABELS.resolve(), name
    return p


def spec_list():
    out = []
    for p in sorted(SPECS.glob("*.json")):
        try:
            s = json.load(open(p))
        except ValueError:
            continue
        if isinstance(s, list) and s and all(isinstance(x, dict) and "repo" in x for x in s):
            out.append(dict(name=p.name, sources=[ds_id_of(x) for x in s], segments=any("segments" in x for x in s)))
    return out


def label_drop(lab):
    """An episode the user dropped in the labels: forced drop, or a manual start / end mark (standing / basket) not forced kept.
    Shared by the trim review (its "drop" button) and the curate rules, so both pages show the same decision."""
    return lab.get("force") == "drop" or (lab.get("manual") in curate.MANUAL and lab.get("force") != "keep")


def rule_eps(o, source, reset_tol, to_target, snap_rule):
    """The rule's cut for every episode of one source (what review_page.py shows)."""
    start_s = {int(k): v for k, v in source.get("start_s", {}).items()}
    out = []
    for ep in o["eps"].index:
        a, s = ep_arrays(o, ep)
        f, e = keep_range(a, s, list(range(12)), 0.2, 2.0, reset_tol)
        if to_target:
            f = reach_start(a, s, f, e)
        f = min(f + round(start_s.get(int(ep), 0) * o["fps"]), e - 1)
        sn = snaps(a, s, 0, len(a))
        pf, pe, pdrop = propose(f, e, [x for x in sn if f <= x[0] < e]) if snap_rule else (f, e, False)
        out.append(dict(ep=int(ep), n=len(a), first=f, end=e, pfirst=pf, pend=pe, pdrop=bool(pdrop), snaps=sn,
                        pauses=pauses(a, f, e), arm=first_arm(a)))
    return out


def range_q(a, s, f, e):
    """report.py quality of the kept frames [f, e) vs the human demos: [q, smooth, speed, idle]. Speed is NaN (not
    counted) where the kept part is not a whole task, e.g. a takeover segment."""
    q, sub = report.quality(pd.DataFrame([report.trial(s[f:e], a[f:e])]), ref())
    return [float(q[0]), *map(float, sub.iloc[0][["smooth", "speed", "idle"]])]


def seg_cards(name, source, labels):
    """One card per segment of a segment source (takeover corrections); the "rule" range is the segment as first
    written (spec_orig), labels are keyed "<ep>.<segment>"."""
    ds_id, orig = ds_id_of(source), OUT / "spec_orig" / name
    o = load(ds_id)
    first = next(s for s in json.load(open(orig)) if ds_id_of(s) == ds_id) if orig.exists() else source
    out = []
    for k, segs in source["segments"].items():
        a, s = ep_arrays(o, int(k))
        for i, (f, e) in enumerate(first["segments"][k]):
            cur = segs[i]
            out.append(dict(ep=int(k), seg=i, n=len(a), first=f, end=e, pfirst=f, pend=e, pdrop=False, snaps=snaps(a, s, 0, len(a)),
                            pauses=pauses(a, *cur), arm=first_arm(a), q=range_q(a, s, *cur), ds=ds_id, repo=source["repo"], video=videos(ds_id, o, int(k)),
                            label=labels.get(ds_id, {}).get(f"{k}.{i}", {}),
                            edit=dict(range=cur, cut=[], drop=i in source.get("drop_seg", {}).get(k, []))))
    return out


def review(name, reset_tol, to_target, snap_rule):
    spec, labels, out = json.load(open(spec_path(name))), read_json(LABELS, {}), []
    for source in spec:
        if "segments" in source:
            out += seg_cards(name, source, labels)
            continue
        ds_id = ds_id_of(source)
        o = load(ds_id)
        rng, cut, drop = source.get("range", {}), source.get("cut", {}), set(source.get("drop", []))
        for r in rule_eps(o, source, reset_tol, to_target, snap_rule):
            k = str(r["ep"])
            lab = labels.get(ds_id, {}).get(k, {})
            a, s = ep_arrays(o, r["ep"])
            r.update(ds=ds_id, repo=source["repo"], video=videos(ds_id, o, r["ep"]), label=lab,
                     q=range_q(a, s, *(rng.get(k) or (r["pfirst"], r["pend"]))),
                     edit=dict(range=rng.get(k), cut=cut.get(k, []), drop=r["ep"] in drop or label_drop(lab)))
            out.append(r)
    return dict(spec=name, fps=30, episodes=out, sources=[dict(repo=s["repo"], ds=ds_id_of(s), dst=s.get("dst", "")) for s in spec],
                rule=next((s["rule"] for s in spec if "rule" in s), None))


def edit_spec(name, ds, ep, edit, seg=None):
    """edit = {a, b, cuts, drop} writes this episode's exact range; edit = None resets it to the original spec.
    seg = index of the segment in a segment source: its [a, b] goes into "segments", drop into "drop_seg"."""
    p, k = spec_path(name), str(ep)
    with lock:
        spec = json.load(open(p))
        orig = OUT / "spec_orig" / name
        if not orig.exists():  # first write ever: keep the hand-written version for "reset"
            orig.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, orig)
        src = next(s for s in spec if ds_id_of(s) == ds)  # not by repo: set1 / set2 of one eval share it
        if seg is not None:
            if edit is None:
                o = next(s for s in json.load(open(orig)) if ds_id_of(s) == ds)
                (a, b), drop = o["segments"][k][seg], seg in o.get("drop_seg", {}).get(k, [])
            else:
                a, b, drop = int(edit["a"]), int(edit["b"]), bool(edit["drop"])
            assert 0 <= a < b, (a, b)
            src["segments"][k][seg] = [a, b]
            sd = src.setdefault("drop_seg", {})
            d = set(sd.get(k, [])) - {seg} | ({seg} if drop else set())
            if d:
                sd[k] = sorted(d)
            else:
                sd.pop(k, None)
            if not sd:
                del src["drop_seg"]
            write_json(p, spec, spec=True)
            return dict(range=[a, b], cut=[], drop=drop)
        if edit is None:
            o = next((s for s in json.load(open(orig)) if ds_id_of(s) == ds), {})
            edit = dict(range=o.get("range", {}).get(k), cuts=o.get("cut", {}).get(k, []), drop=ep in o.get("drop", []))
        else:
            assert 0 <= edit["a"] < edit["b"], edit
            cuts = sorted([max(c0, edit["a"]), min(c1, edit["b"])] for c0, c1 in edit["cuts"])
            edit = dict(range=[int(edit["a"]), int(edit["b"])], cuts=[c for c in cuts if c[0] < c[1]], drop=bool(edit["drop"]))
        for key, val in (("range", edit["range"]), ("cut", edit["cuts"])):
            d = src.setdefault(key, {})
            if val:
                d[k] = val
            else:
                d.pop(k, None)
            if not d:
                del src[key]
        # the drop decision lives in the labels too (shared with the curate rules on the quality page)
        lab = read_json(LABELS, {}).get(ds_id_of(src), {}).get(k, {})
        if edit["drop"] and not label_drop(lab):
            set_label(ds_id_of(src), ep, {"force": "drop"})
        elif not edit["drop"] and label_drop(lab):
            set_label(ds_id_of(src), ep, {"force": "", "manual": "" if lab.get("manual") in curate.MANUAL else lab.get("manual")})
        drop = set(src.get("drop", [])) - {ep} | ({ep} if edit["drop"] else set())
        if drop:
            src["drop"] = sorted(drop)
        else:
            src.pop("drop", None)
        write_json(p, spec, spec=True)
        return dict(range=edit["range"], cut=edit["cuts"], drop=edit["drop"])


# ---------- rebuild ----------

def git_info():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True,
                                text=True).stdout.strip())
    return dict(commit=head, dirty=dirty)


def freeze(name, rule):
    """Write the exact range of every episode into the spec (edited ones keep theirs, the rest get the rule's),
    so the rebuild is exactly what the page shows whatever the CLI defaults are."""
    p = spec_path(name)
    with lock:
        spec = json.load(open(p))
        for source in spec:
            if "segments" in source:
                continue
            o = load(ds_id_of(source))
            rng, drop = source.setdefault("range", {}), set(source.get("drop", []))
            labels = read_json(LABELS, {}).get(ds_id_of(source), {})
            drop |= {int(e) for e, lab in labels.items() if label_drop(lab)}
            for r in rule_eps(o, source, rule["reset_tol"], rule["to_target"], rule["snap_rule"]):
                k = str(r["ep"])
                if k not in rng:
                    rng[k] = [r["pfirst"], r["pend"]]
                    if r["pdrop"]:
                        drop.add(r["ep"])
            if drop:
                source["drop"] = sorted(drop)
            source["rule"] = rule
        write_json(p, spec, spec=True)
        return spec


def rebuild(name, rule, dsts, overwrite, arm_only):
    """dsts = {repo: dst name}; sources with the same dst are merged into one dataset. Starts one detached
    (setsid) trim_stagnant_frames.py per dst; logs in outputs/dataset_viewer/builds."""
    busy = building()
    groups = {}
    for repo, dst in dsts.items():
        assert re.fullmatch(r"[A-Za-z0-9._-]+", dst), f"bad name {dst}"
        assert dst not in busy, f"{dst} is being built right now"
        groups.setdefault(dst, []).append(repo)
    spec = freeze(name, rule)
    with lock:  # remember each source's dst in the spec
        for s in spec:
            if s["repo"] in dsts:
                s["dst"] = dsts[s["repo"]]
        write_json(spec_path(name), spec, spec=True)
    (OUT / "builds").mkdir(parents=True, exist_ok=True)
    jobs, ts, git = [], time.strftime("%Y%m%d_%H%M%S"), git_info()
    for dst, repos in groups.items():
        if (OWNER / dst).exists():
            assert overwrite, f"{dst} exists (tick overwrite: the old copy is renamed, not deleted)"
            os.rename(OWNER / dst, OWNER / f"{dst}.old_{ts}")
        base = OUT / "builds" / f"{ts}_{dst}"
        write_json(f"{base}.spec.json", [s for s in spec if s["repo"] in repos], spec=True)
        cmd = [PYTHON, "examples/xlerobot/trim_stagnant_frames.py", "--spec", f"{base}.spec.json",
               "--dst", f"wwqcleverghost/{dst}", "--reset_tol", str(rule["reset_tol"]), "--num_workers", "6"]
        cmd += ["--to_target"] * rule["to_target"] + ["--arm_only"] * arm_only
        env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
        with open(f"{base}.log", "w") as log:
            pid = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True).pid
        rec = dict(dst=dst, spec=name, cmd=" ".join(cmd), log=f"{base}.log", pid=pid, started=ts, git=git,
                   sources=repos, rule=rule, arm_only=arm_only)
        write_json(f"{base}.json", rec)
        jobs.append(rec)
    return jobs


def builds():
    out = []
    for f in sorted(glob.glob(str(OUT / "builds/*_*.json")))[::-1]:
        if f.endswith(".spec.json"):
            continue
        r = json.load(open(f))
        try:
            alive = "trim_stagnant_frames" in open(f"/proc/{r['pid']}/cmdline").read()
        except OSError:
            alive = False
        tail = open(r["log"], errors="ignore").read()[-6000:].replace("\r", "\n").split("\n")
        tail = [x for x in tail if x.strip()][-25:]
        r.update(running=alive, ok=any(x.startswith("wrote ") for x in tail), tail=tail)
        out.append(r)
        if len(out) == 8:
            break
    return out


# ---------- quality / eval ----------

def ref():
    if "ref" not in _cache:
        df, _ = report.trials(str(root_of(REF)))
        _cache["ref"] = {"rev_rate": float(df.rev_rate.median()), "task_s": float(df.task_s.median())}
    return _cache["ref"]


def scored(ds_id):
    """report.trials + quality per episode, cached with the dataset."""
    o = load(ds_id)
    if "trials" not in o:
        df, is_eval = report.trials(str(o["root"]))
        q, sub = report.quality(df, ref())
        o["trials"] = (df.assign(q=q.values, q_smooth=sub.smooth.values, q_speed=sub.speed.values, q_idle=sub.idle.values),
                       is_eval)
    return o["trials"]


def _quality(ds_id):
    o = load(ds_id)
    m = {}
    for ep in o["eps"].index:
        a, s = ep_arrays(o, ep)
        m[int(ep)] = episode_metrics(a, o["fps"], s)
    flags = flag_outliers(m)
    df, is_eval = scored(ds_id)
    tr = df.set_index("ep")
    rows = [dict(ep=ep, **{k: v for k, v in x.items() if k != "final"}, rev_rate=tr.rev_rate.get(ep), q=tr.q.get(ep),
                 task_s=tr.task_s.get(ep), flags=flags.get(ep, [])) for ep, x in m.items()]
    report.rng = np.random.default_rng(0)  # same CI as running report.py on this dataset alone
    s = report.summarize(ds_id, df, False, ref())
    med = {k: float(np.median([r[k] for r in rows if r[k] is not None and not np.isnan(r[k])] or [np.nan]))
           for k in ("duration_s", "median_jerk", "p95_jerk", "max_velocity", "static_fraction", "max_pause_s", "rev_rate", "q")}
    return dict(id=ds_id, eval=is_eval, rows=rows, quality=[s["quality"], s["quality_lo"], s["quality_hi"]], ref=ref(),
                summary=dict(id=ds_id, n=len(rows), flagged=sum(bool(r["flags"]) for r in rows),
                             quality=[s["quality"], s["quality_lo"], s["quality_hi"]], **med))


def fail_time(st, ac, fps):
    """Rough moment a failed trial went wrong: the end of its last grasp (tape dropped / released), else 3 s before the end."""
    ends = [e for g in GR.values() for _, e in runs((ac[:, g] < st[:, g] - 2.0) & (st[:, g] > 4))]
    return max(max(ends) - fps // 2, 0) if ends else max(len(st) - 3 * fps, 0)


def _evaluation(ds_id):
    o = load(ds_id)
    df, is_eval = scored(ds_id)
    assert is_eval, f"{ds_id} has no success.csv"
    report.rng = np.random.default_rng(0)
    s = report.summarize(ds_id, df, True, ref())
    _, k, n = report.completion(df)
    stages = [dict(stage=st, k=int(ki), n=int(ni), rate=(ki + 1) / (ni + 2), ci=report.wilson(ki, ni))
              for st, ki, ni in zip(report.STAGES, k, n)]
    trials = []
    for _, r in df.iterrows():
        a, st = ep_arrays(o, int(r.ep))
        trials.append(dict(ep=int(r.ep), label=int(o["success"].get(int(r.ep), 0)), stages=[int(r[x]) for x in report.STAGES],
                           success=int(r.success), q=r.q, q_smooth=r.q_smooth, q_speed=r.q_speed, q_idle=r.q_idle,
                           task_s=r.task_s, rev_rate=r.rev_rate, idle_frac=r.idle_frac, n=len(a),
                           fail=None if r.success else fail_time(st, a, o["fps"])))
    ci = lambda k, lo, hi: [s[k], s[lo], s[hi]]  # noqa: E731
    out = dict(id=ds_id, n=len(df), success=s["success"], stages=stages,
               completion=ci("completion", "completion_lo", "completion_hi"), quality=ci("quality", "quality_lo", "quality_hi"),
               total=ci("total", "total_lo", "total_hi"), success_rate=ci("success_rate", "success_lo", "success_hi"))
    return dict(out, trials=trials, summary=out, weights=dict(completion=report.W_COMPLETION, quality=report.W_QUALITY))


# Scores are cached on disk (outputs/dataset_viewer/cache/scores) until the dataset's files or a scoring script change,
# so the history tables open at once; a background thread fills in what is missing.
CODE = [REPO / f for f in ("docs/workflows/eval-scoring/report.py", "docs/workflows/eval-scoring/score.py",
                           "docs/workflows/eval-scoring/smooth.py", "src/lerobot/scripts/lerobot_dataset_quality.py")]
CURATE_CODE = CODE + [REPO / f for f in ("docs/workflows/dataset-viewer/curate.py", "docs/workflows/handover-dataset-curation/review_page.py",
                                          "docs/workflows/handover-dataset-curation/ep_score.py")]
SCORE_VERSION = 1  # bump when _quality / _evaluation change what they return
score_lock = threading.Lock()  # report.rng is global: one scoring at a time
_failed = {}


def score_key(ds_id, kind="quality"):
    root = root_of(ds_id)
    files = sorted(glob.glob(str(root / "data/*/*.parquet"))) + [str(root / "meta/info.json"), str(root / "success.csv"),
                                                                  str(root_of(REF) / "meta/info.json")]
    stamp = [(f, os.path.getmtime(f) if os.path.exists(f) else 0) for f in files] + [os.path.getmtime(f) for f in (CURATE_CODE if kind == "curate" else CODE)] + [SCORE_VERSION]
    return hashlib.md5(repr(stamp).encode()).hexdigest()


def cached(kind, ds_id, compute=True):
    path = OUT / "cache/scores" / f"{ds_id.replace('/', '__')}.{kind}.json"
    key = score_key(ds_id, kind)
    try:
        c = json.load(open(path))
        if c["key"] == key:
            return c["value"]
    except (FileNotFoundError, ValueError, KeyError):
        pass
    if not compute:
        return None
    with score_lock:
        value = jsonable({"quality": _quality, "eval": _evaluation, "curate": _curate}[kind](ds_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"key": key, "value": value})
    return value


def _curate(ds_id):
    """curate.py metrics of every episode (arm arrays + first / last head-camera frame)."""
    o = load(ds_id)
    head, rows = o["cams"][0], []
    for ep in o["eps"].index:
        a, s = ep_arrays(o, ep)
        start, end = (np.asarray(decode_frame(ds_id, head, int(ep), f).convert("RGB")) for f in (0, len(a) - 1))
        rows.append(dict(ep=int(ep), **curate.episode(a, s, o["fps"]), **curate.vision(start, end)))
    return dict(id=ds_id, fps=o["fps"], rows=rows)


def curate_run(ids, preset):
    """Same evaluation for the web page and curate.py: rows (kept, reasons), cumulative counts, the reference medians."""
    p = curate.full(preset)
    metrics = {i: cached("curate", i)["rows"] for i in ids}
    refs = {k: float(np.median([m[col] for m in cached("curate", p[k]["ref"])["rows"]]))
            for k, col in (("rev", "rev_rate"), ("jerk", "jerk")) if p[k]["on"] and p[k].get("ref")}
    labels = read_json(LABELS, {})
    force = {i: {int(e): v["force"] for e, v in labels.get(i, {}).items() if v.get("force")} for i in ids}
    manual = {i: {int(e): v["manual"] for e, v in labels.get(i, {}).items() if v.get("manual")} for i in ids}
    rows, steps = curate.evaluate(metrics, preset, refs, force, manual)
    for r in rows:
        r["label"] = labels.get(r["ds"], {}).get(str(r["ep"]), {})
    return dict(rows=rows, steps=steps, refs=refs, preset=p)


def curate_spec(ids, preset, name, preset_name=""):
    """Trim spec with the kept episodes of each dataset, whole (they are already trimmed): range [0, n), the rest dropped."""
    assert re.fullmatch(r"[A-Za-z0-9._-]+", name), f"bad spec name {name}"
    p = spec_path(f"{name}.json")
    assert not p.exists(), f"{p.name} exists, pick another name"
    res, spec = curate_run(ids, preset), []
    for ds in ids:
        rows = [r for r in res["rows"] if r["ds"] == ds]
        src = {"repo": f"wwqcleverghost/{ds.split('/')[0]}"}
        if "/" in ds:
            src["root"] = str(root_of(ds))
        src["note"] = (f"curated {time.strftime('%Y-%m-%d')} with preset {preset_name or '-'}: " +
                       json.dumps({k: v for k, v in res["preset"].items() if v["on"]}, ensure_ascii=False))
        src["drop"] = [r["ep"] for r in rows if not r["kept"]]
        src["range"] = {str(r["ep"]): [0, r["n"]] for r in rows if r["kept"]}
        src["dst"] = name  # one merged dataset by default; the rebuild dialog can change it
        spec.append(src)
    write_json(p, spec, spec=True)
    return p.name


def save_preset(name, preset):
    assert re.fullmatch(r"[A-Za-z0-9._\-\u4e00-\u9fff]+", name), f"bad preset name {name}"
    curate.PRESETS.mkdir(exist_ok=True)
    write_json(curate.PRESETS / f"{name}.json", curate.full(preset))
    return curate.presets()


def with_labels(ds_id, value, key):
    labels = read_json(LABELS, {}).get(ds_id, {})
    for r in value[key]:
        r["label"] = labels.get(str(r["ep"]), {})
    return value


def quality(ds_id):
    return with_labels(ds_id, cached("quality", ds_id), "rows")


def evaluation(ds_id):
    return with_labels(ds_id, cached("eval", ds_id), "trials")


def version(ds_id):
    m = re.match(r"v(\d+)-", ds_id)
    return int(m[1]) if m else 0


_worker = {"thread": None, "todo": []}


def history(kind):
    """Summary row of every dataset of this kind (training/demo or eval); missing ones are computed in the background."""
    busy, rows, pending = building(), [], []
    for d in scan():
        if is_eval(d["root"]) != (kind == "eval") or d["id"].split("/")[0] in busy:
            continue
        err = _failed.get((kind, d["id"]))
        if err and err[0] == score_key(d["id"], kind):  # retried once the data or a scoring script changes
            rows.append(dict(id=d["id"], version=version(d["id"]), error=err[1]))
            continue
        v = cached(kind, d["id"], compute=False)
        if v is None:
            pending.append(d["id"])
        else:
            rows.append(dict(v["summary"], version=version(d["id"])))
    if pending:
        with lock:
            _worker["todo"] = [(kind, i) for i in pending]
            if not (_worker["thread"] and _worker["thread"].is_alive()):
                _worker["thread"] = threading.Thread(target=fill, daemon=True)
                _worker["thread"].start()
    return dict(kind=kind, rows=rows, pending=pending)


def fill():
    while True:
        with lock:
            if not _worker["todo"]:
                return
            kind, ds_id = _worker["todo"].pop(0)
        try:
            cached(kind, ds_id)
            _failed.pop((kind, ds_id), None)
        except Exception as e:
            _failed[(kind, ds_id)] = (score_key(ds_id, kind), f"{type(e).__name__}: {e}")


# ---------- http ----------

def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if math.isnan(x) or math.isinf(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def new_spec(ds_id):
    """A trim spec for one dataset, to review it on the trim page."""
    name = f"{ds_id.replace('/', '_')}_trim.json"
    p = spec_path(name)
    if not p.exists():
        src = {"repo": f"wwqcleverghost/{ds_id.split('/')[0]}"}
        if "/" in ds_id:
            src["root"] = str(root_of(ds_id))
        write_json(p, [src], spec=True)
    return dict(name=name)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if not self.path.startswith(("/video", "/api/frame", "/api/builds")):
            sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def send(self, code, body, ctype="application/json", headers=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def api(self, fn):
        try:
            body = json.dumps(jsonable(fn()), ensure_ascii=False, allow_nan=False).encode()
        except Exception as e:  # shown on the page
            import traceback
            traceback.print_exc()
            return self.send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False).encode())
        try:
            self.send(200, body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        static = {"/": ("index.html", "text/html"), "/index.html": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript")}
        if u.path in static:
            f, t = static[u.path]
            return self.send(200, (HERE / f).read_bytes(), t + "; charset=utf-8", [("Cache-Control", "no-cache")])
        if u.path == "/video":
            return self.video(q)
        if u.path == "/api/frame":
            try:
                body = frame_jpg(q["id"], q["cam"], int(q["ep"]), int(q["f"]), int(q.get("w", 320)))
            except Exception as e:
                return self.send(404, str(e).encode(), "text/plain")
            try:
                return self.send(200, body, "image/jpeg", [("Cache-Control", "max-age=86400")])
            except (BrokenPipeError, ConnectionResetError):
                return
        routes = {
            "/api/datasets": lambda: dict(datasets=ds_list(), git=git_info()),
            "/api/episodes": lambda: episodes(q["id"]),
            "/api/curves": lambda: curves(q["id"], int(q["ep"])),
            "/api/specs": spec_list,
            "/api/review": lambda: review(q["spec"], float(q.get("reset_tol", 10)), q.get("to_target") == "1",
                                          q.get("snap_rule") == "1"),
            "/api/builds": builds,
            "/api/quality": lambda: quality(q["id"]),
            "/api/eval": lambda: evaluation(q["id"]),
            "/api/history": lambda: history(q["kind"]),
            "/api/curate/presets": lambda: dict(presets=curate.presets(), rules=curate.RULES, default=curate.DEFAULT),
        }
        if u.path in routes:
            return self.api(routes[u.path])
        self.send(404, b"not found", "text/plain")

    def do_POST(self):
        u = urlparse(self.path)
        b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        routes = {
            "/api/label": lambda: set_label(b["id"], b["ep"], b["patch"]),  # ep or "<ep>.<segment>"
            "/api/review/edit": lambda: edit_spec(b["spec"], b["ds"], int(b["ep"]), b.get("edit"), b.get("seg")),
            "/api/rebuild": lambda: rebuild(b["spec"], b["rule"], b["dsts"], b.get("overwrite", False), b.get("arm_only", False)),
            "/api/spec/new": lambda: new_spec(b["id"]),
            "/api/curate/run": lambda: curate_run(b["ids"], b["preset"]),
            "/api/curate/spec": lambda: dict(name=curate_spec(b["ids"], b["preset"], b["name"], b.get("preset_name", ""))),
            "/api/curate/preset": lambda: save_preset(b["name"], b["preset"]),
        }
        if u.path in routes:
            return self.api(routes[u.path])
        self.send(404, b"not found", "text/plain")

    def video(self, q):
        """A dataset's own mp4 with HTTP Range (the browser seeks with it; raw recordings keep the moov at the end)."""
        try:
            root = root_of(q["id"])
            p = (root / q["p"]).resolve()
            assert p.is_relative_to(root) and p.suffix == ".mp4"
            size = p.stat().st_size
        except Exception:
            return self.send(404, b"no video", "text/plain")
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m[1] or m[2]):
            start, end = (int(m[1]), int(m[2] or size - 1)) if m[1] else (size - int(m[2]), size - 1)
            # at most 2 MB per answer: the browser has 6 connections per host, and a player that streams a whole
            # 100 MB file would hold one; with short answers dozens of players share them (it asks for the rest itself)
            end = min(end, size - 1, start + (2 << 20) - 1)
        n = end - start + 1
        try:
            self.send_response(206 if m else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(n))
            self.send_header("Cache-Control", "max-age=31536000, immutable")  # the url carries the file's mtime
            if m:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(p, "rb") as f:
                f.seek(start)
                while n > 0:
                    chunk = f.read(min(n, 1 << 20))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    n -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


def selfcheck():
    assert label_drop({"force": "drop"}) and label_drop({"manual": "basket"}) and not label_drop({"manual": "ok"})
    assert not label_drop({"manual": "standing", "force": "keep"})
    assert jsonable({"a": float("nan"), "b": np.int64(3), "c": [np.float32(1.5)]}) == {"a": None, "b": 3, "c": [1.5]}
    # fail_time: left gripper holds the tape (command 1, state 8) frames 10-40, then drops it
    st, ac = np.zeros((100, 12)), np.zeros((100, 12))
    st[:, 5], ac[:, 5] = 20, 20
    st[10:40, 5], ac[10:40, 5] = 8, 1
    assert fail_time(st, ac, 30) == 40 - 15, fail_time(st, ac, 30)
    assert fail_time(np.zeros((200, 12)), np.zeros((200, 12)), 30) == 110
    assert version("v6-handover-curated") == 6 and version("v10-x") == 10 and version("handover2") == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8770)))
    ap.add_argument("--host", default="127.0.0.1", help="listen address; use the Tailscale IP to open it on a phone (no login, so never 0.0.0.0)")
    x = ap.parse_args()
    selfcheck()
    print(f"dataset viewer on http://{x.host}:{x.port}", flush=True)
    ThreadingHTTPServer((x.host, x.port), Handler).serve_forever()
