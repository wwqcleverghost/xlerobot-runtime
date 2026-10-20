"""Per-episode quality metrics for recorded datasets, adapted from huggingface/lerobot#3760 (PR #3762).

Changes vs upstream: only the 12 arm dims (head/base are dropped from training), jerk is the true
3rd derivative (upstream's "jerk" is the 2nd diff = acceleration), units are per second, and the
static threshold is a fixed speed so datasets are comparable with each other.
Usage: python src/lerobot/scripts/lerobot_dataset_quality.py <dataset_root> [<dataset_root> ...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ARM_DIMS = slice(0, 12)
# ponytail: fixed threshold in normalized units/s (~5% of median arm speed on v5); tune per robot
STATIC_SPEED = 3.0
K_IQR = 1.5
KEYS = ["duration_s", "median_jerk", "p95_jerk", "max_velocity", "static_fraction", "max_pause_s"]


def load_actions(root: Path) -> tuple[dict[int, np.ndarray], int]:
    fps = json.loads((root / "meta/info.json").read_text())["fps"]
    df = pd.concat(
        pd.read_parquet(p, columns=["episode_index", "frame_index", "action"])
        for p in sorted((root / "data").glob("*/*.parquet"))
    ).sort_values(["episode_index", "frame_index"])
    eps = {int(ep): np.stack(g["action"].values).astype(np.float64)[:, ARM_DIMS]
           for ep, g in df.groupby("episode_index")}
    return eps, fps


def episode_metrics(a: np.ndarray, fps: int) -> dict:
    v = np.linalg.norm(np.diff(a, axis=0), axis=1) * fps
    j = np.linalg.norm(np.diff(a, n=3, axis=0), axis=1) * fps**3
    static = v < STATIC_SPEED
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, static.astype(int), 0])))[::2]  # static run lengths
    return {
        "duration_s": len(a) / fps,
        "median_jerk": float(np.median(j)),
        "p95_jerk": float(np.percentile(j, 95)),
        "max_velocity": float(v.max()),
        "static_fraction": float(static.mean()),
        "max_pause_s": float(runs.max() / fps) if len(runs) else 0.0,
        "final": a[-1],
    }


def flag_outliers(m: dict[int, dict]) -> dict[int, list[str]]:
    flags = defaultdict(list)
    eps = list(m)
    cols = {k: [m[e][k] for e in eps] for k in KEYS}
    final = np.stack([m[e]["final"] for e in eps])
    cols["final_dist"] = np.linalg.norm(final - final.mean(0), axis=1)
    for name, vals in cols.items():
        q1, q3 = np.percentile(vals, [25, 75])
        hi, lo = q3 + K_IQR * (q3 - q1), q1 - K_IQR * (q3 - q1)
        for e, x in zip(eps, vals):
            if x > hi:
                flags[e].append(f"{name}↑")
            elif name == "duration_s" and x < lo:
                flags[e].append(f"{name}↓")
    return dict(flags)


def report(root: Path) -> dict:
    eps, fps = load_actions(root)
    m = {e: episode_metrics(a, fps) for e, a in eps.items()}
    flags = flag_outliers(m)
    print(f"\n=== {root.name}  ({len(m)} eps, {fps} fps) ===")
    print(f"{'ep':>3} {'dur':>6} {'medJerk':>9} {'p95Jerk':>9} {'maxVel':>7} {'static':>7} {'maxPause':>8}  flags")
    for e, x in m.items():
        print(f"{e:>3} {x['duration_s']:6.1f} {x['median_jerk']:9.0f} {x['p95_jerk']:9.0f} {x['max_velocity']:7.0f} "
              f"{x['static_fraction']:7.1%} {x['max_pause_s']:8.1f}  {' '.join(flags.get(e, []))}")
    ranked = sorted(flags.items(), key=lambda kv: -len(kv[1]))
    print(f"flagged {len(flags)}/{len(m)}; worst: " + ", ".join(f"ep{e}({len(f)})" for e, f in ranked[:10]))
    return {"name": root.name, "n": len(m), "flagged": len(flags),
            **{k: float(np.median([x[k] for x in m.values()])) for k in KEYS}}


def _selfcheck():
    ramp = np.linspace(0, 10, 31)[:, None].repeat(12, 1)  # 30 static frames, then constant velocity
    x = episode_metrics(np.r_[np.zeros((30, 12)), ramp], 30)
    assert x["median_jerk"] < 1e-6 and abs(x["max_pause_s"] - 30 / 30) < 0.05, x
    assert 0.4 < x["static_fraction"] < 0.6, x


if __name__ == "__main__":
    _selfcheck()
    rows = [report(Path(p)) for p in sys.argv[1:]]
    print("\n=== dataset summary (median over episodes) ===")
    print(f"{'dataset':28} {'n':>3} {'flag':>4} {'dur':>6} {'medJerk':>8} {'p95Jerk':>8} {'maxVel':>7} {'static':>7} {'maxPause':>8}")
    for r in rows:
        print(f"{r['name']:28} {r['n']:>3} {r['flagged']:>4} {r['duration_s']:6.1f} {r['median_jerk']:8.0f} "
              f"{r['p95_jerk']:8.0f} {r['max_velocity']:7.0f} {r['static_fraction']:7.1%} {r['max_pause_s']:8.1f}")
