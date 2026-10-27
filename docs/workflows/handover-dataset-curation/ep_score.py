"""The per-episode checks of README.md, automated for the raw handover datasets (handover2, handover1): trim range,
final placement (tape pixels in the last head frame), tape standing at the start (tall blob), gripper closes per arm,
first holding arm vs the midline rule, duration / longest pause and the IQR flags of lerobot_dataset_quality.py,
plus start / release / end head-camera sheets to confirm by eye.
usage: PYTHONPATH=src:docs/workflows/eval-scoring:examples/xlerobot python docs/workflows/handover-dataset-curation/ep_score.py <out_dir>
Writes <out_dir>/episodes.csv and start|release|end head-camera sheets (25 episodes per jpg)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from trim_stagnant_frames import keep_range
from score import score, runs, GAP
sys.path.insert(0, 'src/lerobot/scripts')
from lerobot_dataset_quality import episode_metrics, flag_outliers

HEAD = "observation.images.head(RGDB)"
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
POS = list(range(12))


def closes(cmd):   # number of open->closed command transitions (open > 10, closed < 10)
    o = cmd > 10
    return int((o[:-1] & ~o[1:]).sum())


def yellow(img):   # tape pixels (yellow) in a head frame: count and centroid
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    m = (r > 120) & (np.minimum(r, g) - b > 40)
    ys, xs = np.nonzero(m)
    if len(xs) < 50: return int(m.sum()), -1, -1, 0
    x0, x1, y0, y1 = *np.percentile(xs, [5, 95]), *np.percentile(ys, [5, 95])
    return int(m.sum()), float(np.median(xs)), float(np.median(ys)), round((y1 - y0) / max(x1 - x0, 1), 2)


if __name__ == '__main__':
    OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for repo in ["handover2", "handover1"]:
        ds = LeRobotDataset(f"wwqcleverghost/{repo}")
        t = ds.hf_dataset.with_format("numpy")
        A, S = np.asarray(t["action"]), np.asarray(t["observation.state"])
        bounds = list(zip(ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"]))
        thumbs, metrics = [], {}
        for ep, (lo, hi) in enumerate(bounds):
            f, e = keep_range(A[lo:hi], S[lo:hi], POS, 0.2, 2.0, 3.0)
            a, s = A[lo + f:lo + e], S[lo + f:lo + e]
            r = score(s, a)
            q = episode_metrics(a[:, :12].astype(float), ds.fps, s[:, :12].astype(float))
            rel = int(r['done'] * ds.fps) if r['done'] else len(a) - 1
            frames = {k: (ds[lo + f + i][HEAD].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                      for k, i in [('start', 0), ('release', min(rel + 15, len(a) - 1)), ('end', len(a) - 1)]}
            ys = {k: yellow(v) for k, v in frames.items()}
            hold = {k: runs((a[:, g] < s[:, g] - GAP) & (s[:, g] > 4)) for k, g in (('L', 5), ('R', 11))}
            first_arm = min((v[0][0], k) for k, v in hold.items() if v)[1] if any(hold.values()) else ''
            metrics[ep] = q
            rows.append(dict(repo=repo, ep=ep, first=f, end=e, raw_s=round((hi - lo) / ds.fps, 1),
                             dur=round(len(a) / ds.fps, 1), stages=''.join(map(str, r['stage'])), total=r['total'],
                             done_s=r['done'], holds=r['grasps'], closesL=closes(a[:, 5]), closesR=closes(a[:, 11]),
                             idle=r['idle'], medJerk=round(q['median_jerk']), p95Jerk=round(q['p95_jerk']),
                             maxVel=round(q['max_velocity']), static=round(q['static_fraction'], 3),
                             maxPause=round(q['max_pause_s'], 1), tail_s=round((len(a) - 1 - rel) / ds.fps, 1),
                             tape_x_start=round(ys['start'][1] / frames['start'].shape[1], 2), first_arm=first_arm, y_start=ys['start'][0], y_end=ys['end'][0], yx_end=round(ys['end'][1]), yy_end=round(ys['end'][2]), hw_start=ys['start'][3], hw_end=ys['end'][3]))
            cell = Image.new("RGB", (480, 145))
            for k, name in enumerate(['start', 'release', 'end']):
                cell.paste(Image.fromarray(frames[name]).resize((160, 120)), (k * 160, 22))
            d = ImageDraw.Draw(cell)
            d.text((3, 3), f"{repo[-1]}:{ep} {r['stage']} {len(a)/ds.fps:.0f}s cL{rows[-1]['closesL']} cR{rows[-1]['closesR']} y{ys['end'][0]}",
                   font=FONT, fill=(255, 255, 0))
            thumbs.append(cell)
            print(rows[-1], flush=True)
        flags = flag_outliers(metrics)
        for r in rows[-len(bounds):]:
            r['iqr_flags'] = ' '.join(flags.get(r['ep'], []))
        for p in range(0, len(thumbs), 25):
            sheet = Image.new("RGB", (480 * 5, 145 * 5))
            for k, c in enumerate(thumbs[p:p + 25]):
                sheet.paste(c, ((k % 5) * 480, (k // 5) * 145))
            sheet.save(OUT / f"{repo}_{p:02d}.jpg", quality=88)
    pd.DataFrame(rows).to_csv(OUT / "episodes.csv", index=False)
