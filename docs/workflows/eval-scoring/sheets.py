"""Contact sheets for scoring eval episodes by eye: one cell per time step with the head camera on top and the
left/right wrist below, captioned "<t>s L <cmd>/<state> R <cmd>/<state>" for the two grippers.
Gripper reading: 1 = closed. Command ~1-5 with the state clearly higher (~7-20) = holding the tape; state falling
to the command while the command stays closed = the tape slipped out; command opening first = a deliberate release.

usage:
  python sheets.py <eval_dataset_dir> <out_dir>                    every 2 s, one sheet per episode (ep01.jpg ...)
  python sheets.py <eval_dataset_dir> <out_dir> <ep> <t0> <t1>     episode <ep> (1-based) from t0 to t1 s, every 1/3 s
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lerobot.datasets.lerobot_dataset import LeRobotDataset

HEAD, LW, RW = "observation.images.head(RGDB)", "observation.images.left_wrist", "observation.images.right_wrist"
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
W, H, COLS = 230, 173, 6


def pil(t, size):
    return Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize(size)


def sheet(ds, lo, frames, out):
    rows = -(-len(frames) // COLS)
    img = Image.new("RGB", (COLS * W, rows * (20 + H + H // 2)))
    d = ImageDraw.Draw(img)
    for k, i in enumerate(frames):
        x, y = (k % COLS) * W, (k // COLS) * (20 + H + H // 2)
        it = ds[i]
        a, s = it["action"].numpy(), it["observation.state"].numpy()
        img.paste(pil(it[HEAD], (W, H)), (x, y + 20))
        img.paste(pil(it[LW], (W // 2, H // 2)), (x, y + 20 + H))
        img.paste(pil(it[RW], (W // 2, H // 2)), (x + W // 2, y + 20 + H))
        d.text((x + 3, y + 2), f"{(i - lo) / ds.fps:.1f}s L {a[5]:.0f}/{s[5]:.0f} R {a[11]:.0f}/{s[11]:.0f}",
               font=FONT, fill=(255, 255, 0))
    img.save(out, quality=85)
    print(out)


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ds = LeRobotDataset(root.name, root=root)
    ep = ds.meta.episodes
    bounds = [(int(ep["dataset_from_index"][e]), int(ep["dataset_to_index"][e])) for e in range(ds.num_episodes)]
    if len(sys.argv) == 6:
        e, t0, t1 = int(sys.argv[3]) - 1, float(sys.argv[4]), float(sys.argv[5])
        lo, hi = bounds[e]
        frames = range(lo + int(t0 * ds.fps), min(hi, lo + int(t1 * ds.fps) + 1), ds.fps // 3)
        sheet(ds, lo, frames, out / f"ep{e + 1:02d}_{t0:g}-{t1:g}s.jpg")
        return
    for e, (lo, hi) in enumerate(bounds):
        sheet(ds, lo, list(range(lo, hi, 2 * ds.fps)) + [hi - 1], out / f"ep{e + 1:02d}.jpg")


if __name__ == "__main__":
    main()
