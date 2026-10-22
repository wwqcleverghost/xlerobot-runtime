"""2x5 grid video of the 10 episodes of one eval dataset: head camera on top, left/right wrist below, a banner
"T1 OK  arm L  tape x .39" (green OK / red FAIL from success.csv), the timer bottom right; a finished episode
freezes on its last frame and says "end".

usage: python make_grid.py <eval_dataset_dir> <out.mp4>
"""
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

CAMS = ["head(RGDB)", "left_wrist", "right_wrist"]
HW, HH = 384, 288  # head tile; the two wrists share the row below at half size
WW, WH = HW // 2, HH // 2
BAR = 34
TW, TH = HW, BAR + HH + WH
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
GREEN, RED = (60, 220, 90), (235, 70, 60)
FPS = 30


def episode_frames(root, cam, ep):
    """The episode's frames of one camera, resized, then its last frame forever."""
    size = (HW, HH) if cam == CAMS[0] else (WW, WH)
    v = f"videos/observation.images.{cam}"
    t0 = ep[f"{v}/from_timestamp"]
    c = av.open(str(root / v / f"chunk-{int(ep[f'{v}/chunk_index']):03d}/file-{int(ep[f'{v}/file_index']):03d}.mp4"))
    s = c.streams.video[0]
    c.seek(int(t0 / s.time_base), stream=s)
    last, n = None, 0
    for f in c.decode(s):
        if f.time < t0 - 0.5 / FPS:
            continue
        if n == ep["length"]:
            break
        last = f.to_image().resize(size)
        n += 1
        yield last
    c.close()
    while True:
        yield last


def first_arm(a):
    """Arm whose joints (gripper excluded) first move > 15 deg from the start pose."""
    for t in range(len(a)):
        moved = [np.abs(a[t, s:s + 5] - a[0, s:s + 5]).max() > 15 for s in (0, 6)]
        if any(moved):
            return "L" if moved[0] else "R"
    return "-"


def tape_x(img):
    """Normalised x of the yellow tape in a head frame (nan when not visible)."""
    hsv = np.asarray(img.convert("HSV")).astype(int)
    m = (hsv[..., 0] > 25) & (hsv[..., 0] < 50) & (hsv[..., 1] > 90) & (hsv[..., 2] > 90)
    return np.nonzero(m)[1].mean() / m.shape[1] if m.sum() > 50 else float("nan")


def main(root, out):
    eps = pd.concat(map(pd.read_parquet, sorted((root / "meta/episodes").rglob("*.parquet")))).sort_values("episode_index")
    data = pd.concat(map(pd.read_parquet, sorted((root / "data").rglob("*.parquet"))))
    succ = pd.read_csv(root / "success.csv").set_index("episode")["success"]
    streams, labels = [], []
    for i, (_, ep) in enumerate(eps.iterrows()):
        gens = [episode_frames(root, c, ep) for c in CAMS]
        first = [next(g) for g in gens]
        streams.append((gens, first))
        a = np.stack(data[data.episode_index == ep.episode_index]["action"].to_numpy())
        ok = bool(succ.get(int(ep.episode_index), 0))
        x = tape_x(first[0])
        xs = "?" if np.isnan(x) else f"{x:.2f}".lstrip("0")
        labels.append((f"T{i + 1} {'OK' if ok else 'FAIL'}  arm {first_arm(a)}  tape x {xs}", GREEN if ok else RED))
    oc = av.open(str(out), "w")
    vs = oc.add_stream("libx264", rate=FPS)
    vs.width, vs.height, vs.pix_fmt = 5 * TW, 2 * TH, "yuv420p"
    vs.options = {"crf": "26", "preset": "veryfast"}
    for t in range(int(eps["length"].max())):
        canvas = Image.new("RGB", (5 * TW, 2 * TH), (20, 20, 20))
        d = ImageDraw.Draw(canvas)
        for i, ((gens, first), (text, color)) in enumerate(zip(streams, labels)):
            head, lw, rw = first if t == 0 else [next(g) for g in gens]
            x, y = (i % 5) * TW, (i // 5) * TH
            canvas.paste(head, (x, y + BAR))
            canvas.paste(lw, (x, y + BAR + HH))
            canvas.paste(rw, (x + WW, y + BAR + HH))
            d.rectangle([x, y, x + TW - 1, y + BAR - 1], fill=(35, 35, 35))
            d.text((x + 8, y + 5), text, font=FONT, fill=color)
            if t >= eps.iloc[i]["length"]:
                d.text((x + HW - 60, y + BAR + 6), "end", font=FONT, fill=(255, 255, 255))
        d.text((5 * TW - 120, 2 * TH - 32), f"t = {t // FPS} s", font=FONT, fill=(255, 255, 255))
        for p in vs.encode(av.VideoFrame.from_image(canvas)):
            oc.mux(p)
    for p in vs.encode():
        oc.mux(p)
    oc.close()
    print(out, [label for label, _ in labels])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
