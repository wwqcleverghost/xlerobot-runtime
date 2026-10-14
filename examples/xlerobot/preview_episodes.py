"""Preview a LeRobot v3 dataset in the browser, 10 episodes per page (e.g. to review trim_stagnant_frames.py).

Cuts every episode into its own clip per camera (stream copy, the frames training sees) and writes a page
with a camera picker, playback speed and paging (arrow keys). Then serve it:

    python examples/xlerobot/preview_episodes.py ~/.cache/huggingface/lerobot/wwqcleverghost/handover \
        --original ~/.cache/huggingface/lerobot/wwqcleverghost/handover4 --out outputs/preview_handover
    python -m http.server 8765 --directory outputs/preview_handover
"""

import argparse
import glob
import json
from pathlib import Path

import av
import pandas as pd


def episodes_meta(root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files]).set_index("episode_index")


def cut_clip(inp, t0: float, t1: float, fps: int, path: Path):
    """Copy the packets of [t0, t1) from an open container into their own mp4."""
    ist = inp.streams.video[0]
    off = int(round(t0 / ist.time_base))
    inp.seek(off, stream=ist, backward=True)
    with av.open(str(path), "w", options={"movflags": "faststart"}) as out:
        ost = out.add_stream_from_template(ist)
        for pkt in inp.demux(ist):
            if pkt.dts is None or pkt.dts * ist.time_base >= t1 + 5 / fps:
                break
            # B-frames arrive out of order, so filter on pts and stop on dts.
            if not t0 - 0.5 / fps <= pkt.pts * ist.time_base < t1 - 0.5 / fps:
                continue
            pkt.pts -= off
            pkt.dts -= off
            pkt.stream = ost
            out.mux(pkt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="local dataset directory")
    parser.add_argument("--original", type=Path, help="dataset it was trimmed from, to show the original lengths")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    info = json.loads((args.root / "meta/info.json").read_text())
    fps = info["fps"]
    cams = [k for k, ft in info["features"].items() if ft["dtype"] == "video"]
    eps = episodes_meta(args.root)
    orig = episodes_meta(args.original)["length"] if args.original else eps["length"]

    for cam in cams:
        (args.out / "clips" / cam).mkdir(parents=True, exist_ok=True)
        for (chunk, file), group in eps.groupby([f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"]):
            video = args.root / info["video_path"].format(video_key=cam, chunk_index=chunk, file_index=file)
            with av.open(str(video)) as inp:
                for ep, row in group.iterrows():
                    t0, t1 = row[f"videos/{cam}/from_timestamp"], row[f"videos/{cam}/to_timestamp"]
                    cut_clip(inp, t0, t1, fps, args.out / "clips" / cam / f"ep{ep:03d}.mp4")
        print("clips done:", cam, flush=True)

    data = {
        "title": args.root.name,
        "fps": fps,
        "cams": cams,
        "episodes": [{"ep": int(ep), "len": int(n), "orig": int(orig[ep])} for ep, n in eps["length"].items()],
    }
    (args.out / "index.html").write_text(PAGE.replace("__DATA__", json.dumps(data)))
    print(f"wrote {args.out / 'index.html'}; serve with: python -m http.server 8765 --directory {args.out}")


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Episode Preview</title>
<style>
  :root { --bg:#f6f6f4; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#dcdcd6; --accent:#2f6fde; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161615; --card:#222220; --ink:#ececea; --muted:#9a9a94; --line:#3a3a37; --accent:#6c9cf0; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.4 system-ui, sans-serif; }
  header { position:sticky; top:0; z-index:2; display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center;
           padding:10px 16px; background:var(--bg); border-bottom:1px solid var(--line); }
  h1 { font-size:16px; margin:0 8px 0 0; }
  .group { display:flex; gap:4px; align-items:center; flex-wrap:wrap; }
  button, select { font:inherit; color:var(--ink); background:var(--card); border:1px solid var(--line);
                   border-radius:6px; padding:4px 10px; cursor:pointer; }
  button[aria-pressed="true"] { background:var(--accent); border-color:var(--accent); color:#fff; }
  .muted { color:var(--muted); }
  main { display:grid; grid-template-columns:repeat(auto-fill, minmax(300px, 1fr)); gap:12px; padding:12px 16px 32px; }
  .tile { background:var(--card); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  video { display:block; width:100%; aspect-ratio:4/3; background:#000; }
  .bar { height:6px; background:var(--line); }
  .bar div { height:100%; width:0; background:var(--accent); opacity:.6; }
  .meta { padding:6px 10px; display:flex; justify-content:space-between; gap:8px; }
  .meta b { font-variant-numeric:tabular-nums; }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="group" id="pages"></div>
  <div class="group">
    <label class="muted" for="cam">Camera</label><select id="cam"></select>
    <label class="muted" for="rate">Speed</label>
    <select id="rate"><option>1</option><option selected>2</option><option>4</option></select>
    <button id="sync">Replay all</button>
  </div>
  <span class="muted">&larr; / &rarr; to page</span>
</header>
<main id="grid"></main>
<script>
const DATA = __DATA__;
const PER = 10, pages = Math.ceil(DATA.episodes.length / PER);
let page = 0;
document.getElementById("title").textContent = DATA.title;
const camSel = document.getElementById("cam"), rateSel = document.getElementById("rate");
DATA.cams.forEach(c => camSel.add(new Option(c.replace("observation.images.", ""), c)));
const pageBox = document.getElementById("pages");
for (let p = 0; p < pages; p++) {
  const b = document.createElement("button");
  b.textContent = `${p * PER}-${Math.min((p + 1) * PER, DATA.episodes.length) - 1}`;
  b.onclick = () => show(p);
  pageBox.append(b);
}

function show(p) {
  page = (p + pages) % pages;
  [...pageBox.children].forEach((b, i) => b.setAttribute("aria-pressed", i === page));
  const grid = document.getElementById("grid");
  grid.querySelectorAll("video").forEach(v => { v.pause(); URL.revokeObjectURL(v.src); v.removeAttribute("src"); v.load(); });
  grid.replaceChildren();
  for (const e of DATA.episodes.slice(page * PER, (page + 1) * PER)) {
    const tile = document.createElement("div");
    tile.className = "tile";
    const cut = e.orig === e.len ? "" : ` (from <b>${e.orig}</b>)`;
    tile.innerHTML = `<video muted autoplay loop playsinline></video><div class="bar"><div></div></div>
      <div class="meta"><span>ep <b>${e.ep}</b></span><span class="muted"><b>${e.len}</b> frames${cut}</span></div>`;
    const v = tile.querySelector("video"), pos = tile.querySelector(".bar div");
    // Load each clip as a blob: http.server has no Range support, so a plain URL can't seek and every
    // loop would re-buffer (black flash).
    fetch(`clips/${encodeURIComponent(camSel.value)}/ep${String(e.ep).padStart(3, "0")}.mp4`)
      .then(r => r.blob()).then(b => { if (v.isConnected) v.src = URL.createObjectURL(b); });
    v.addEventListener("loadedmetadata", () => { v.playbackRate = +rateSel.value; });
    v.addEventListener("timeupdate", () => { pos.style.width = `${v.currentTime / (e.len / DATA.fps) * 100}%`; });
    grid.append(tile);
  }
}

camSel.onchange = () => show(page);
rateSel.onchange = () => document.querySelectorAll("video").forEach(v => v.playbackRate = +rateSel.value);
document.getElementById("sync").onclick = () => document.querySelectorAll("video").forEach(v => { v.currentTime = 0; v.play(); });
addEventListener("keydown", ev => {
  if (ev.key === "ArrowRight") show(page + 1);
  if (ev.key === "ArrowLeft") show(page - 1);
});
// Browsers pause muted autoplay in hidden tabs; resume when the page is shown again.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) document.querySelectorAll("video").forEach(v => v.play());
});
show(0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
