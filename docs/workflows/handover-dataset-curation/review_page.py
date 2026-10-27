"""One static page to review where every kept episode of a curation spec is cut: per episode a card with head-camera
stills and the raw clip (autoplays while on screen, playhead on the timeline), a timeline of the raw recording that is
also an editor (drag the start/end handles, add mid cuts, select one and press Delete to remove it, drop the episode; starts from the current trim +
the snap rule), and OK / problem buttons. "Export" downloads every episode's frame ranges and review as JSON.
Snap = a follower joint jumps > SNAP units in one frame while its command barely moves (servo stuck, then springs).
Proposed with the snap rule: snap in the first 30% of the kept range -> start after it; last 30% -> end before it; else drop.

usage: PYTHONPATH=src:examples/xlerobot:docs/workflows/eval-grid-video python review_page.py <spec.json> <out_dir> [--reset_tol 10] [--to_target] [--snap_rule]
       python serve.py <out_dir> 8766   (Range support: python -m http.server cannot seek videos)   then open <out_dir>/index.html
"""
import html
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from make_grid import episode_frames
from trim_stagnant_frames import keep_range, reach_start

CAMS = ["head(RGDB)", "left_wrist", "right_wrist"]


def clip(root, ep_row, out):
    """Raw episode as 320x360: head on top, left/right wrist below; faststart so the browser can seek at once."""
    import av
    gens = [episode_frames(root, c, ep_row) for c in CAMS]
    oc = av.open(str(out), "w")
    try:  # GPU encoder when there is one (frees the CPU for dataset builds), else libx264
        vs = oc.add_stream("h264_nvenc", rate=30)
        vs.options = {"cq": "28", "preset": "p4", "g": "15"}
    except Exception:
        vs = oc.add_stream("libx264", rate=30)
        vs.options = {"crf": "26", "preset": "veryfast", "g": "15"}
    vs.width, vs.height, vs.pix_fmt = 320, 360, "yuv420p"
    for _ in range(int(ep_row["length"])):
        head, lw, rw = (next(g) for g in gens)
        c = Image.new("RGB", (320, 360))
        c.paste(head.resize((320, 240)), (0, 0)); c.paste(lw.resize((160, 120)), (0, 240)); c.paste(rw.resize((160, 120)), (160, 240))
        for pk in vs.encode(av.VideoFrame.from_image(c)):
            oc.mux(pk)
    for pk in vs.encode():
        oc.mux(pk)
    oc.close()

SNAP, CMD_STILL = 20.0, 5.0
J = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
NAMES = ["左肩转", "左肩抬", "左肘", "左腕屈", "左腕转", "左夹爪", "右肩转", "右肩抬", "右肘", "右腕屈", "右腕转", "右夹爪"]
HEAD = "observation.images.head(RGDB)"


def snaps(a, s, first, end):
    js, jc = np.abs(np.diff(s[:, J], axis=0)), np.abs(np.diff(a[:, J], axis=0))
    hit = ((js > SNAP) & (jc < CMD_STILL)).any(1)
    return [(int(t + 1), J[int(js[t].argmax())], round(float(js[t].max()))) for t in np.flatnonzero(hit) if first <= t + 1 < end]


def propose(first, end, sn):
    f, e, drop = first, end, False
    for t, _, _ in sn:
        pos = (t - first) / max(end - first, 1)
        if pos < 0.3:
            f = max(f, t + 2)
        elif pos > 0.7:
            e = min(e, t - 1)
        else:
            drop = True
    return f, e, drop


def pauses(a, first, end, min_s=1.0):
    """Mid-episode stretches >= min_s where the arm command stands still (< 3 units/s, as the quality check)."""
    still = np.r_[np.linalg.norm(np.diff(a, axis=0), axis=1) * 30 < 3.0, False]
    out, s0 = [], None
    for t in range(first, end):
        if still[t] and s0 is None:
            s0 = t
        if not still[t] and s0 is not None:
            if t - s0 >= min_s * 30 and s0 > first:
                out.append((s0, t))
            s0 = None
    return out


def main(spec_path, out, reset_tol=3.0, to_target=False, snap_rule=False, title="裁剪审阅"):
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "vid").mkdir(exist_ok=True)
    cards, n = [], 0
    for src in json.load(open(spec_path)):
        root = Path(src["root"]) if src.get("root") else None
        ds = LeRobotDataset(src["repo"], root=root)
        t = ds.hf_dataset.with_format("numpy")
        root = ds.root
        meta_eps = pd.concat(map(pd.read_parquet, sorted((root / "meta/episodes").rglob("*.parquet")))).set_index("episode_index")
        A, S = np.asarray(t["action"])[:, :12].astype(float), np.asarray(t["observation.state"])[:, :12].astype(float)
        for ep in range(ds.num_episodes):
            if ep in src.get("drop", []):
                continue
            lo, hi = int(ds.meta.episodes["dataset_from_index"][ep]), int(ds.meta.episodes["dataset_to_index"][ep])
            a, s = A[lo:hi], S[lo:hi]
            f, e = keep_range(a, s, list(range(12)), 0.2, 2.0, reset_tol)
            if to_target:
                f = reach_start(a, s, f, e)
            sn = snaps(a, s, f, e)
            pf, pe, drop = propose(f, e, sn) if snap_rule else (f, e, False)
            ps = pauses(a, f, e)
            shots = [(max(f - 15, 0), "剪掉部分（起点前 0.5 秒）"), (f, "保留起点"), (e - 1, "保留终点"),
                     (min(e + 14, hi - lo - 1), "剪掉部分（终点后 0.5 秒）")]
            for k, (tt, _, _) in enumerate(sn):
                shots += [(tt - 2, f"突变{k + 1}之前"), (tt + 1, f"突变{k + 1}之后")]
            imgs = []
            for fr, label in shots:
                p = f"img/{n:02d}_{fr}.jpg"
                if not (out / p).exists():
                    Image.fromarray((ds[lo + fr][HEAD].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((224, 168)).save(out / p, quality=80)
                imgs.append((p, f"{label} · {fr / 30:.1f}s"))
            vid = f"vid/{n:02d}.mp4"
            if not (out / vid).exists():
                clip(root, meta_eps.loc[ep], out / vid)
            cards.append(dict(vid=vid, i=n, src=src["repo"].split("/")[-1], ep=ep, n=hi - lo, first=f, end=e, pfirst=pf, pend=pe,
                              drop=drop, cut=src.get("cut", {}).get(str(ep), []), snaps=sn, pauses=ps, imgs=imgs))
            print(n, cards[-1]["src"], ep, sn, "DROP" if drop else "", flush=True)
            n += 1
    (out / "index.html").write_text(page(cards, title, out.name))


def page(cards, title="裁剪审阅", key="review"):
    data = [{k: c[k] for k in ("i", "src", "ep", "n", "first", "end", "pfirst", "pend", "drop", "cut", "snaps", "pauses")} for c in cards]

    def card(c):
        sn = "；".join(f"{t / 30:.1f}秒 {NAMES[j]} 跳 {d}" for t, j, d in c["snaps"]) or "无"
        pa = "；".join(f"{a / 30:.1f}–{b / 30:.1f}秒" for a, b in c["pauses"]) or "无"
        im = "".join(f'<figure><img src="{p}" loading="lazy" width="200" height="150"><figcaption>{html.escape(t)}</figcaption></figure>'
                     for p, t in c["imgs"])
        return (f'<section class="card" id="c{c["i"]}" data-i="{c["i"]}"><header><b>#{c["i"]}</b> {c["src"]} ep{c["ep"]}'
                f'<span>原始 {c["n"] / 30:.1f} 秒</span><span class="sel"></span></header>'
                f'<div class="bar"><div class="ph"></div></div>'
                f'<p class="meta">突变（红线）：{sn}。停顿 > 1 秒（黄色）：{pa}。细框 = 规则算出的保留范围</p>'
                f'<div class="row"><video src="{c["vid"]}" poster="{c["imgs"][1][0]}" preload="none" muted loop playsinline></video>'
                f'<div class="strip">{im}</div></div>'
                f'<footer><button data-v="ok">通过</button><button data-v="bad">有问题</button><button class="drop">整条不要</button>'
                f'<button class="addcut">加剪切段</button><button class="reset">恢复初始</button><input placeholder="备注" data-note></footer></section>')

    return """<!doctype html><meta charset="utf-8"><title>"""+title+"""</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f5f1;--card:#fff;--ink:#1b1b1a;--mute:#6b6a65;--line:#e1e0d9;--kept:#1baf7a;--cut:#b4b2a9;--snap:#e34948;--h:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#151514;--card:#1f1f1d;--ink:#eeede8;--mute:#9a998f;--line:#33332f;--cut:#55544f}}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.5 system-ui,sans-serif}
.top{position:sticky;top:0;z-index:5;background:var(--bg);padding:8px 16px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.legend span{display:inline-flex;align-items:center;gap:4px;margin-right:10px;color:var(--mute)} .legend i{display:inline-block;width:14px;height:8px;border-radius:2px}
main{padding:12px 16px;display:grid;gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;min-width:0}
.card.ok{border-color:var(--kept)} .card.bad{border-color:var(--snap);box-shadow:0 0 0 1px var(--snap)} .card.dropped{opacity:.45}
header{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap} header span{color:var(--mute)} .sel{color:var(--ink)!important}
.bar{position:relative;height:22px;background:var(--cut);border-radius:4px;margin:10px 8px 4px;cursor:pointer;touch-action:none}
.seg{position:absolute;top:0;bottom:0;background:var(--kept)} .cutseg{position:absolute;top:0;bottom:0;background:var(--cut);outline:1px dashed var(--ink);cursor:pointer}
.cutseg.selected{outline:3px solid var(--h);z-index:1}
.orig{position:absolute;top:-4px;bottom:-4px;border:1px solid var(--ink);border-radius:3px;pointer-events:none;opacity:.5}
.pause{position:absolute;top:2px;bottom:2px;background:#eda100;opacity:.85;pointer-events:none}
.snap{position:absolute;top:-6px;bottom:-6px;width:3px;margin-left:-1px;background:var(--snap);pointer-events:none}
.hd{position:absolute;top:-6px;bottom:-6px;width:12px;margin-left:-6px;background:var(--h);border-radius:3px;cursor:ew-resize;z-index:2}
.hd.c{width:6px;margin-left:-3px;background:#eb6834;top:-9px;bottom:-9px}
.ph{position:absolute;top:-8px;bottom:-8px;width:2px;margin-left:-1px;background:var(--ink);left:0;pointer-events:none;z-index:3}
.meta{margin:2px 0 6px;color:var(--mute)}
.row{display:flex;gap:10px;align-items:flex-start} video{flex:none;width:320px;height:360px;background:var(--cut);border-radius:4px;cursor:pointer}
.strip{display:flex;flex-wrap:wrap;gap:6px;min-width:0} figure{margin:0} img{display:block;border-radius:4px;background:var(--cut)}
figcaption{font-size:11px;color:var(--mute);width:200px}
footer{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap} footer input{flex:1;min-width:120px} button{cursor:pointer}
@media (max-width:700px){.row{flex-direction:column} video{width:100%;height:auto}}
</style>
<div class="top"><b>"""+title+"""</b><span id="count"></span>
<span class="legend"><span><i style="background:var(--cut)"></i>剪掉</span><span><i style="background:var(--kept)"></i>保留</span>
<span><i style="background:var(--h)"></i>拖动把手改剪切点</span><span><i style="background:var(--snap)"></i>突变</span><span><i style="background:#eda100"></i>停顿 > 1 秒</span>
<span><i style="border:1px solid var(--ink)"></i>原来保留的范围</span><span>点剪切段或它的橙色把手选中，按 Delete 删除</span></span>
<label>播放速度 <select id="rate"><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option></select></label>
<label><input type="checkbox" id="only"> 只看有突变的</label><button id="exp">导出审阅结果</button></div>
<main>""" + "".join(card(c) for c in cards) + """</main>
<script>
const D=""" + json.dumps(data) + """;
const KEY='"""+key+"""', FPS=30;
let st={}; try{st=JSON.parse(localStorage.getItem(KEY))||{}}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){};count()};
const init=d=>({a:d.pfirst,b:d.pend,cuts:d.cut.map(x=>[...x]),drop:d.drop});
const E=i=>(st[i]=st[i]||{},st[i].edit=st[i].edit||init(D[i]),st[i].edit);
const count=()=>{const v=Object.values(st);document.getElementById('count').textContent=
  `${v.filter(x=>x.v==='ok').length} 通过 · ${v.filter(x=>x.v==='bad').length} 有问题 · ${D.filter(d=>E(d.i).drop).length} 整条不要 · 共 ${D.length} 条`};
const pct=(d,f)=>(100*f/d.n)+'%';
function draw(c){const i=+c.dataset.i,d=D[i],e=E(i),bar=c.querySelector('.bar'),ph=bar.querySelector('.ph');
  bar.querySelectorAll(':not(.ph)').forEach(x=>x.remove());
  const add=(cls,l,w,extra={})=>{const x=document.createElement('div');x.className=cls;x.style.left=pct(d,l);if(w!=null)x.style.width=pct(d,w);Object.assign(x.dataset,extra);bar.insertBefore(x,ph);return x};
  add('seg',e.a,e.b-e.a); e.cuts.forEach(([s,t],k)=>add('cutseg',s,t-s,{k}));
  add('orig',d.first,d.end-d.first); d.pauses.forEach(([s,t])=>add('pause',s,t-s)); d.snaps.forEach(([t])=>add('snap',t));
  add('hd',e.a,null,{h:'a'}); add('hd',e.b,null,{h:'b'});
  e.cuts.forEach(([s,t],k)=>{add('hd c',s,null,{h:'s',k});add('hd c',t,null,{h:'t',k})});
  if(sel&&sel.i===i)bar.querySelectorAll(`.cutseg[data-k="${sel.k}"]`).forEach(x=>x.classList.add('selected'));
  const sec=f=>(f/FPS).toFixed(2);
  c.querySelector('.sel').textContent=e.drop?'整条不要':`保留 ${sec(e.a)}–${sec(e.b)} 秒`+(e.cuts.length?`，中间剪掉 `+e.cuts.map(([s,t])=>`${sec(s)}–${sec(t)}`).join('、'):'');
  c.classList.toggle('dropped',e.drop)}
let sel=null; const vids=[];
document.addEventListener('keydown',ev=>{if(!sel||!['Delete','Backspace'].includes(ev.key)||ev.target.tagName==='INPUT')return;
  ev.preventDefault();E(sel.i).cuts.splice(sel.k,1);draw(sel.c);save();sel=null});
document.querySelectorAll('.card').forEach(c=>{const i=+c.dataset.i,d=D[i],v=c.querySelector('video'),bar=c.querySelector('.bar'),inp=c.querySelector('[data-note]');vids.push(v);
  const s=st[i]||{}; if(s.v)c.classList.add(s.v); inp.value=s.note||''; draw(c);
  const frameAt=x=>{const r=bar.getBoundingClientRect();return Math.max(0,Math.min(d.n,Math.round((x-r.left)/r.width*d.n)))};
  const seek=f=>{const go=()=>{v.currentTime=Math.min(f/FPS,v.duration-0.01)};v.readyState>0?go():(v.preload='auto',v.addEventListener('loadedmetadata',go,{once:true}),v.load())};
  let drag=null;
  bar.addEventListener('pointerdown',ev=>{const h=ev.target.dataset.h;
    if(h){drag={h,k:+ev.target.dataset.k};v.pause();bar.setPointerCapture(ev.pointerId);ev.preventDefault();
      if(h==='s'||h==='t'){sel={c,i,k:drag.k};document.querySelectorAll('.cutseg.selected').forEach(x=>x.classList.remove('selected'));draw(c)}
      return}
    if(ev.target.classList.contains('cutseg')){sel={c,i,k:+ev.target.dataset.k};document.querySelectorAll('.cutseg.selected').forEach(x=>x.classList.remove('selected'));ev.target.classList.add('selected');return}
    drag={h:'scrub'};v.pause();bar.setPointerCapture(ev.pointerId);ev.preventDefault();seek(frameAt(ev.clientX))});
  bar.addEventListener('pointermove',ev=>{if(!drag)return;const e=E(i),f=frameAt(ev.clientX);
    if(drag.h==='scrub'){seek(f);return}
    if(drag.h==='a')e.a=Math.min(f,e.b-15); else if(drag.h==='b')e.b=Math.max(f,e.a+15);
    else{const cu=e.cuts[drag.k];if(drag.h==='s')cu[0]=Math.min(f,cu[1]-1);else cu[1]=Math.max(f,cu[0]+1)}
    seek(f);draw(c)});
  bar.addEventListener('pointerup',()=>{if(!drag)return;const scrub=drag.h==='scrub';drag=null;if(scrub)v.play().catch(()=>{});else save()});

  c.querySelector('.addcut').onclick=()=>{const e=E(i),m=Math.round((e.a+e.b)/2);e.cuts.push([m-30,m+30]);sel={c,i,k:e.cuts.length-1};draw(c);save()};
  c.querySelector('.drop').onclick=()=>{const e=E(i);e.drop=!e.drop;draw(c);save()};
  c.querySelector('.reset').onclick=()=>{st[i]={...st[i],edit:init(d)};draw(c);save()};
  c.querySelectorAll('[data-v]').forEach(b=>b.onclick=()=>{st[i]={...st[i],v:b.dataset.v};c.classList.remove('ok','bad');c.classList.add(b.dataset.v);save()});
  inp.oninput=()=>{st[i]={...st[i],note:inp.value};save()};
  v.onclick=()=>v.paused?v.play().catch(()=>{}):v.pause()});
const io=new IntersectionObserver(es=>es.forEach(e=>{const v=e.target;if(e.isIntersecting){v.preload='auto';v.play().catch(()=>{})}else v.pause()}),{threshold:0.3});
vids.forEach(v=>io.observe(v));
let RATE=1; try{RATE=+localStorage.getItem('review-rate')||1}catch(e){}
const rateSel=document.getElementById('rate'); rateSel.value=String(RATE);
vids.forEach(v=>{v.defaultPlaybackRate=RATE;v.playbackRate=RATE;v.addEventListener('play',()=>{v.playbackRate=RATE})});
rateSel.onchange=()=>{RATE=+rateSel.value;try{localStorage.setItem('review-rate',RATE)}catch(e){};vids.forEach(v=>{v.defaultPlaybackRate=RATE;v.playbackRate=RATE})};
const tick=()=>{vids.forEach((v,k)=>{if(v.duration)v.closest('.card').querySelector('.ph').style.left=(100*v.currentTime/v.duration)+'%'});requestAnimationFrame(tick)};requestAnimationFrame(tick);
document.getElementById('only').onchange=ev=>document.querySelectorAll('.card').forEach(c=>c.style.display=ev.target.checked&&!D[+c.dataset.i].snaps.length?'none':'');
document.getElementById('exp').onclick=()=>{const out=D.map(d=>({i:d.i,src:d.src,ep:d.ep,...E(d.i),review:(st[d.i]||{}).v||'',note:(st[d.i]||{}).note||''}));
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));a.download='"""+key+""".json';a.click()};
count();
</script>"""


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("out", type=Path)
    ap.add_argument("--reset_tol", type=float, default=3.0, help="tail: keep until every joint stays within this of the final pose")
    ap.add_argument("--to_target", action="store_true", help="head: start where the arm heads for the tape (reach_start)")
    ap.add_argument("--snap_rule", action="store_true", help="initial cuts move start/end past snaps, drop mid snaps")
    ap.add_argument("--title", default="裁剪审阅")
    x = ap.parse_args()
    main(x.spec, x.out, x.reset_tol, x.to_target, x.snap_rule, x.title)
