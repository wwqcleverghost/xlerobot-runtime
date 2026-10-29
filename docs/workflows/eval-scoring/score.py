"""Auto stage scoring for handover evals from proprioception only (no vision).
hold = gripper commanded closed but stalled open by the tape (state - action > GAP)."""
import sys, numpy as np, pandas as pd
FPS, GAP, OPEN, MIN = 30, 2.0, 10.0, 15   # MIN frames = 0.5 s
ARM = {'L': slice(0, 5), 'R': slice(6, 11)}; GR = {'L': 5, 'R': 11}

def runs(mask):
    """[(start, end)] of True runs with length >= MIN."""
    out, s = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and s is None: s = i
        if not m and s is not None:
            if i - s >= MIN: out.append((s, i))
            s = None
    return out

def locate(run):
    """(data dir, success.csv path or None). Some eval uploads nest the dataset in dataset/; others (v6) keep data at
    the top and put only success.csv in an otherwise empty dataset/."""
    import os
    base = f'{run}/dataset' if os.path.isdir(f'{run}/dataset/data') else run
    csv = next((c for c in (f'{base}/success.csv', f'{run}/success.csv', f'{run}/dataset/success.csv') if os.path.exists(c)), None)
    return base, csv

def holds(st, ac):
    """{'L'/'R': [(start, end)]} hold runs per arm."""
    return {k: runs((ac[:, g] < st[:, g] - GAP) & (st[:, g] > 4)) for k, g in GR.items()}

def stage_frames(st, ac):
    """Per-frame stage label: 0 reaching, 1 A holds, 2 both hold, 3 B holds, 4 placed (B opened after its hold).
    Boundaries come from the same hold pairs score() uses; first pair that completes the place wins, else first
    handover pair (stays 3 to the end). No hold pair: 0 before the first hold, 1 after."""
    hr, lab = holds(st, ac), np.zeros(len(st), np.int64)
    first = min((s for v in hr.values() for s, _ in v), default=None)
    if first is None: return lab
    lab[first:] = 1
    pairs = [(bs, a_end, be, B) for A, B in (('L', 'R'), ('R', 'L')) for _, a_end in hr[A]
             for bs, be in hr[B] if be > a_end and bs < a_end + 5]
    done = [p for p in pairs if p[2] - p[1] >= MIN and p[2] < len(st) and st[p[2]:, GR[p[3]]].max() > OPEN]
    if not (done or pairs): return lab
    bs, a_end, be, _ = min(done or pairs, key=lambda p: p[1])
    lab[min(bs, a_end):a_end] = 2; lab[a_end:] = 3
    if done: lab[be:] = 4
    return lab

def score(st, ac):
    hr = holds(st, ac)
    moved = {k: np.abs(st[:, a] - st[0, a]).max() > 30 for k, a in ARM.items()}
    opened = {k: (st[:, g] > OPEN).any() for k, g in GR.items()}
    stg = [0] * 5; tdone = None
    stg[0] = int(any(moved[k] and opened[k] for k in 'LR'))
    # B tried to take the tape: its open gripper got commanded closed while A was holding. Counts as reaching
    # the handover with the tape still held (no drop) even when B never holds it for MIN frames.
    closing = {k: (ac[:, g] < OPEN) & np.r_[False, st[:-1, g] > OPEN] for k, g in GR.items()}
    def attempt(A, B):
        return moved[B] and opened[B] and any(closing[B][s:e + 5].any() for s, e in hr[A])
    if stg[0] and any(hr.values()):
        stg[1] = 1   # any arm held the tape; regrasps allowed, so search every hold pair for a handover
        stg[2] = int(attempt('L', 'R') or attempt('R', 'L'))
        for A, B in (('L', 'R'), ('R', 'L')):
            for _, a_end in hr[A]:
                for _, be in hr[B]:
                    if not (be > a_end and any(s < a_end + 5 for s, e in hr[B] if e == be)): continue
                    stg[2] = 1
                    if be - a_end >= MIN and st[a_end:be, GR[A]].max() > OPEN:
                        stg[3] = 1
                        if be < len(st) and st[be:, GR[B]].max() > OPEN: stg[4] = 1; tdone = be  # still holding at the end = not placed
    # quality
    sp = np.abs(np.diff(st[:, list(range(5)) + list(range(6, 11))], axis=0)).max(1) * FPS  # deg/s
    move = sp > 5
    start = int(np.argmax(move)) if move.any() else len(st)
    end = tdone if tdone else len(st) - 1
    seg = ~move[start:end]
    still = np.convolve(seg, np.ones(MIN), 'valid') >= MIN if len(seg) >= MIN else np.zeros(0)   # still for 0.5 s window
    J = [0,1,2,3,4,6,7,8,9,10]
    def wobble(x):   # RMS deviation from a 0.3 s moving average, on moving frames only (deg)
        k = 9; xp = np.pad(x[:, J], ((k//2, k//2), (0, 0)), mode='edge')
        sm = np.stack([np.convolve(xp[:, j], np.ones(k)/k, 'valid') for j in range(len(J))], 1)
        r = np.abs(x[:, J] - sm).max(1)[1:]
        return float(np.sqrt((r[move]**2).mean())) if move.any() else 0.0
    jumps = int((np.abs(np.diff(ac[:, J], axis=0)).max(1) > 5).sum())   # >5 deg in one frame (150 deg/s)
    return dict(stage=stg, total=sum(stg), start=start / FPS,
                done=round(tdone / FPS, 1) if tdone else None,
                idle=round(still.sum() / FPS, 1), sj=round(wobble(st), 2), aj=round(wobble(ac), 2), jumps=jumps, dur=len(st)/FPS,
                grasps=sum(len(v) for v in hr.values()))

if __name__ == '__main__':
    for run in sys.argv[1:]:
        run = run.rstrip('/'); base, csv = locate(run)
        d = pd.concat(pd.read_parquet(f) for f in sorted(__import__('glob').glob(f'{base}/data/*/*.parquet')))
        lab = pd.read_csv(csv).success.values
        rows = []
        for ep, g in d.groupby('episode_index'):
            r = score(np.stack(g['observation.state'].values), np.stack(g['action'].values))
            r.update(ep=ep, label=lab[ep]); rows.append(r)
        df = pd.DataFrame(rows)
        st = np.array(df.stage.tolist())
        reach = st.cumprod(1).sum(0)  # funnel
        cond = [f'{reach[i]}/{reach[i-1] if i else len(df)}' for i in range(5)]
        print(f'== {run.split("/")[-1][:-16]}  人工成功 {lab.sum()}/{len(lab)}  自动阶段分 {df.total.sum()}/{5*len(df)}')
        print('   漏斗(条件通过) 前往/抓取/不掉/交接/放置:', ' '.join(cond))
        ok = df[df.done.notna()]
        print(f'   SUM|{lab.sum()}|{df.total.sum()}|{" ".join(cond)}|{ok.done.mean():.1f}|{df.idle.mean():.1f}|{df.sj.mean():.2f}|{df.aj.mean():.2f}|{df.jumps.sum()/df.dur.sum()*60:.1f}')
        for r in rows:
            print(f'   ep{r["ep"]} 人工{r["label"]} 阶段{"".join(map(str,r["stage"]))} 完成{r["done"]} 等待{r["idle"]}s 抖{r["sj"]}/{r["aj"]} 跳{r["jumps"]} 抓{r["grasps"]}次')
