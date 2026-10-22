"""Jitter = small back-and-forth reversals: turning points (deadband DB) whose swing is < BIG deg."""
import sys, glob, os, numpy as np, pandas as pd
J = [0,1,2,3,4,6,7,8,9,10]; FPS = 30; DB, BIG = 0.5, 5.0
def zigzag(x):
    piv = ext = x[0]; d = 0; swings = []
    for v in x[1:]:
        if d == 0:
            if abs(v - piv) > DB: d = 1 if v > piv else -1; ext = v
        elif d * (v - ext) > 0: ext = v
        elif d * (ext - v) > DB:
            swings.append(abs(ext - piv)); piv, ext, d = ext, v, -d
    return sum(1 for s in swings if s < BIG)
def ep_metrics(st):
    sp = np.abs(np.diff(st[:, J], axis=0)).max(1) * FPS
    mt = max((sp > 5).sum() / FPS, 1e-6)
    return sum(zigzag(st[:, j]) for j in J), mt
def run(path):
    base = path + '/dataset' if os.path.isdir(path + '/dataset') else path
    d = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f'{base}/data/*/*.parquet')))
    z = t = 0
    for _, g in d.groupby('episode_index'):
        a, b = ep_metrics(np.stack(g['observation.state'].values)); z += a; t += b
    return z / t
if __name__ == '__main__':
  for p in sys.argv[1:]:
    print(f'{os.path.basename(p.rstrip("/"))[:40]:42s} 小幅往返 {run(p):5.2f} 次/秒(运动中)')
