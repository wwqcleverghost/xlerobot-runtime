"""Per-frame task-stage labels for XLeRobot handover episodes (definition: docs/workflows/stage-labels/README.md).

A = the arm that grasps the tape first, B = the other arm. One causal tracker (StageTracker) detects every event; each event has
the frame it is confirmed at and its onset (the first frame of the evidence, found by looking back). Rules (--rule; the viewer
shows fine9 and eval5):
  fine9  stages switch when events are confirmed: frame t only uses frames <= t, so the robot computes the same live. Training
         inputs use this. Stages never go down except when the tape is dropped (back to 0); a miss or retry stays put.
  std9   the same events, switched at their onsets: the reference for checking and for training targets. Same order as fine9.
  eval5  score.stage_frames from eval scoring (5 stages, not causal), for comparison.
The tracker also uses the tape diameter in the wrist cameras (A对位) and forward kinematics (AB对位: B in A's view).

usage: PYTHONPATH=src python examples/xlerobot/stage_labels.py --dataset v7-handover-clean [--rule fine9] [--episodes 0 1] [--compare]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/workflows/eval-scoring"))
import score  # noqa: E402
import xlerobot_fk  # noqa: E402

# gripper: ~1 closed empty, ~30 open; holding the tape the command is ~2-5 but the gripper stalls at ~7-12
OPEN, GAP, EMPTY = score.OPEN, score.GAP, 4.0   # command > OPEN = open; stall = state - command > GAP; state <= EMPTY = closed on nothing
W = 5                # frames a hold / empty grip / near / in-view must last to be confirmed
GRIP_MAX = 20.0      # a gripper stalled on the tape reads <= GRIP_MAX (7-12 typically); a wide-open one that has not started
                     # moving yet also looks stalled and steady, but reads ~30
DROP = 2.0           # closing starts (onset) when the command falls more than DROP below the peak of this opening
CLOSING = 5.0        # ... and is confirmed once it is CLOSING below the peak
SIZE_TH = 0.18       # A对位: apparent tape diameter >= SIZE_TH (fraction of the frame width) with the gripper open
                     # (0.5 x median diameter when A starts closing on v8-handover2: 0.362; see README)
STEADY = 0.5        # 夹住: the W-frame average of the gripper reading changes by < STEADY over W frames
B_OPEN = 15.0        # B抓住: B's closing counts only after B opened to >= B_OPEN (real grasps 17.7+ on v7 / v8; v8 ep26 f234: 11)
STILL, FPS = 2.0, 30    # A抓住 starts once A's gripper is closing and its tip has nearly stopped along the gripper axis: < STILL cm/s
                     # over the last W frames (user: strict for A, axial only: v8 ep32 slides along the table). B is not held to it
HOME_DROP = 2.0      # 复位: B's joint distance to its start pose (first frame, recorded units) falls every frame for W frames,
                     # by at least HOME_DROP in all
VIEW = 35.0          # AB对位: B's tip within VIEW deg of A's gripper axis = B in A's wrist camera (xlerobot_fk.view_angle). By eye on
                     # v7 ep3 / ep20 / ep33 B appears at 25-32 deg; 30 missed v8 ep26 / ep38 (closest 33.5-34 deg), so 35
SIZE_VERSION = 2     # bump when tape_size changes (cached sizes are keyed by it)
GR = score.GR
CAM = {"L": "left_wrist", "R": "right_wrist"}
STAGES = ["A前往", "A对位", "A抓住", "A抓着", "AB对位", "B抓住", "B抓着", "B放下", "复位"]


def tape_size(rgb):
    """Apparent tape diameter: the larger 2nd-98th percentile extent of the yellow-green tape pixels (PIL HSV hue 28-80 of 255),
    as a fraction of the frame width; 0 when fewer than 30 pixels. The gripper fingers hide part of the tape but rarely its edges."""
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"))
    ys, xs = np.nonzero((hsv[..., 0] > 28) & (hsv[..., 0] < 80) & (hsv[..., 1] > 70) & (hsv[..., 2] > 60))
    if len(xs) < 30:
        return 0.0
    return float(max(np.ptp(np.percentile(xs, [2, 98])), np.ptp(np.percentile(ys, [2, 98]))) / rgb.shape[1])


def video_sizes(path, t0, n, fps, size=(160, 120)):
    """tape_size of the n frames of an episode that starts at t0 seconds in an mp4."""
    import av
    out = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(max(int((t0 - 1) / s.time_base), 0), stream=s)
        for fr in c.decode(s):
            if fr.time < t0 - 0.5 / fps:
                continue
            out.append(tape_size(fr.to_ndarray(width=size[0], height=size[1], format="rgb24")))
            if len(out) == n:
                break
    return np.array(out + out[-1:] * (n - len(out))) if out else np.zeros(n)


class StageTracker:
    """Feed frames one by one: 12-dim state / action (left arm 0-5, right arm 6-11, gripper last) and the tape diameter of each
    wrist camera {'L': .., 'R': ..}. step() returns the stage index into STAGES; self.events lists (confirmed at, onset, stage)."""

    def __init__(self):
        self.t, self.stage, self.A, self.B, self.cand = -1, 0, None, None, None
        self.S, self.C, self.D, self.events = [], [], [], []
        self.peak = {"L": 0.0, "R": 0.0}    # command peak of the current opening
        self.close_on = {"L": None, "R": None}   # onset of the current closing (None: not closing)
        self.hs = self.hc = 0.0             # grip state (tightest so far) / command of the arm holding alone
        self.since = 0                      # last return to 前往 / 对位: only closings that start after it count
        self.a_off = None                   # A let go while B was still closing (the hand-off goes on)

    # --- detectors: each returns the onset frame when confirmed now, else None ---
    def _last(self, n, ok):
        return self.t - n + 1 if self.t + 1 >= n and all(ok(i) for i in range(self.t - n + 1, self.t + 1)) else None

    def closing(self, k):
        on = self.close_on[k]
        return on if on is not None and on >= self.since and self.C[-1][GR[k]] < self.peak[k] - CLOSING else None

    def stopped(self, k, on):   # the first frame >= on from which the tip stays nearly still (< STILL cm/s), else None
        pts = lambda i: xlerobot_fk.arm_points(self.S[max(i, 0)], k)

        def still(i):       # no motion along the gripper axis (wrist -> tip, ~ the wrist z axis); sliding along the table is fine
            (w, p), (_, p0) = pts(i), pts(i - W)
            return abs((p - p0) @ ((p - w) / np.linalg.norm(p - w))) * 100 / W * FPS < STILL
        i = next((i for i in range(on, self.t + 1) if still(i)), None)
        return None if i is None else max(on, i - W)

    def grasp(self, k):     # A: closing AND the tip nearly still, from the later of the two (a close while still moving
        on = self.closing(k)    # down to align is not a grasp yet)
        return None if on is None else self.stopped(k, on)

    def clamp(self, k):     # B: closed onto the tape after its current closing started: held still at a tape width, and the
        on = self.close_on[k]   # reading came down from where it was (a half-open gripper also reads still, v8 ep26 f230)
        h = self.holding(k) if on is not None else None
        return h if h is not None and h >= on and self.S[self.t][GR[k]] < self.S[on][GR[k]] - GAP else None

    def b_grasp(self, k):   # B: a confirmed closing from a real opening (peak >= B_OPEN; not a close-and-reopen from half open,
        on = self.closing(k)    # v8 ep26 f234 peaked at 11). B is not required to be still: it moves in to meet A
        return on if on is not None and self.peak[k] >= B_OPEN else None

    def holding(self, k):
        """Closed onto the tape: the reading, averaged over W frames (small joint jitter averages out), sits at a tape width and
        changes by < STEADY over the last W frames, and the command is not opening it. Not "command well below the reading":
        once the tape is held the operator often eases the leader to about the reading (~10). Onset: where the reading settled."""
        g, t = GR[k], self.t
        if t + 1 < 2 * W:
            return None
        m = lambda i: float(np.mean([self.S[j][g] for j in range(i - W + 1, i + 1)]))
        ok = abs(m(t) - m(t - W)) < STEADY and all(EMPTY < m(i) <= GRIP_MAX and self.C[i][g] < m(i) + GAP
                                                   for i in range(t - W + 1, t + 1))
        return t - 2 * W + 1 if ok else None

    def empty(self, k):
        return self._last(W, lambda i: self.C[i][GR[k]] < OPEN and self.S[i][GR[k]] <= EMPTY)

    def let_go(self, k):    # commanded wider than the grip on the tape; loosening a little is not letting go
        th = max(OPEN, self.hs + 2 * GAP)
        if self.C[-1][GR[k]] <= th:
            return None
        i = self.t          # onset: first frame of the run above th - DROP (an earlier loosening below it is not part of it)
        while i > 0 and self.C[i - 1][GR[k]] > th - DROP:
            i -= 1
        return i

    def near(self, k):      # the gripper has finished opening (open and still) and the tape looks big enough
        on = self._last(W, lambda i: self.D[i][k] >= SIZE_TH and self.S[i][GR[k]] > OPEN)
        return on if on is not None and np.ptp([self.S[i][GR[k]] for i in range(on, self.t + 1)]) < 1 else None

    def seen(self, A, B):
        return self._last(W, lambda i: xlerobot_fk.view_angle(self.S[i], A, B) < VIEW)

    def homing(self, k, since):         # the arm heads back to its start pose: joint distance to it keeps falling
        sl_ = slice(0, 5) if k == "L" else slice(6, 11)
        d = lambda i: float(np.linalg.norm(self.S[i][sl_] - self.S[0][sl_]))
        if self.t - 1 <= since or d(self.t) >= d(self.t - 1):
            return None
        i = self.t          # onset: first frame of the falling run that ends now (every frame closer than the one before)
        while i - 1 > since and d(i - 1) < d(i - 2):
            i -= 1
        return i if self.t - i + 1 >= W and d(i - 1) - d(self.t) >= HOME_DROP else None

    def _go(self, g, on):
        self.events.append((self.t, min(on, self.t), g))
        if g in (0, 1) and self.stage not in (0, 1):
            self.since = self.t      # back after a miss / drop: an older closing is not a new grasp
        if g < 5:
            self.a_off = None
        if g == 7:
            self.t7 = self.t
        self.stage = g

    def step(self, s, a, size):
        self.t += 1
        s, a = np.asarray(s, float), np.asarray(a, float)
        self.S.append(s), self.C.append(a), self.D.append(dict(size))
        for k in "LR":      # track the peak of each opening and the onset of each closing
            c, prev = a[GR[k]], self.C[-2][GR[k]] if self.t else 0.0
            if c >= OPEN and prev < OPEN and self.close_on[k] is None:   # a new opening (hovering around OPEN while closing is not)
                self.peak[k] = c
            if self.close_on[k] is None:
                self.peak[k] = max(self.peak[k], c) if c >= OPEN else self.peak[k]
                if self.peak[k] >= OPEN and c < self.peak[k] - DROP:
                    self.close_on[k] = self.t
            elif c >= self.peak[k] - DROP:
                self.close_on[k] = None     # back up near the peak: not closing after all
        A, B, g = self.A, self.B, self.stage
        other = lambda k: "R" if k == "L" else "L"
        if g in (0, 1):         # 前往 / 对位: whichever arm grasps first is A
            k = next((k for k in "LR" if self.grasp(k) is not None), None)
            if k:
                self.A, self.B = k, other(k)
                self._go(2, self.grasp(k))
            elif g == 0:
                k = next((k for k in "LR" if self.near(k) is not None), None)
                if k:
                    self.cand = k
                    self._go(1, self.near(k))
        elif g == 2:            # A closing: holds the tape -> 3 (a miss or reopening stays in 抓住 until it holds)
            if self.holding(A) is not None:
                self.hs, self.hc = s[GR[A]], a[GR[A]]
                self._go(3, self.holding(A))
        elif g in (3, 4):       # A holds (3), B in A's view (4); B grasping counts from 3 too (B never seen)
            self.hs = min(self.hs, s[GR[A]])         # A can squeeze tighter after the hold is confirmed
            if self.b_grasp(B) is not None:
                self._go(5, self.b_grasp(B))
            elif self.let_go(A) is not None or self.empty(A) is not None:
                if self.close_on[B] is not None:     # B is closing: the hand-off goes on, not a drop
                    self._go(5, self.close_on[B])
                    self.a_off = self.let_go(A) if self.let_go(A) is not None else self.t
                else:
                    self._go(0, self.t)
            elif g == 3 and self.seen(A, B) is not None:
                self._go(4, self.seen(A, B))
        elif g == 5:            # B closing until it holds the tape (B抓着 then; A lets go during B抓着). A B miss / retry stays here
            if self.a_off is None:
                self.hs = min(self.hs, s[GR[A]])
                self.a_off = self.let_go(A)
            if self.clamp(B) is not None:
                self.hs, self.hc = s[GR[B]], a[GR[B]]
                self._go(6, self.clamp(B))
            elif self.a_off is not None and self.empty(B) is not None:
                self._go(0, self.empty(B))           # A let go and B is empty: dropped
        elif g == 6:            # B holds (A lets go meanwhile): B lets go to put the tape down, or drops it
            self.hs = min(self.hs, s[GR[B]])
            if self.let_go(B) is not None:
                self._go(7, self.let_go(B))
            elif self.empty(B) is not None:
                self._go(0, self.empty(B))
        elif g == 7:            # B let go; reset starts when B's arm heads back to its start pose (not when it opens, not a fixed delay)
            if self.homing(B, self.t7) is not None:
                self._go(8, self.homing(B, self.t7))
        if self.stage == 0:
            self.A = self.B = self.cand = None
        return self.stage


def _track(st, ac, size):
    """Run the tracker over a recorded episode. Frame t gets the command of frame t-1 (the state at frame 0): the action
    recorded at t is decided after observing t, so live the robot only knows the previous command (see LiveStage)."""
    t = StageTracker()
    st, prev = st[:, :12], np.r_[st[:1, :12], ac[:-1, :12]]
    lab = np.array([t.step(s, a, {"L": size["L"][i], "R": size["R"][i]}) for i, (s, a) in enumerate(zip(st, prev))], np.int64)
    return lab, t.events


class LiveStage:
    """Stage input for a policy trained with observation.environment_state (real-robot eval): call once per control step with
    the raw robot observation (joint '<arm>_<joint>.pos' values, wrist images 'left_wrist' / 'right_wrist' as HxWx3 RGB) and
    the command sent at the previous step (None at the start); returns the fine9 one-hot (9 float32), the same as the
    training labels (examples/xlerobot/add_stage_labels.py)."""
    JOINTS = [f"{arm}_arm_{j}.pos" for arm in ("left", "right")
              for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]

    def __init__(self):
        self.tracker = StageTracker()

    def __call__(self, obs, prev_cmd):
        s = np.array([float(obs[k]) for k in self.JOINTS])
        a = s if prev_cmd is None else np.array([float(prev_cmd[k]) for k in self.JOINTS])
        size = {k: tape_size(np.asarray(Image.fromarray(np.asarray(obs[cam])).resize((160, 120), Image.BILINEAR)))
                for k, cam in CAM.items()}
        return np.eye(len(STAGES), dtype=np.float32)[self.tracker.step(s, a, size)]


def fine9(st, ac, size):
    return _track(st, ac, size)[0]


def std9(st, ac, size):
    lab, prev = np.zeros(len(st), np.int64), 0
    for _, on, g in _track(st, ac, size)[1]:
        prev = max(prev, on)    # onsets never go before an earlier event: same order as fine9
        lab[prev:] = g
    return lab


RULES = {  # name -> (stage names, fn(state, action, size) -> per-frame stage index, description, needs wrist-camera sizes)
    "fine9": (STAGES, fine9, "9 阶段，只看过去的帧（机器人运行时也能算，训练输入用它）", True),
    "std9": (STAGES, std9, "9 阶段标准答案：同样的事件，切换在事件起点（核对、训练目标用它）", True),
    "eval5": (["前往", "A抓着", "两手抓", "B抓着", "已放下"], lambda st, ac, size: score.stage_frames(st[:, :12], ac[:, :12]),
              "评测打分的 5 阶段，看整段（事后标）", False),
}


def segments(lab):
    """[(start, end, stage)] runs of a per-frame label array."""
    cut = np.flatnonzero(np.diff(lab)) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, len(lab)]
    return [(int(a), int(b), int(lab[a])) for a, b in zip(starts, ends)]


def compare(ref, lab):
    """(fraction of frames that agree, {stage: frames lab enters it after ref (negative = earlier)}) for one episode."""
    first = lambda x, g: int(np.argmax(x == g)) if (x == g).any() else None
    lag = {g: first(lab, g) - first(ref, g) for g in range(len(STAGES)) if first(ref, g) is not None and first(lab, g) is not None}
    return float((ref == lab).mean()), lag


def selfcheck():
    # L opens at 5 and the tape looks big in its camera from 10; L closes at 15 and holds (stalled 20-100); R comes into L's view
    # at 60, opens at 50, closes at 80 (B抓住) and holds from 85 (B抓着; L lets go at 100 within it); R lets go at 140,
    # heads back at 170
    n = 200
    s, a = np.zeros((n, 12)), np.zeros((n, 12))
    s[:, [5, 11]] = a[:, [5, 11]] = 1
    a[5:15, 5] = s[5:15, 5] = 30
    a[15:100, 5], s[15:20, 5], s[20:100, 5] = 2, [25, 20, 15, 11, 9], 8   # closes over 5 frames, then stalls on the tape
    a[100:, 5] = s[100:, 5] = 30
    a[50:80, 11] = s[50:80, 11] = 30
    a[80:140, 11], s[80:85, 11], s[85:140, 11] = 3, 15, 10
    a[140:, 11] = s[140:, 11] = 30
    s[140:145, 11] = 15                   # R gripper takes 5 frames to open
    size = {"L": np.r_[np.zeros(10), np.full(n - 10, 0.5)], "R": np.zeros(n)}
    s[60:, 6] = 60                        # R comes into L's view at 60 (stand-in for the kinematics, tested in xlerobot_fk)
    s[170:, 6] = np.maximum(60 - 2 * np.arange(1, n - 169), 0)   # from 170 the R arm heads back to its start pose: reset
    real_view, xlerobot_fk.view_angle = xlerobot_fk.view_angle, lambda x, A, B: 10.0 if x[6] > 50 else 90.0
    try:
        lab, ref = fine9(s, a, size), std9(s, a, size)
        starts = lambda x: [b for b, _, _ in segments(x)]
        assert [g for _, _, g in segments(lab)] == list(range(9)) == [g for _, _, g in segments(ref)], (segments(lab), segments(ref))
        assert starts(ref) == [0, 10, 16, 19, 60, 81, 85, 141, 170], segments(ref)    # command events 1 frame later (frame t
        assert starts(lab) == [0, 14, 16, 28, 64, 81, 94, 141, 174], segments(lab)    # sees the command of t-1); holds settle
        # LiveStage (eval) gives the training labels frame by frame; tape_size stand-in reads the size from the pixel value
        real_size, globals()["tape_size"] = tape_size, lambda rgb: rgb[0, 0, 0] / 255
        live, prev, out = LiveStage(), None, []
        for i in range(n):
            obs = {k: v for k, v in zip(LiveStage.JOINTS, s[i])}
            obs.update({cam: np.full((4, 4, 3), round(size[k][i] * 255), np.uint8) for k, cam in CAM.items()})
            out.append(int(live(obs, prev).argmax()))
            prev = dict(zip(LiveStage.JOINTS, a[i]))
        globals()["tape_size"] = real_size
        assert out == lab.tolist(), segments(np.array(out))
        s2, a2 = s.copy(), a.copy()       # L closes on nothing first, reopens and grasps: stays in 抓住, never goes back
        a2[15:20, 5], s2[15:20, 5] = 2, 1
        a2[20:22, 5] = s2[20:22, 5] = 30
        lab2 = fine9(s2, a2, size)
        assert lab2[19] == 2 and lab2[35] == 3 and (np.diff(lab2) >= 0).all(), segments(lab2)
        assert compare(ref, ref)[0] == 1.0
        s3 = s.copy()                     # L still moving down while it starts closing at 16, stops at 17: A抓住 from 17
        s3[:18, 1] = np.linspace(-40, 0, 18)
        r3, f3 = std9(s3, a, size), fine9(s3, a, size)
        assert r3[16] == 1 and r3[17] == 2 and f3[21] == 1 and f3[22] == 2, (segments(r3), segments(f3))
    finally:
        xlerobot_fk.view_angle = real_view


if __name__ == "__main__":
    selfcheck()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset id under ~/.cache/huggingface/lerobot/wwqcleverghost")
    ap.add_argument("--rule", default="fine9", choices=RULES)
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--compare", action="store_true", help="also compare --rule with std9: frame agreement and lag per stage")
    x = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/workflows/dataset-viewer"))
    import server
    o = server.load(x.dataset)
    names, fn, _, vision = RULES[x.rule]
    done, total, agree, lags = 0, 0, [], {}
    for ep in x.episodes if x.episodes is not None else o["eps"].index:
        a, s = server.ep_arrays(o, ep)
        size = server.tape_sizes(x.dataset, ep) if vision or x.compare else None
        lab = fn(s, a, size)
        segs = segments(lab)
        back = sum(g2 < g1 for (_, _, g1), (_, _, g2) in zip(segs, segs[1:]))
        done += segs[-1][2] == len(names) - 1
        total += 1
        line = f"ep{ep:<3} {len(a) / o['fps']:5.1f}s {'回退 %d 次 ' % back if back else ''}" + \
            " → ".join(f"{names[g]} {(e - b) / o['fps']:.1f}s" for b, e, g in segs)
        if x.compare:
            fr, lag = compare(std9(s, a, size), lab)
            agree.append(fr)
            for g, d in lag.items():
                lags.setdefault(g, []).append(d)
            line = f"{line}  | 和 std9 一致 {fr:.0%}"
        print(line)
    print(f"{done}/{total} 条走到最后一个阶段（{names[-1]}）")
    if x.compare:
        print(f"逐帧一致：中位数 {np.median(agree):.0%}，平均 {np.mean(agree):.0%}，最低 {min(agree):.0%}")
        print("进入每个阶段比 std9 晚多少帧（中位数 / 最大 / 条数，负数 = 提早）：")
        for g in sorted(lags):
            d = np.array(lags[g])
            print(f"  {STAGES[g]:<5} {int(np.median(d)):>4} / {int(d[np.argmax(np.abs(d))]):>5} / {len(d)}")
