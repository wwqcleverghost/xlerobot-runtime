"""Trim the idle head and tail of every XLeRobot episode (handover2 -> handover2-trimmed).

Recordings start before the operator moves and stop a while after the reset, so only the head and tail
are cut; pauses inside an episode are kept, so every episode stays one continuous recording.
- Head: keep from the last frame at the start position, i.e. the last still frame (no joint moves more
  than --max_delta in the action or in observation.state) before the command first leaves the start pose
  by more than --start_tol. The robot settling into the start pose is not a task start: only the command
  counts for that.
- Tail: keep until the reset is basically done, i.e. through the first frame from which every joint (action
  and state) stays within --reset_tol of the final pose. The full reset is done by the eval script.
The kept frames get contiguous timestamps and are re-encoded like a recording (libx264).
The defaults turn handover2 (37,551 frames) into 29,002 frames.

--spec merges several datasets into one instead of --src: a JSON list of {"repo": ..., "drop": [episodes],
"start_s": {"<episode>": seconds}}. "drop" leaves episodes out; "start_s" starts an episode that many seconds
after its trimmed start (e.g. right after a failed grasp, to keep only the recovery). See handover_perfect.json.
"cut": {"<episode>": [[start, end], ...]} removes those frame ranges [start, end) (frame index within the source episode)
from inside the kept range and joins the rest, e.g. to shorten a mid-episode pause where the arms stand still.
"root" reads a local dataset directory instead of the HF cache. "segments": {"<episode>": [[start, end], ...]}
keeps exactly those frame ranges [start, end) instead of trimming, each range as its own episode (e.g. human
takeover corrections cut out of eval recordings, see takeover_corrections.json); "drop_seg": {"<episode>": [i, ...]}
leaves out those segments (by index; the dataset viewer writes it).
"range": {"<episode>": [first, end]} keeps exactly frames [first, end) of that episode instead of the head/tail rule
(it replaces keep_range, --to_target and start_s there; "cut" still applies). The dataset viewer
(docs/workflows/dataset-viewer) writes it for every episode before a rebuild, so the build is what the page shows.

Push afterwards with: LeRobotDataset("wwqcleverghost/v3-handover2-trimmed").push_to_hub(private=True)
"""

import argparse
import copy
import os
import json

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES


def keep_range(actions: np.ndarray, states: np.ndarray, pos: list[int], max_delta: float, start_tol: float,
               reset_tol: float) -> tuple[int, int]:
    """actions, states: (T, D) for one episode. Returns (first, end): keep frames [first, end)."""
    a, p = actions[:, pos], np.c_[actions[:, pos], states[:, pos]]
    still = np.r_[0.0, np.abs(np.diff(p, axis=0)).max(1)] <= max_delta
    j = min(2, len(a) - 1)
    ref = a[j]  # frames 0-1 often hold a recording-start jitter
    left = np.abs(a - ref).max(1) > start_tol
    left[:j] = False  # the jitter itself is not the task leaving the start pose
    task = int(np.argmax(left)) if left.any() else len(a)
    first = int(np.flatnonzero(still[:task])[-1]) if still[:task].any() else 0
    away = np.abs(p - p[-1]).max(1) >= reset_tol
    end = min(int(np.flatnonzero(away)[-1]) + 2, len(a)) if away.any() else len(a)
    assert first < end, (first, end)
    return first, end


def reach_start(actions: np.ndarray, states: np.ndarray, first: int, end: int, near: float = 0.05) -> int:
    """With --to_target: start the episode where the arm really heads for the tape. Target = the command at the first
    grasp (gripper commanded closed but held open by the tape for >= 0.5 s); progress = how much of the way from the
    start pose to the target the grasping arm's 5 joints have covered. Keeps from the last frame with progress <= near,
    so jitter or wandering near the start pose before the reach is cut. Arm joints 0-4 / gripper 5 = left, 6-10 / 11 = right."""
    best = None
    for arm, g in ((slice(0, 5), 5), (slice(6, 11), 11)):
        hold = (actions[first:end, g] < states[first:end, g] - 2) & (states[first:end, g] > 4)
        run = np.convolve(hold, np.ones(15), "valid") >= 15
        if run.any() and (best is None or np.argmax(run) < best[0]):
            best = (int(np.argmax(run)), arm)
    if best is None:
        return first
    tg, arm = first + best[0], best[1]
    a = actions[first:tg + 1, arm]
    total = np.linalg.norm(a[0] - a[-1])
    if total < 1e-6:
        return first
    progress = 1 - np.linalg.norm(a - a[-1], axis=1) / total
    return first + int(np.flatnonzero(progress <= near)[-1])


def episode_range(actions, states, pos, args, start_shift=0, exact=None):
    """(first, end) of one episode: the spec's exact "range" when given, else the head/tail rule (+ --to_target, start_s)."""
    if exact is not None:
        first, end = map(int, exact)
        assert 0 <= first < end <= len(actions), (first, end, len(actions))
        return first, end
    first, end = keep_range(actions, states, pos, args.max_delta, args.start_tol, args.reset_tol)
    if args.to_target:
        first = reach_start(actions, states, first, end)
    return first + start_shift, end


def check_episode_range():
    a = np.array([0, 0.5, 0, 0, 0, 0, 3, 8, 8, 4, 1.5, 0.8, 0.4, 0, 0, 0], float)[:, None]
    args = argparse.Namespace(max_delta=0.2, start_tol=2, reset_tol=1, to_target=False)
    assert episode_range(a, a, [0], args) == (5, 12)
    assert episode_range(a, a, [0], args, start_shift=2) == (7, 12)
    assert episode_range(a, a, [0], args, start_shift=2, exact=[3, 14]) == (3, 14)  # exact range wins over rule and start_s


def check_reach_start():
    # left arm: idle, jitter around the start, then a straight reach to 10 and a grasp held from frame 12
    a = np.zeros((40, 12)); a[3:7, 0] = [1, -1, 1, -1]; a[7:12, 0] = np.linspace(2, 10, 5); a[12:, 0] = 10
    s = a.copy(); a[12:, 5] = 1; s[12:, 5] = 8
    assert reach_start(a, s, 0, 40) == 6, reach_start(a, s, 0, 40)


def check_keep_range():
    # 2 jitter frames, idle at 0 until frame 5, task 6-9, reset back to ~0 with a slow creep, idle at the end.
    a = np.array([0, 0.5, 0, 0, 0, 0, 3, 8, 8, 4, 1.5, 0.8, 0.4, 0, 0, 0], float)[:, None]
    assert keep_range(a, a, [0], 0.2, 2, 1) == (5, 12)  # from the last start frame to the first frame within 1
    # The state still settling into the start pose (frames 2-4) does not start the task.
    s = a + np.r_[0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][:, None]
    assert keep_range(a, s, [0], 0.2, 2, 1) == (5, 12)
    # Recording stopped mid-reset: nothing to cut at the tail.
    assert keep_range(a[:10], a[:10], [0], 0.2, 2, 1) == (5, 10)
    # A start jitter that leaves the command more than start_tol away (episodes 0, 7, 26 of handover2)
    # does not start the task either.
    b = np.array([3, 1.5, 0, 0, 0, 0, 0, 4, 8, 8, 0, 0], float)[:, None]
    assert keep_range(b, b, [0], 0.2, 2, 1) == (6, 11)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="wwqcleverghost/v2-handover2")
    parser.add_argument("--dst", default="wwqcleverghost/v3-handover2-trimmed")
    parser.add_argument("--max_delta", type=float, default=0.2)
    parser.add_argument("--start_tol", type=float, default=2.0)
    parser.add_argument("--reset_tol", type=float, default=3.0)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--spec", help="JSON file merging several sources (see the module docstring)")
    parser.add_argument("--to_target", action="store_true",
                        help="start each episode where the grasping arm really heads for the tape (see reach_start)")
    parser.add_argument("--arm_only", action="store_true",
                        help="keep only the 12 arm joints in action/state (what training uses), e.g. to merge 12-dim eval recordings")
    args = parser.parse_args()
    check_keep_range()
    check_reach_start()
    check_episode_range()
    sources = json.load(open(args.spec)) if args.spec else [{"repo": args.src}]

    dst, features, n_frames, n_episodes = None, None, 0, 0
    for source in sources:
        root = source.get("root")
        src = LeRobotDataset(source["repo"], root=root and os.path.expanduser(root))
        names = src.meta.features["action"]["names"]
        assert src.meta.features["observation.state"]["names"] == names
        # Arm joints only: the head servos are not part of the task, and their reading drift is not motion.
        pos = [i for i, n in enumerate(names) if n.endswith(".pos") and not n.startswith("head_")]
        vel = [i for i, n in enumerate(names) if n.endswith(".vel")]
        table = src.hf_dataset.with_format("numpy")
        actions, states = np.asarray(table["action"]), np.asarray(table["observation.state"])
        assert not actions[:, vel].any(), "the base moves: keep_range only looks at the arm joints"
        drop, start_s = set(source.get("drop", [])), {int(k): v for k, v in source.get("start_s", {}).items()}
        cut = {int(k): v for k, v in source.get("cut", {}).items()}
        exact = {int(k): v for k, v in source.get("range", {}).items()}
        bounds = list(zip(src.meta.episodes["dataset_from_index"], src.meta.episodes["dataset_to_index"]))
        episodes = []  # one list of source frame indices per output episode
        if "segments" in source:
            for ep, ranges in source["segments"].items():
                lo, hi = bounds[int(ep)]
                for i, (start, end) in enumerate(ranges):
                    if i in source.get("drop_seg", {}).get(ep, []):
                        continue
                    assert 0 <= start < end <= hi - lo, (source["repo"], ep, start, end)
                    episodes.append(list(range(lo + start, lo + end)))
        else:
            for ep, (lo, hi) in enumerate(bounds):
                if ep in drop:
                    continue
                first, end = episode_range(actions[lo:hi], states[lo:hi], pos, args, round(start_s.get(ep, 0) * src.fps),
                                           exact.get(ep))
                assert first < end, (source["repo"], ep, first, end)
                cut_out = {i for c0, c1 in cut.get(ep, []) for i in range(c0, c1)}
                assert all(first <= i < end for i in cut_out), (source["repo"], ep, "cut outside the kept range")
                episodes.append([lo + i for i in range(first, end) if i not in cut_out])
        n_kept = sum(map(len, episodes))
        print(f"{source['repo']}: keeping {len(episodes)} episodes, {n_kept} / {src.num_frames} frames", flush=True)

        src_features = {
            k: {f: v for f, v in ft.items() if f != "info"}
            for k, ft in src.meta.features.items()
            if k not in DEFAULT_FEATURES
        }
        if args.arm_only:
            for k in ("action", "observation.state"):
                src_features[k] = {**src_features[k], "shape": (len(pos),), "names": [names[i] for i in pos]}
        if dst is None:
            features = src_features
            # A deep copy: the dataset writes each video's encoding "info" into its feature dicts.
            dst = LeRobotDataset.create(
                args.dst, fps=src.fps, features=copy.deepcopy(features), robot_type=src.meta.robot_type,
                image_writer_threads=12,
            )
        assert src_features == features and src.fps == dst.fps, f"{source['repo']} does not match the first source"
        for frames in episodes:
            loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(src, frames), batch_size=None, num_workers=args.num_workers
            )
            for item in loader:
                frame = {"task": item["task"]}
                for key, ft in features.items():
                    value = item[key]
                    if args.arm_only and key in ("action", "observation.state"):
                        value = value[pos]
                    if ft["dtype"] == "video":
                        value = (value.permute(1, 2, 0) * 255).round().to(torch.uint8)
                    frame[key] = value
                dst.add_frame(frame)
            dst.save_episode()
        n_frames, n_episodes = n_frames + n_kept, n_episodes + len(episodes)
    dst.finalize()
    assert dst.meta.total_frames == n_frames, (dst.meta.total_frames, n_frames)
    assert dst.meta.total_episodes == n_episodes, (dst.meta.total_episodes, n_episodes)
    print(f"wrote {dst.root}: {dst.meta.total_episodes} episodes, {dst.meta.total_frames} frames")


if __name__ == "__main__":
    main()
