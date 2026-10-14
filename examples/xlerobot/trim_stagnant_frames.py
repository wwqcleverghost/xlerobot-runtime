"""Trim the idle head and tail of every XLeRobot episode (handover4 -> handover).

Recordings start before the operator moves and stop a while after the reset, so only the head and tail
are cut; pauses inside an episode are kept, so every episode stays one continuous recording.
- Head: keep from the last frame at the start position, i.e. the last still frame (no joint moves more
  than --max_delta in the action or in observation.state) before the command first leaves the start pose
  by more than --start_tol. The robot settling into the start pose is not a task start: only the command
  counts for that.
- Tail: keep until the reset is basically done, i.e. through the first frame from which every joint (action
  and state) stays within --reset_tol of the final pose. The full reset is done by the eval script.
The kept frames get contiguous timestamps and are re-encoded like a recording (libx264).
The defaults turn handover4 (37,551 frames) into 29,002 frames.

Push afterwards with: LeRobotDataset("wwqcleverghost/handover").push_to_hub(private=True)
"""

import argparse

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES


def keep_range(actions: np.ndarray, states: np.ndarray, pos: list[int], max_delta: float, start_tol: float,
               reset_tol: float) -> tuple[int, int]:
    """actions, states: (T, D) for one episode. Returns (first, end): keep frames [first, end)."""
    a, p = actions[:, pos], np.c_[actions[:, pos], states[:, pos]]
    still = np.r_[0.0, np.abs(np.diff(p, axis=0)).max(1)] <= max_delta
    ref = a[min(2, len(a) - 1)]  # frames 0-1 often hold a recording-start jitter
    left = np.abs(a - ref).max(1) > start_tol
    task = int(np.argmax(left)) if left.any() else len(a)
    first = int(np.flatnonzero(still[:task])[-1]) if still[:task].any() else 0
    away = np.abs(p - p[-1]).max(1) >= reset_tol
    end = min(int(np.flatnonzero(away)[-1]) + 2, len(a)) if away.any() else len(a)
    assert first < end, (first, end)
    return first, end


def check_keep_range():
    # 2 jitter frames, idle at 0 until frame 5, task 6-9, reset back to ~0 with a slow creep, idle at the end.
    a = np.array([0, 0.5, 0, 0, 0, 0, 3, 8, 8, 4, 1.5, 0.8, 0.4, 0, 0, 0], float)[:, None]
    assert keep_range(a, a, [0], 0.2, 2, 1) == (5, 12)  # from the last start frame to the first frame within 1
    # The state still settling into the start pose (frames 2-4) does not start the task.
    s = a + np.r_[0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0][:, None]
    assert keep_range(a, s, [0], 0.2, 2, 1) == (5, 12)
    # Recording stopped mid-reset: nothing to cut at the tail.
    assert keep_range(a[:10], a[:10], [0], 0.2, 2, 1) == (5, 10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="wwqcleverghost/handover4")
    parser.add_argument("--dst", default="wwqcleverghost/handover")
    parser.add_argument("--max_delta", type=float, default=0.2)
    parser.add_argument("--start_tol", type=float, default=2.0)
    parser.add_argument("--reset_tol", type=float, default=3.0)
    parser.add_argument("--num_workers", type=int, default=12)
    args = parser.parse_args()
    check_keep_range()

    src = LeRobotDataset(args.src)
    names = src.meta.features["action"]["names"]
    assert src.meta.features["observation.state"]["names"] == names
    pos = [i for i, n in enumerate(names) if n.endswith(".pos")]
    vel = [i for i, n in enumerate(names) if n.endswith(".vel")]
    table = src.hf_dataset.with_format("numpy")
    actions, states = np.asarray(table["action"]), np.asarray(table["observation.state"])
    assert not actions[:, vel].any(), "the base moves: keep_range only looks at the arm/head joints"
    kept = []
    for lo, hi in zip(src.meta.episodes["dataset_from_index"], src.meta.episodes["dataset_to_index"]):
        first, end = keep_range(
            actions[lo:hi], states[lo:hi], pos, args.max_delta, args.start_tol, args.reset_tol
        )
        kept.extend(range(lo + first, lo + end))
    print(f"keeping {len(kept)} / {src.num_frames} frames", flush=True)

    features = {
        k: {f: v for f, v in ft.items() if f != "info"}
        for k, ft in src.meta.features.items()
        if k not in DEFAULT_FEATURES
    }
    dst = LeRobotDataset.create(
        args.dst, fps=src.fps, features=features, robot_type=src.meta.robot_type, image_writer_threads=12
    )
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(src, [int(i) for i in kept]), batch_size=None, num_workers=args.num_workers
    )
    episode = None
    for item in loader:
        if episode is not None and int(item["episode_index"]) != episode:
            dst.save_episode()
        episode = int(item["episode_index"])
        frame = {"task": item["task"]}
        for key, ft in features.items():
            value = item[key]
            if ft["dtype"] == "video":
                value = (value.permute(1, 2, 0) * 255).round().to(torch.uint8)
            frame[key] = value
        dst.add_frame(frame)
    dst.save_episode()
    dst.finalize()
    assert dst.meta.total_frames == len(kept), (dst.meta.total_frames, len(kept))
    assert dst.meta.total_episodes == src.meta.total_episodes
    print(f"wrote {dst.root}: {dst.meta.total_episodes} episodes, {dst.meta.total_frames} frames")


if __name__ == "__main__":
    main()
