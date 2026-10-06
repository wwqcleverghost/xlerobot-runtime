"""
Multi-trial success-rate evaluation of a trained policy on a remote XLerobot (no dataset recording).
Pass a checkpoint from any of the ACT, SmolVLA, or Diffusion handover runs.
The task prompt defaults to the training dataset's single task.

Robot side (orin) first:
    PYTHONPATH=src python -m lerobot.robots.xlerobot.xlerobot_host --robot.id=joyandai_xlerobot
Then here:
    PYTHONPATH=src python examples/xlerobot/eval_remote_trials.py \
        --policy_path outputs/train/act_handover4/checkpoints/last/pretrained_model

Flow: arms reset -> place object, Enter -> episode 0 runs -> press -> (global key) when done ->
arms reset -> put the object back, then answer y/n (success?) in the terminal -> next episode starts.
Results are appended to --results_csv and the running success rate is printed.
Esc during an episode or Ctrl+C anytime quits.
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots.xlerobot.xlerobot_client import XLerobotClient, XLerobotClientConfig
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.control_utils import ask_success, init_keyboard_listener, predict_action
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device
from record_remote_bi_so101_leader_keyboard import smooth_arm_action  # same filter used when recording

BASE_STOP = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}


def dataset_start_pose(repo_id: str) -> dict[str, float]:
    """Median of the first observation.state of every episode in the local dataset."""
    root = HF_LEROBOT_HOME / repo_id
    names = json.loads((root / "meta/info.json").read_text())["features"]["observation.state"]["names"]
    df = pd.concat(
        pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
        for f in sorted((root / "data").rglob("*.parquet"))
    )
    firsts = df.sort_values("frame_index").groupby("episode_index").first()["observation.state"]
    median = np.median(np.stack(firsts.values), axis=0)
    return {n: float(v) for n, v in zip(names, median) if n.endswith(".pos")}


def move_to(robot, target: dict[str, float], fps: int, speed: float) -> None:
    """Cosine-eased joint interpolation from the current pose; duration set by the largest joint move."""
    obs = robot.get_observation()
    start = {k: float(obs[k]) for k in target}
    duration = max(2.0, max(abs(target[k] - start[k]) for k in target) / speed)
    n = int(duration * fps)
    for i in range(n + 1):
        a = 0.5 - 0.5 * math.cos(math.pi * i / n)
        robot.send_action({**{k: start[k] + a * (target[k] - start[k]) for k in target}, **BASE_STOP})
        time.sleep(1 / fps)


def run_episode(robot, policy, preprocessor, postprocessor, features, device, events, args) -> float:
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    action_sent = None
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.duration_s and not events["exit_early"]:
        t = time.perf_counter()
        obs = robot.get_observation()
        frame = build_dataset_frame(features, obs, prefix=OBS_STR)
        out = predict_action(
            frame, policy, device, preprocessor, postprocessor,
            policy.config.use_amp, task=args.task, robot_type=robot.robot_type,
        )
        action = make_robot_action(out, features)
        if args.max_arm_step > 0:
            prev = action_sent if action_sent is not None else obs
            action = smooth_arm_action(action, prev, max_step_per_frame=args.max_arm_step)
        if not args.allow_base:
            action.update(BASE_STOP)
        robot.send_action(action)
        action_sent = action
        precise_sleep(1 / args.fps - (time.perf_counter() - t))
    events["exit_early"] = False
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy_path", required=True)
    p.add_argument("--remote_ip", default="192.168.0.243")
    p.add_argument("--robot_id", default="joyandai_xlerobot")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--duration_s", type=float, default=60, help="Max seconds per episode")
    p.add_argument("--task", help="Task prompt (default: the training dataset's single task)")
    p.add_argument("--n_action_steps", type=int, help="ACT: steps executed per chunk (default: from checkpoint)")
    p.add_argument("--temporal_ensemble", type=float, help="ACT: temporal ensemble coeff, e.g. 0.01 (forces n_action_steps=1)")
    p.add_argument("--max_arm_step", type=float, default=3.0, help="Per-frame arm limit as in recording; 0 disables smoothing")
    p.add_argument("--allow_base", action="store_true", help="Let the policy drive the base (zeroed by default)")
    p.add_argument("--dataset_repo_id", help="Dataset for the reset pose (default: from the checkpoint's train_config.json)")
    p.add_argument("--results_csv", default="outputs/eval/results.csv", help="Per-episode results are appended here")
    p.add_argument("--reset_speed", type=float, default=20.0, help="Reset speed, joint units per second")
    args = p.parse_args()

    train_repo_id = json.loads(
        (Path(args.policy_path) / "train_config.json").read_text()
    )["dataset"]["repo_id"]
    repo_id = args.dataset_repo_id or train_repo_id
    reset_pose = dataset_start_pose(repo_id)
    if args.task is None:
        tasks = pd.read_parquet(HF_LEROBOT_HOME / train_repo_id / "meta/tasks.parquet").index
        if len(tasks) != 1:
            p.error("Training dataset has multiple tasks; pass --task explicitly")
        args.task = str(tasks[0])
    print(f"Policy task: {args.task}")

    robot = XLerobotClient(XLerobotClientConfig(remote_ip=args.remote_ip, id=args.robot_id))
    features = {
        **hw_to_dataset_features(robot.action_features, ACTION),
        **hw_to_dataset_features(robot.observation_features, OBS_STR),
    }

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    if args.temporal_ensemble is not None:
        cfg.temporal_ensemble_coeff, cfg.n_action_steps = args.temporal_ensemble, 1
    elif args.n_action_steps is not None:
        cfg.n_action_steps = args.n_action_steps
    policy = get_policy_class(cfg.type).from_pretrained(args.policy_path, config=cfg)
    device = get_safe_torch_device(policy.config.device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    listener, events = init_keyboard_listener()
    if listener is None:
        raise RuntimeError("Keyboard listener unavailable (headless?). Run this from a desktop terminal.")

    results_csv = Path(args.results_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not results_csv.exists()
    results = []

    try:
        robot.connect()
        print("Resetting arms...")
        move_to(robot, reset_pose, args.fps, args.reset_speed)
        input("Place the object, then press Enter to start episode 0: ")
        episode = 0
        while True:
            events["exit_early"] = False  # ignore -> presses made while not running
            print(f"Episode {episode} running, press -> when it is done (Esc to quit)")
            elapsed = run_episode(robot, policy, preprocessor, postprocessor, features, device, events, args)
            if events["stop_recording"]:
                break
            print("Resetting arms...")
            move_to(robot, reset_pose, args.fps, args.reset_speed)
            success = ask_success()
            results.append(success)
            with results_csv.open("a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["time", "policy_path", "episode", "success", "duration_s"])
                    new_file = False
                w.writerow([datetime.now().isoformat(timespec="seconds"), args.policy_path, episode, int(success), round(elapsed, 1)])
            print(f"Success rate: {sum(results)}/{len(results)} = {sum(results) / len(results):.0%}")
            episode += 1
    except KeyboardInterrupt:
        pass
    finally:
        if robot.is_connected:
            obs = robot.get_observation()
            robot.send_action({**{k: float(obs[k]) for k in reset_pose}, **BASE_STOP})  # hold pose, stop base
            robot.disconnect()
        listener.stop()


if __name__ == "__main__":
    main()
