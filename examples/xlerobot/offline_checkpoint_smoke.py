"""Offline checkpoint smoke test using one local dataset frame.

This checks loading and action inference, not handover success.
Use --device cuda --repeats 4 to measure full action-chunk inference after warmup.
For Diffusion, --num_inference_steps overrides the checkpoint's sampling steps.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.control_utils import predict_action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--num_inference_steps", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    checkpoint = args.checkpoint.resolve()
    train = json.loads((checkpoint / "train_config.json").read_text())
    repo_id = train["dataset"]["repo_id"]
    dataset_root = Path.home() / ".cache/huggingface/lerobot" / repo_id
    assert dataset_root.is_dir(), dataset_root

    ds = LeRobotDataset(repo_id, root=dataset_root)
    sample = ds[0]
    observation = {}
    for key, value in sample.items():
        if key == "observation.state":
            observation[key] = value.numpy().astype(np.float32)
        elif key.startswith("observation.images."):
            observation[key] = (value.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.device = args.device
    if args.device == "cpu":
        cfg.use_amp = False
    if args.num_inference_steps is not None:
        if cfg.type != "diffusion":
            parser.error("--num_inference_steps requires a Diffusion checkpoint")
        if not 1 <= args.num_inference_steps <= cfg.num_train_timesteps:
            parser.error("--num_inference_steps must be between 1 and num_train_timesteps")
        cfg.num_inference_steps = args.num_inference_steps
    started = time.monotonic()
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg, strict=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    loaded = time.monotonic()
    expected = sample["action"].numel()
    timings = []
    for _ in range(args.repeats):
        # Clear cached actions so every repetition measures a full model prediction.
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        started_inference = time.monotonic()
        action = predict_action(
            observation, policy, torch.device(args.device), preprocessor, postprocessor,
            cfg.use_amp, task=sample.get("task", "handover"), robot_type="xlerobot_client",
        ).detach().cpu().numpy().reshape(-1)
        timings.append(time.monotonic() - started_inference)
        assert action.size == expected, (action.size, expected)
        assert np.isfinite(action).all(), action
    print(json.dumps({
        "checkpoint": str(checkpoint), "policy_type": cfg.type,
        "device": args.device, "use_amp": cfg.use_amp,
        "num_inference_steps": getattr(getattr(policy, "diffusion", None), "num_inference_steps", None),
        "action_dim": int(action.size), "load_seconds": round(loaded - started, 2),
        "inference_seconds": round(timings[0], 3),
        "median_inference_seconds": round(float(np.median(timings[1:] or timings)), 3),
        "timings_seconds": [round(t, 3) for t in timings], "action_min": float(action.min()),
        "action_max": float(action.max()), "status": "pass",
    }), flush=True)


if __name__ == "__main__":
    main()
