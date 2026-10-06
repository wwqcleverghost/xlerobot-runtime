"""Offline checkpoint smoke test using one local dataset frame.

Run with PYTHONPATH pointing at the local-training-test-run Claude worktree's src
directory for the diffusion checkpoint, which needs its resize_shape support.
This checks loading and action inference, not handover success.
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
    args = parser.parse_args()
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
    cfg.device = "cpu"
    cfg.use_amp = False
    started = time.monotonic()
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg, strict=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    loaded = time.monotonic()
    action = predict_action(
        observation, policy, torch.device("cpu"), preprocessor, postprocessor,
        False, task=sample.get("task", "handover"), robot_type="xlerobot_client",
    )
    elapsed = time.monotonic() - loaded
    action = action.detach().cpu().numpy().reshape(-1)
    expected = sample["action"].numel()
    assert action.size == expected, (action.size, expected)
    assert np.isfinite(action).all(), action
    print(json.dumps({
        "checkpoint": str(checkpoint), "policy_type": cfg.type,
        "action_dim": int(action.size), "load_seconds": round(loaded - started, 2),
        "inference_seconds": round(elapsed, 2), "action_min": float(action.min()),
        "action_max": float(action.max()), "status": "pass",
    }), flush=True)


if __name__ == "__main__":
    main()
