"""Keep the last three model snapshots and only the latest complete training state."""

import json
import shutil
import sys
from pathlib import Path


def prune_checkpoints(run: Path) -> None:
    root = run / "checkpoints"
    link = root / "last"
    if not link.is_symlink() or not link.exists():
        return
    latest = link.resolve()
    if latest.parent != root.resolve() or not latest.name.isdecimal():
        raise ValueError(f"Unexpected last checkpoint: {latest}")
    # last is published only after all weights and training state have been saved.
    required = [
        latest / "pretrained_model" / "model.safetensors",
        latest / "pretrained_model" / "config.json",
        latest / "pretrained_model" / "train_config.json",
        *(
            latest / "training_state" / name
            for name in (
                "training_step.json", "optimizer_param_groups.json", "raw_model.safetensors",
                "optimizer_state.safetensors", "scheduler_state.json", "rng_state.safetensors",
            )
        ),
    ]
    if any(not p.is_file() or p.stat().st_size == 0 for p in required):
        return
    step = json.loads((latest / "training_state" / "training_step.json").read_text())["step"]
    if step != int(latest.name):
        raise ValueError(f"Checkpoint step mismatch: {latest}")
    completed = sorted(
        (p for p in root.iterdir()
         if p.is_dir() and not p.is_symlink() and p.name.isdecimal() and int(p.name) <= step),
        key=lambda p: int(p.name),
    )
    for path in completed[:-3]:
        shutil.rmtree(path)
        print(f"Removed old checkpoint: {path}")
    for path in completed[-3:-1]:
        state = path / "training_state"
        if state.is_dir():
            shutil.rmtree(state)
            print(f"Removed older training state: {state}")


if __name__ == "__main__":
    prune_checkpoints(Path(sys.argv[1]))
