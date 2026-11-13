"""Upload finished runs to private HF model repos wwqcleverghost/<name>: the final EMA policy, train config, processors,
train.log and a card that says which code and data trained it (the "=== ... start" lines queue_v3.sh writes into the log;
docs/workflows/sync/models.py reads them back into docs/models.csv).
usage (on the training machine, HF_TOKEN from secrets.env):
  cd <W> && set -a && . ./secrets.env && set +a && W=<W> HF_HOME=<W>/hf envs/lerobot/bin/python \
      xlerobot-runtime/docs/workflows/lab-training-queue/upload_model.py <name> ...
DRY=1 prints the card and uploads nothing."""
import json
import os
import sys


def card_tail(cfg: dict, starts: list[str]) -> str:
    data = cfg["dataset"]["repo_id"]
    return (f"\n\n{cfg['steps']} steps, batch {cfg['batch_size']}, training set "
            f"[{data}](https://huggingface.co/datasets/{data}).\n"
            "Training starts (code commit, training set revision):\n\n" + "".join(f"    {s}\n" for s in starts))


def main(names: list[str]) -> None:
    from huggingface_hub import HfApi
    from lerobot.policies.pretrained import PreTrainedPolicy

    W, api = os.environ["W"], HfApi()
    for name in names:
        folder, log = f"{W}/outputs/{name}/checkpoints/last/pretrained_model", f"{W}/logs/{name}.log"
        repo = "wwqcleverghost/" + name
        cfg = json.load(open(f"{folder}/train_config.json"))
        starts = [line.strip() for line in open(log, errors="replace") if line.startswith("=== ")]
        card = PreTrainedPolicy.generate_model_card(None, cfg["dataset"]["repo_id"], cfg["policy"]["type"], None, None)
        text = str(card) + card_tail(cfg, starts)
        if os.environ.get("DRY"):
            print(text[-600:])
            continue
        api.create_repo(repo, private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=folder, allow_patterns=["*.safetensors", "*.json"],
                          commit_message="Upload final EMA policy, train config and processors")
        api.upload_file(path_or_fileobj=text.encode(), path_in_repo="README.md", repo_id=repo)
        api.upload_file(path_or_fileobj=log, path_in_repo="train.log", repo_id=repo)
        local = {f: os.path.getsize(f"{folder}/{f}") for f in os.listdir(folder) if f.endswith((".safetensors", ".json"))}
        info = api.model_info(repo, files_metadata=True)
        remote = {s.rfilename: s.size for s in info.siblings}
        bad = [f for f, n in local.items() if remote.get(f) != n]
        print(name, info.sha[:8], f"MISMATCH {bad}" if bad else "sizes ok", flush=True)


if __name__ == "__main__":
    t = card_tail({"steps": 5, "batch_size": 2, "dataset": {"repo_id": "o/d"}}, ["=== x start m git abc data d@123"])
    assert "5 steps, batch 2" in t and "    === x start m git abc data d@123\n" in t and "datasets/o/d" in t
    main(sys.argv[1:])
