"""Refresh docs/models.csv, the one table of every model: what it is, what trained it, how it did on the robot.
Read from HF on every run (never typed):
  dataset, steps, batch, wandb = train_config.json of the model repo
  git, data_rev                = the "... start <name> git <commit> data <set>@<rev>" line upload_model.py puts in the card
  eval                         = the eval-<model>* test sets: name@revision successes/trials (their success.csv)
Typed by a person or an agent and kept as they are: note_zh, note_en, and git / data_rev of models whose card has no
start line (trained before 2026-11-14).
usage: python docs/workflows/sync/models.py     (needs the HF token; lists the models that still have no note)"""
import csv
import json
import re
from pathlib import Path

OWNER = "wwqcleverghost"
CSV = Path(__file__).resolve().parents[2] / "models.csv"
COLS = ["model", "note_zh", "note_en", "dataset", "data_rev", "git", "steps", "batch", "wandb", "eval"]
START = re.compile(r"start \S+ git (\S+) data \S+@(\w*)")


def owner_of(eval_name: str, models: list[str]) -> str | None:
    """The model a test set belongs to: the longest model name that eval-<...> starts with (then the end or '_')."""
    n = eval_name.removeprefix("eval-")
    return max((m for m in models if n == m or n.startswith(m + "_")), key=len, default=None)


def success_files(files: list[str]) -> list[str]:
    """success.csv of every set in a test set repo; <x>/dataset/success.csv (the one the viewer edits) wins over <x>/success.csv."""
    f = [x for x in files if x.endswith("success.csv")]
    return [x for x in f if x.replace("success.csv", "dataset/success.csv") not in f]


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    rows = {r["model"]: r for r in csv.DictReader(CSV.open())} if CSV.exists() else {}
    models = sorted(m.id.split("/")[1] for m in api.list_models(author=OWNER))
    evals: dict[str, list[str]] = {}
    for d in sorted(x.id for x in api.list_datasets(author=OWNER)):
        m = owner_of(d.split("/")[1], models) if d.split("/")[1].startswith("eval-") else None
        if m is None:
            continue
        info = api.dataset_info(d)
        marks = [r["success"].strip().lower() in ("1", "true") for f in success_files([s.rfilename for s in info.siblings])
                 for r in csv.DictReader(open(hf_hub_download(d, f, repo_type="dataset")))]
        evals.setdefault(m, []).append(f"{d.split('/')[1]}@{info.sha[:8]} {sum(marks)}/{len(marks)}")
    for m in models:
        row = rows.setdefault(m, dict.fromkeys(COLS, ""))
        row["model"], row["eval"] = m, "; ".join(evals.get(m, []))
        files = [s.rfilename for s in api.model_info(f"{OWNER}/{m}").siblings]
        cfgs = sorted(f for f in files if f.endswith("train_config.json"))  # root first, else the last checkpoint
        if cfgs:
            c = json.load(open(hf_hub_download(f"{OWNER}/{m}", cfgs[0] if "/" not in cfgs[0] else cfgs[-1])))
            w = c.get("wandb") or {}
            row.update(dataset=c["dataset"]["repo_id"].split("/")[-1], steps=c.get("steps", ""), batch=c.get("batch_size", ""),
                       wandb=f"{w.get('project')}/{w.get('run_id')}" if w.get("run_id") else "")
        if "README.md" in files:
            starts = START.findall(open(hf_hub_download(f"{OWNER}/{m}", "README.md")).read())
            if starts:
                row["git"], row["data_rev"] = starts[-1][0], starts[-1][1] or row["data_rev"]
    with CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows[m] for m in sorted(rows))
    gone, bare = sorted(set(rows) - set(models)), [m for m in models if not rows[m]["note_zh"]]
    print(f"{CSV}: {len(rows)} models" + (f"\nnot on HF any more: {gone}" if gone else "") + (f"\nno note yet: {bare}" if bare else ""))


if __name__ == "__main__":
    ms = ["v6-flow_h16_s8", "v6-flow_h16_s8_progdelta", "v6-flow_h16_s8_progdelta_rabc"]
    assert owner_of("eval-v6-flow_h16_s8_progdelta_rabc", ms) == ms[2] and owner_of("eval-v6-flow_h16_s8_progdelta_s2", ms) == ms[1]
    assert owner_of("eval-v6-flow_h16_s8_takeover", ms) == ms[0] and owner_of("eval-v1-records_20261001", ms) is None
    assert success_files(["success.csv", "dataset/success.csv", "set1/dataset/success.csv", "a/success.csv.bak"]) == [
        "dataset/success.csv", "set1/dataset/success.csv"]
    assert START.findall("=== 2026-11-14 10:00:00 start m git abc1234+dirty data d@7c8ed927") == [("abc1234+dirty", "7c8ed927")]
    main()
