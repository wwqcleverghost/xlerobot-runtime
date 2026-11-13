# Keeping the machines in sync: where the original of each thing lives

中文: [README.zh.md](README.zh.md)

One rule: **each thing has one original; what the other machines hold are copies.** When a copy differs, the original wins.

| Thing | Original | How it stays the same everywhere |
|---|---|---|
| Code, docs, spec JSON | GitHub `main` (the repo is public) | Every machine is a git checkout. Laptop and lab computer commit and push the same day; servers only `git pull --ff-only`, nothing is edited there |
| Training sets, models, eval recordings | HF `wwqcleverghost/` (private) | A training set is **uploaded before it is trained on**; `queue_v3.sh` downloads one that is missing on the training machine and writes down its revision; models go up with `upload_model.py`; evals as `eval-<model>` |
| Training curves | W&B `wwqcleverghost/handover` | Written while training |
| What each model is | [`docs/models.csv`](../../models.csv) | `models.py` regenerates it from HF; only the two note columns are typed |
| Agent memory | Each machine's own memory folder (not in the repo: it holds machine addresses) | `memory.sh` on the laptop syncs both ways |

## Machines

| Machine | What it does | Code folder |
|---|---|---|
| Laptop | Write code, process data, queue training; ssh to the others with the NUS VPN on | `~/xlerobot-runtime` |
| Lab computer | Record data, robot evals, the dataset viewer (8770) | `~/wwq/lerobot` |
| server1, server2 | Training only | `/media/data/wwq/xlerobot-runtime` |
| server4, server6 | Training only | `~/wwq/xlerobot-runtime` |

A server's `<W>` (`/media/data/wwq` or `~/wwq`) holds only `secrets.env`, `envs/`, `data/`, `hf/`, `logs/`, `outputs/`,
`xlerobot-runtime/`. Scripts run straight from the repo, not from copies in `<W>`. Servers pull anonymously; no GitHub
credentials are kept there.

## One experiment, once around

1. **Data** (laptop or lab computer): prepare it, name it by [`../naming/`](../naming/README.md), upload it to HF.
2. **Queue training** (laptop, over ssh to the server):

   ```bash
   cd <W>/xlerobot-runtime && git pull --ff-only
   WQ=<W> GPU=<n> setsid -f bash docs/workflows/lab-training-queue/queue_v3.sh <jobs> >> <W>/logs/queue_v3_gpu<n>.log 2>&1 < /dev/null
   ```

   Every start adds a line `=== <time> start <model> git <commit> data <training set>@<revision>` to `<W>/logs/<model>.log`.
   `+dirty` after the commit means that machine had uncommitted changes. A local data folder without a `REVISION` file
   (not downloaded from HF) makes the job skip: upload it first.
3. **After training** (server): `upload_model.py <model>` uploads the final model, `train.log` and a model card with those
   lines (usage at the top of the script).
4. **Eval** (lab computer): see [`../eval-scoring/`](../eval-scoring/README.md); upload the recording as `eval-<model>`.
5. **Register** (laptop):

   ```bash
   python docs/workflows/sync/models.py
   ```

   It lists the models that have no note yet; fill `note_zh` and `note_en` in `docs/models.csv`, commit and push.

## Where every machine stands

```bash
bash docs/workflows/sync/where.sh
```

One line per machine: code commit, commits behind `main`, uncommitted files, running training. The lab computer's address
changes often; then `LAB_IP=10.249.x.x bash docs/workflows/sync/where.sh` (`hostname -I` on the lab computer shows it).

Before leaving a machine: uncommitted files should be 0 (commit and push). On arriving at one: `git pull --ff-only`.

## The model table `docs/models.csv`

One row per model on HF. `note_zh` / `note_en` say in one sentence what it is; `dataset`, `steps`, `batch`, `wandb` come from
the model repo's `train_config.json`; `git` and `data_rev` are the code commit and training set revision it was trained with
(for models trained before 2026-11-14 only the checked ones are filled, the rest are empty); `eval` lists its test sets as
`name@revision successes/trials`. Total scores with confidence intervals: `../eval-scoring/report.py` or the dataset viewer.

## Memory

```bash
bash docs/workflows/sync/memory.sh
```

Completes the memory folders of the laptop and the lab computer from each other: of two files with the same name the newer
one is kept, the index lines of `MEMORY.md` are merged.

State when this was set up on 2026-11-14: `main` replaced the former `diffusion-flow-react` branch; the four servers went
from copies to git checkouts; `v6-handover-curated-stagelabel` (`1b87da63` = labels of 11-03 18:35, `2eb9ebda` = labels of
19:00), `-stagepred` (`f1a4ffed`), `-progpred` (`b59eb468`) and `-cumprog` (`7c8ed927`) were uploaded to HF, and the training
copies on the machines are byte-identical to these revisions.
