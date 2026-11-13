# 多机同步：每样东西的正本在哪

English: [README.md](README.md)

一条规则：**每样东西只有一个正本，其他机器上的都是拷贝。** 拷贝和正本不一样时，以正本为准。

| 东西 | 正本 | 怎么保持一致 |
|---|---|---|
| 代码、文档、规格 JSON | GitHub `main`（仓库是公开的） | 每台机器都是 git 仓库。笔记本和实验室电脑改完当天提交推送；服务器只 `git pull --ff-only`，不在上面改 |
| 训练集、模型、评测录像 | HF `wwqcleverghost/`（私有） | 训练集**先上传再训练**；训练机缺数据时 `queue_v3.sh` 自己从 HF 下载并记下 revision；模型用 `upload_model.py` 上传；评测上传为 `eval-<模型>` |
| 训练曲线 | W&B `wwqcleverghost/handover` | 训练时自动写 |
| 每个模型是什么 | [`docs/models.csv`](../../models.csv) | `models.py` 从 HF 重新生成；只有说明两列是手写的 |
| agent 记忆 | 各机器自己的记忆目录（不进仓库：里面有机器地址） | 在笔记本上跑 `memory.sh` 双向同步 |

## 机器

| 机器 | 做什么 | 代码目录 |
|---|---|---|
| 笔记本 | 写代码、处理数据、排训练；开 NUS VPN 后 ssh 到其他机器 | `~/xlerobot-runtime` |
| 实验室电脑 | 录数据、真机评测、常驻网页工具（8770） | `~/wwq/lerobot` |
| server1、server2 | 只跑训练 | `/media/data/wwq/xlerobot-runtime` |
| server4、server6 | 只跑训练 | `~/wwq/xlerobot-runtime` |

服务器的 `<W>`（`/media/data/wwq` 或 `~/wwq`）下只放 `secrets.env`、`envs/`、`data/`、`hf/`、`logs/`、`outputs/`、
`xlerobot-runtime/`。脚本直接从仓库里运行，不拷到 `<W>`。服务器匿名拉代码，不放 GitHub 凭据。

## 一次实验走一圈

1. **数据**（笔记本或实验室电脑）：整理好，按 [`../naming/`](../naming/README.zh.md) 起名，上传到 HF。
2. **排训练**（笔记本 ssh 到服务器）：

   ```bash
   cd <W>/xlerobot-runtime && git pull --ff-only
   WQ=<W> GPU=<n> setsid -f bash docs/workflows/lab-training-queue/queue_v3.sh <任务> >> <W>/logs/queue_v3_gpu<n>.log 2>&1 < /dev/null
   ```

   每次启动，`<W>/logs/<模型>.log` 里会多一行 `=== <时间> start <模型> git <提交> data <训练集>@<revision>`。
   `git` 后面带 `+dirty` 说明那台机器上有没提交的改动。本地有数据但没有 `REVISION` 文件（不是从 HF 下载的）时任务会跳过：先上传。
3. **训练完**（服务器）：`upload_model.py <模型>` 上传最终模型、`train.log` 和写着上面那几行的模型卡片（用法见脚本开头）。
4. **评测**（实验室电脑）：见 [`../eval-scoring/`](../eval-scoring/README.zh.md)；录完上传为 `eval-<模型>`。
5. **登记**（笔记本）：

   ```bash
   python docs/workflows/sync/models.py
   ```

   它会列出还没有说明的模型；在 `docs/models.csv` 里补上 `note_zh`、`note_en` 两列，提交推送。

## 看各机器现在的状态

```bash
bash docs/workflows/sync/where.sh
```

每台机器一行：代码提交、落后 `main` 几个提交、没提交的文件数、正在跑的训练。实验室电脑的地址经常变，变了就
`LAB_IP=10.249.x.x bash docs/workflows/sync/where.sh`（地址在实验室电脑上用 `hostname -I` 看）。

离开一台机器前：没提交的文件数应为 0（提交并推送）。到一台机器先 `git pull --ff-only`。

## 模型总表 `docs/models.csv`

一行一个 HF 上的模型。`note_zh` / `note_en` 是一句话说明；`dataset`、`steps`、`batch`、`wandb` 来自模型仓库的
`train_config.json`；`git`、`data_rev` 是训练用的代码提交和训练集 revision（2026-11-14 之前训练的模型里，只填了核对过的，其余留空）；
`eval` 是它的各个评测集：`名字@revision 成功数/试次数`。带置信区间的总分用 `../eval-scoring/report.py` 或网页工具看。

## 记忆

```bash
bash docs/workflows/sync/memory.sh
```

把笔记本和实验室电脑的记忆目录互相补齐：同名文件留较新的那份，`MEMORY.md` 的索引行取并集。

2026-11-14 建立时的状态：`main` 取代了原来的 `diffusion-flow-react` 分支；四台服务器从拷贝改成了 git 仓库；
`v6-handover-curated-stagelabel`（`1b87da63` = 11-03 18:35 的标签，`2eb9ebda` = 19:00 的标签）、`-stagepred`（`f1a4ffed`）、
`-progpred`（`b59eb468`）、`-cumprog`（`7c8ed927`）上传到了 HF，各机器上的训练副本和这些 revision 逐字节一致。
