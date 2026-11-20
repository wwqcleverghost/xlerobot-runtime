# v9：用诚实的进度，从机器人自己的评测试次里学

English: [README.md](README.md)

**一句话。** v6 系列模型现在几乎所有失败都是同一个样子：A 臂停在胶带旁边，夹爪张开，始终不合（A对位）。把进度喂给模型没有改变这一点，
因为动作根本不看进度输入。v9 让进度只作为模型**预测**的输出；给 216 个实机评测试次的每一帧标上诚实的进度（机器人原地悬停时进度不涨），
用这些试次教模型在那里**不该**做什么。推理时可以用模型自己预测的进度在多个采样里挑，也可以在卡住时先退回去再重新对位。

## 为什么（v9 之前已测到的）

1. **动作不看进度输入**（2026-11-16 干预测试）：输入 s 平移 ±2 个阶段，关节指令只变 0.1–0.6°，不超过换一次噪声；把 s 推过 A对位，夹爪也不合。
2. **模型卡住的地方，正是它几乎采不出抓取的地方**（2026-11-20，笔记本）：在失败试次悬停的帧上（关节 1 秒内不动、A 夹爪张开），
   32 个采样动作块里让 A 夹爪合上的比例：

   | 卡住帧来自 \ 模型 | v6 基线 | progdelta | progself |
   |---|---|---|---|
   | 基线的失败（74 帧） | **0.027** | 0.000 | 0.000 |
   | progdelta 的失败（80） | 0.383 | **0.090** | 0.059 |
   | progself 的失败（100） | 0.098 | 0.135 | **0.004** |

   在自己卡住的地方（对角线）每个模型只有 0–9% 的样本会合爪，换一个模型在同样的帧上却常常会合。
   所以同一个模型换噪声重采样帮助有限，必须改变模型在这些状态下学到的东西。
3. **进度维度和动作是联动的**：同一帧里，自己预测的进度涨得多的样本，正是合爪的样本（与夹爪指令的相关系数 −0.28 到 −0.44）；
   用它在 32 个里挑一个，合爪比例翻倍（在 progdelta 的卡住帧上，progdelta 0.09 → 0.175，progself 0.059 → 0.125）。代价小，但受第 2 点限制。

## 想法 → v9 的组成部分

| 想法（你提的） | 相关工作 | v9 里对应的部分 |
|---|---|---|
| 进度应该隐式，不做输入 | copycat / 因果混淆（Wen 等 2020） | 进度 = 动作后面多出的 9 维，只预测（`--policy.progress_pred_dims=9`） |
| 更好的进度定义，卡住时不涨 | ReWiND（倒放视频）、SARM、RedFlow（按进度 + 本体感受把动作块分组） | 每个评测帧的诚实进度：同阶段最近的示范帧的路程进度 |
| 用测试集重标进度（类似 DAgger），再用 RA-BC | π\*0.6 RECAP（价值 = 剩余步数，失败很负）、SARM RA-BC | 每帧的优势 = 1 秒内涨的进度对比示范的速度；`v9...rabc` = 用诚实进度做 RA-BC |
| 学失败的进度，不学失败的动作 | AFIL、CCDP（失败当负向引导） | `v9...negmask`：失败帧的关节损失屏蔽，进度维照学 |
| 进度的正负能控制动作 | RECAP / CFGRL（优势条件策略 + 无分类器引导） | `v9...adv`：好 / 坏指示作为输入，30% 丢弃；采样 `v_无 + w (v_好 − v_无)` |
| 一个监督者发现卡住就退回去重新对位 | Sentinel（任务进展失败）、Rewind-IL（检测后回到检查点）、RaC（先恢复再纠正） | `STALL=rewind:<秒>`：<秒> 内既不动、进度也不涨，就把卡住前发出的指令倒着重放，再重新规划 |
| 重采样能过去 | V-GPS、RoboMonkey（多采样，用验证器挑） | `--policy.best_of_n=N`：留下自己预测的进度涨得最多的动作块 |

学出来的后退（`advback`）：每条示范的前往段（A前往 + A对位）倒着放，从抓取前的姿态回到起点，作为一个 episode，指示为"后退"
（帧标签 adv = 2），进度下降。录制的动作比从臂领先约 250 ms，倒放后反而会落后；所以这些帧的动作改用倒放时间里 8 帧之后的状态
（`build_v9.py --label_back`、`--finish --back`；58 段，10,709 帧）。推理时 `STALL=learned:<秒>` 在卡住后以"后退"规划 1.5 秒，
再换回"好"。脚本式的 `STALL=rewind` 不需要新模型，可以先验证同一个想法。

屏蔽有一个局限：flow matching 是把关节和进度一起去噪的，被屏蔽帧的带噪关节仍然进入网络输入，仍然影响它的进度输出。
屏蔽去掉的是关节的学习目标，不是失败动作的全部影响。

## 训练集 `v9-handover-curated-rollouts`

[`examples/xlerobot/build_v9.py`](../../../examples/xlerobot/build_v9.py)，在 server2 上跑（步骤见脚本开头）。来源（HF revision 记在数据集的
`v9_sources.json`）：`v6-handover-curated` 的 58 条示范，加 11 个 v6 系列评测集（`eval-v6-flow_h16_s8`、`_progdelta`、`_progdelta_rabc`、
`_progpred`、`_progself`、`_progself_s2head`、`_progself_s2mix35`、`_rabc`、`_stagelabel`、`_stagelabel_stagehead`、`_stagepred`）。

每一帧：

- **阶段**：fine9（示范带手改的起点）。失败试次如果 `dataset_labels.json` 里的备注写了 `<阶段>失败`，阶段不超过它
  （规则会把评审判为 A对位失败 的帧看成 A抓住）。
- **p**（阶段内进度）：示范 = 路程进度（`stage_progress_cum.py`）；评测试次 = 同阶段最近 10 个示范帧的路程进度平均
  （关节 / 示范标准差，A 臂在前，5 帧平滑）。悬停的手臂一直像同一批示范帧，进度就是平的。
- **v** = 阶段 + p（0..9）。**gain** = 之后 30 帧内最好的 v − v。
- **adv** = 1：gain ≥ 示范在该阶段 gain 的 30% 分位数（RECAP 的阈值）；否则 0。**neg** = adv 为 0 的评测帧。

评测试次保留的帧：从开始到任务结束（`episode_end.csv`）；adv 为 0 的连续段超过 150 帧时，只保留前 150 帧和最后 15 帧
（失败结尾的停滞只留前 150 帧）。每个评测集每第 5 个试次（`episode % 5 == 4`，共 54 个）留出不训练，用于离线检查（标签在
`v9_labels_holdout.npz`）。

数据集里除 LeRobot 数据外的文件：`frame_labels.parquet`（index、episode、source、source_episode、demo、success、v、adv、neg），
`sarm_progress.parquet`（v / 9，给 RA-BC），`episodes_demo.json` / `episodes_succ.json`（子集），`v9_spec.json`、`v9_summary.txt`、
`v9_holdout.json`、`v9_labels_holdout.npz`、`v9_sources.json`。动作 = 12 个关节 + 9 个进度维（恒等归一化，同 `-cumprog`）；
没有 `observation.environment_state`。

数字（2026-11-20，`v9_summary.txt`）：274 个 episode（58 条示范，216 个评测试次，其中 127 个成功），124,773 帧，neg 占 32%。
各阶段 gain 阈值（30 帧内涨多少个阶段）：0.27、0.12、0.47、0.32、0.17、0.67、0.48、0.37、0.06。评测试次和所有评测录像一样，
实际以约 25 fps 录制、却标成 30 fps；所以模型从中学到的动作块，和它执行时覆盖的真实时间相同。

## 模型（队列任务，`queue_v3.sh`）

全部：FLOW h16 s8，140K 步，batch 24，从头训练，预测进度维（`progress_pred_dims=9`）。

| 任务 | 模型 | 数据 | 检验什么 |
|---|---|---|---|
| `flow:v9prog` | `v9-flow_h16_s8_prog` | 示范 + 成功的评测试次 | 过滤式 BC：只加自己成功的试次有没有用 |
| `flow:v9adv` | `v9-flow_h16_s8_adv` | 全部 | 优势条件 + 引导（RECAP / CFGRL） |
| `flow:v9negmask` | `v9-flow_h16_s8_negmask` | 全部 | 失败只教进度（你的屏蔽想法） |
| `flow:v9rabc` | `v9-flow_h16_s8_rabc` | 示范 + 成功的评测试次 | 用诚实进度做 RA-BC（往后看 90 帧，κ 0.15） |
| `flow:v9advdemo` | `v9-flow_h16_s8_adv_demo` | 只有示范 | 不加评测试次的优势条件：评测试次本身重不重要 |
| `flow:v9advr2` | `v9-flow_h16_s8_adv_r2` | 全部 | `adv` 换一个随机种子：离线数字单靠偶然会差多少 |
| `flow:v9advback` | `v9-flow_h16_s8_advback` | 全部 + 58 段倒放的前往（`-back` 数据集） | `adv` 加第三种指示"后退"：学出来的后退 |

参照：`v6-flow_h16_s8`（同样的示范，没有评测试次，没有进度）。

## 推理选项（实机评测，`lerobot_record_eval.py`）

- `--policy.adv_guidance=<w>`（adv 模型）：1 = 以"好"为条件；1.5–3 离坏动作块更远。
- `--policy.best_of_n=<N>`（带进度维的模型，包括 v6 的进度模型）：每次规划采 N 个，留自己预测的进度涨得最多的那个。
- `STALL=resample:<秒>` / `STALL=rewind:<秒>`（环境变量，任何模型）/ `STALL=learned:<秒>`（advback）：见 `StallGuard`。`STALL_BOX`（3°）、`STALL_REWIND_S`（1.5 秒）、
  `STALL_MAX`（3 次）可调；触发记录在评测录像的 `stalls.csv`。
- v9 模型打印 `[v9] progress`，并记到 `stages.csv`（pred = 阶段，progress = v / 9）。

## 上实机前的离线检查

[`examples/xlerobot/offline_v9.py`](../../../examples/xlerobot/offline_v9.py) 在 54 个留出试次上：卡住帧上采样动作块合爪的比例（以及按自己的进度
best-of-N），进度相对标签的误差，预测的进度增益区分"在推进"和"没推进"帧的能力（AUC）。adv 模型分别在 w = 1、1.5、2、3 下测。

## 实机评测计划

实机评测才是真正的检验（每个 30 次，胶带位置尽量相同）。按离线卡住帧合爪比例排序；先测最好的 v9 模型（用它最好的 w）对比
`v6-flow_h16_s8`，再加 `STALL=rewind:4`。

## 结果

离线和实机结果出来后补在这里。

## 相关工作

- π\*0.6 / RECAP：[arXiv 2511.14759](https://arxiv.org/abs/2511.14759) — CFGRL：[arXiv 2505.23458](https://arxiv.org/abs/2505.23458)
- AFIL（失败试次当负向引导）：[arXiv 2605.08434](https://arxiv.org/abs/2605.08434) — CCDP：[arXiv 2503.15386](https://arxiv.org/abs/2503.15386) — RedFlow：[arXiv 2607.27782](https://arxiv.org/abs/2607.27782)
- RaC（恢复 + 纠正数据）：[arXiv 2509.07953](https://arxiv.org/abs/2509.07953) — Rewind-IL：[arXiv 2604.16683](https://arxiv.org/abs/2604.16683)
- V-GPS：[arXiv 2410.13816](https://arxiv.org/abs/2410.13816) — RoboMonkey：[arXiv 2506.17811](https://arxiv.org/abs/2506.17811)
- ReWiND：[arXiv 2505.10911](https://arxiv.org/abs/2505.10911) — SARM：[arXiv 2509.25358](https://arxiv.org/abs/2509.25358) — Sentinel：[arXiv 2410.04640](https://arxiv.org/abs/2410.04640)
- Copycat agents：[arXiv 2010.14876](https://arxiv.org/abs/2010.14876)
