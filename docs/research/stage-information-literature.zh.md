# 视觉运动策略中的阶段信息：文献检索

English: [stage-information-literature.md](stage-information-literature.md)

检索日期：2026-11-06。目的：针对 XLeRobot 交接项目的六个研究点（9 阶段规则见
[`examples/xlerobot/stage_labels.py`](../../examples/xlerobot/stage_labels.py)；v6 变体 v6stage、v6stagepred、v6progpred、
v6rabc、v6stagehead，对比基线 v6_flow_h16_s8），逐项找最接近的已有工作，说明相同处和不同处，判断该点是**已有 / 部分已有 / 可能是新的**。

核实方式：先 WebSearch，再对关键论文打开 arXiv 摘要页（标注处还读了 HTML 全文的相关段落）。
**✓** = 在 arXiv 上读过（摘要，或引用的 HTML 段落）。**（未核实）** = 只在搜索摘要或二手总结里看到；标为未核实的数字，
不读原文不要引用。"没找到"指本次检索没找到，不代表一定没有。

## 总结

| # | 研究点 | 结论 |
|---|---|---|
| 1 | 基于事件规则、因果地逐帧打阶段标签，不用人工 / 语言标注 | **部分已有**。从夹爪事件自动得到阶段标签已有（StaKe、MimicGen 式终止信号）；训练和真机实时**同一份代码**、用多种信号的事件、面向双臂交接的因果标注器，没找到 |
| 2 | 阶段作为策略**输入**在闭环中使用，对比辅助目标、RA-BC 加权、学出来的记忆 | 每种用法单独都**已有**（PHASER、ABC、MS-Bot / StaKe、SADP / SARM / CAMP、TF-DP）。**同一数据、同一机器人上把四种放在一起比较**没找到 → **可能是新的** |
| 3 | 闭环阶段误判的失败模式和修复（去抖、允许回退、HMM/HSMM 前向滤波） | 修复技术在其他领域**已有**（手术阶段识别、语音、在线动作分割、RL 切换）。对视觉运动策略中"策略 ↔ 阶段估计"反馈环的分析，以及用 HSMM 滤波给 diffusion 策略提供阶段：没找到 → **可能是新的** |
| 4 | 按阶段门控各路输入（相机 / 关节特征乘门控值、FiLM、适配器） | **已有**（MS-Bot，CoRL 2024 oral；PHASER 的 FiLM；Selective Perception 相机路由；MoE-DP） |
| 5 | 少人工的阶段标注（规则 + 分类器、少量标注 + DTW/TCC、VLM、主动学习） | 各部分都**已有**；把它们组合起来给机器人阶段打标签属于工程，不算研究贡献 |
| 6 | 预测的阶段回喂当输入 → 照抄 / 误差累积 | 一般机制**已有**（copycat 问题、因果混淆、手术阶段误差级联）。目前没有回喂预测的变体（v6stagehead 的预测头只用于记录），对照实验在**计划中**；针对策略**自己**预测的阶段、带消融的展示没找到 → **作为分析可能是新的** |

**关于"我是不是在复刻 SARM"**：SARM（[2509.25358](https://arxiv.org/abs/2509.25358) ✓）是一个**奖励模型**：阶段分类器 +
阶段内进度回归器，用人工子任务标注训练（按标注协议标；论文 HTML v2 里没有出现 "Qwen"；LeRobot 的 SARM 实现默认用
Qwen3-VL-30B 做标注，见第 5 点），用来给 BC 样本**加权**（RA-BC）。它从不把阶段喂给策略。我们的变体里只有 **v6rabc** 沿用 SARM
（用规则标签代替学出来的奖励模型），属于应用。v6stage / v6stagehead（规则阶段作输入；v6stagehead 另加一个看不到阶段输入、
也不回喂的辅助阶段预测头）更接近 PHASER 和 ABC；v6stagepred /
v6progpred（阶段作为额外的去噪目标）更接近 StaKe / SADP。这些论文都没有把几种用法放在一起比较。

## 1. 因果的、基于规则的逐帧阶段标签

**最接近的工作**

- **StaKe** — *Improving VLA Fine-Tuning with Structured Stage and Keyframe Supervision*，[2606.26801](https://arxiv.org/abs/2606.26801) ✓（读了 HTML）。
  从演示的夹爪状态全自动提取阶段标签和关键帧目标，不需要人工标注。阶段由夹爪开 / 合切换得到（双臂时两只手的切换合并）。
  标签**只作训练时的辅助头**，推理不变。报告的相对提升：双臂仿真 14 %，Franka 真机 56 % ✓（摘要）。
- **MimicGen 子任务终止信号** — [文档](https://mimicgen.github.io/docs/tutorials/subtask_termination_signals.html) ✓。
  每个子任务一个由环境给出（仿真特权状态）或人工标注的 0/1 信号；用来切分源演示做数据生成，不作策略输入。
- 其他自动切分（未核实，搜索摘要）：SparTa（夹爪状态 / 低速）、DynaMimicGen（[2511.16223](https://arxiv.org/abs/2511.16223)，物体状态事件）、
  *The Art of Imitation*（[2407.13432](https://arxiv.org/abs/2407.13432)，速度阈值）、UVD（[2310.08581](https://arxiv.org/abs/2310.08581)，嵌入距离平台，无标签，NeurIPS 2023）。
- **PHASER**（[2605.29407](https://arxiv.org/abs/2605.29407) ✓ HTML）：遥操作时**操作员按手柄按钮**标阶段切换。
- **MS-Bot**（[2408.01366](https://arxiv.org/abs/2408.01366) ✓ HTML）：每个样本加阶段标签；阶段由人划分，作者把用 LLM 自动标注列为未来工作（后半句来自搜索总结，未核实）。

**和我们的区别**：StaKe 和 MimicGen 式标签只用一种信号（夹爪）或特权状态，对整条轨迹离线计算。我们的：(a) 严格**因果**——每个事件只用
当前和过去帧以及**上一帧**指令，所以训练标签和评测时的实时阶段由**同一个函数**算出；(b) 多信号事件（正运动学算出的指尖沿夹爪轴的速度、
腕部相机里胶带直径、另一只夹爪进入 A 腕部相机视锥、回到起始姿态的距离）；(c) 双臂交接中 A/B 角色在线判定；(d) 掉落时显式回退。

**结论：部分已有。**"从夹爪事件自动得到阶段标签、不用人工"已有（StaKe）。新的部分很窄：训练 / 部署一致、可作为实时策略输入的因果标注。
单独看是工程贡献；要和第 2 或第 3 点一起（例如证明标签和实时不一致会掉分，或因果性有影响）才算研究点。

## 2. 阶段作为策略输入，对比其他用法

**阶段作输入，闭环**

- **PHASER** — [2605.29407](https://arxiv.org/abs/2605.29407) ✓（HTML，IEEE/ASME T-Mech）。用 FiLM 按阶段调制 ACT；另一个多模态阶段预测器
  （视觉 + 力 + 位姿，冻结策略主干 + MLP）实时估计阶段，输出回喂给策略。消融：FiLM 优于 token 级条件和无条件 ✓。挂衣成功率 56 % → 87 %（含自动恢复）✓。
  论证：没有阶段时策略学到的是混合分布，在外观相似的状态上相互竞争的模式会产生平均或来回摆动的动作 ✓。
- **ABC** — *Scalable Behavior Cloning with Open Data, Training, and Evaluation*，[2606.27375](https://arxiv.org/abs/2606.27375) ✓（HTML 附录 H.3）。
  一个"受 SARM 阶段奖励模型启发"的小子任务分类器，推理时给 DiT 策略设定子任务提示。T 恤从半叠状态开始：不加提示时 10 次里 5 次重新铺平，
  加提示后没有出现（1 次因抓空后恢复失败超时）✓。训练时以 0.2 的概率加子任务条件 ✓。和 SARM 是同一批作者（Wu、Shentu、Yu、Abbeel）。
- **MS-Bot** — [2408.01366](https://arxiv.org/abs/2408.01366) ✓（HTML，CoRL 2024 oral）。阶段在策略**内部**预测（状态 token 上的门控 MLP、
  软阶段 token、在阶段边界附近放宽的损失），用来驱动多模态融合。
- **SADP** — [2605.16871](https://arxiv.org/abs/2605.16871) ✓（RA-L 2026）。diffusion 策略以自然语言子目标为条件（标签由基础模型生成），
  加一个辅助的"继续分数"头，驱动在线子目标切换。
- **BEACON** — [2609.22730](https://arxiv.org/abs/2609.22730) ✓。diffusion 策略以隐状态的贝叶斯信念（估计 + 不确定度）为条件。

**阶段作辅助目标**

- **StaKe**（见上）✓：阶段分类头 + 关键帧头，和动作头分开。
- 多粒度语言引导模仿学习，[2609.37135](https://arxiv.org/abs/2609.37135) ✓（摘要）；"子任务感知损失"的细节来自搜索摘要（未核实）。
- PALM，[2601.07060](https://arxiv.org/abs/2601.07060)（未核实）：预测阶段 + 阶段内进度。

**阶段 / 进度作训练权重**：**SARM + RA-BC** [2509.25358](https://arxiv.org/abs/2509.25358) ✓（HTML）。进度标签按各标注子任务在数据集中的平均时间占比给出，
子任务内线性插值 ✓；权重由一个动作块前后的进度差得到 ✓。相关：π\*0.6 / RECAP [2511.14759](https://arxiv.org/abs/2511.14759) 把二值化的优势作为策略输入，而不是加权（未核实，二手来源）。

**学出来的记忆 / 历史**：**CAMP** [2606.21188](https://arxiv.org/abs/2606.21188) ✓（压缩的动作记忆，无需额外监督即可隐式跟踪任务进度）；
**TF-DP** [2602.07388](https://arxiv.org/abs/2602.07388) ✓（把执行轨迹投影到图像中，提供"阶段感知的上下文"）；
DSSP [2605.14598](https://arxiv.org/abs/2605.14598)、SeedPolicy [2603.05117](https://arxiv.org/abs/2603.05117)（未核实）。

**和我们的区别**：上面每篇都只测**一种**用法，对比无条件基线。没找到在同一数据、同一机器人、同一打分下比较 *输入（v6stage）*、
*与动作一起去噪的辅助目标（v6stagepred / v6progpred）*、*RA-BC 加权（v6rabc）*、*规则输入 + 看不到输入的辅助预测头（v6stagehead）* 和 *学出来的记忆* 的工作。
把阶段 one-hot **拼进 diffusion/flow 的目标向量**（和动作一起去噪）而不是单独的头，也没见到（StaKe、SADP、MS-Bot 都用单独的头）。

**结论**：「阶段作为输入在闭环中有用」**已经有人证明**（PHASER、ABC），不要当作新点。**系统比较**可能是新的，是最有希望的贡献，
前提是控制种子 / 步数一致，并用评测打分的置信区间。

## 3. 闭环阶段误判的失败模式和修复

**最接近的工作**

- **在线手术阶段识别** — *Stabilizing Temporal Inference Dynamics for Online Surgical Phase Recognition*，
  [2605.16387](https://arxiv.org/abs/2605.16387) ✓。早期误判会向后传播形成"误差级联"；无记忆的逐帧决策不稳定；提出 *Evidence-Gated Transition Predictor*，
  累积证据超过阈值才允许切换阶段（≈ 滞回 / 去抖）。更早的：在线阶段识别用从左到右的 HMM（Blum 等，MICCAI 2008，
  [pdf](https://www.campar.in.tum.de/pub/blum2008miccai/blum2008miccai.pdf)，未核实）；TeCNO 因果 TCN（[2003.10751](https://arxiv.org/abs/2003.10751)，细节未核实）。
- **在线时序动作分割**：OnlineTAS（[2411.01122](https://arxiv.org/abs/2411.01122)，NeurIPS 2024）和 O-TALC（[2404.06894](https://arxiv.org/abs/2404.06894)）——针对流式预测过度分割的后处理（细节未核实）。
- **语音识别 hybrid NN/HMM**（Bourlard & Morgan，*Connectionist Speech Recognition*，1994）：分类器后验除以类先验 = 缩放似然，作为 HMM 发射概率
  （[爱丁堡 ASR 讲义](https://www.inf.ed.ac.uk/teaching/courses/asr/2024-25/asr10-hybrid_hmm_nn.pdf)）。这正是"分类器输出当发射概率、HMM 前向滤波"的做法。
- **机器人技能的 HSMM**：GP-HSMM 切分（Nakamura 等，2017，[PMC5742615](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5742615/)，批处理，未核实）；任务参数化 HSMM
  （Rozo 等，[pdf](https://lcas.lincoln.ac.uk/wp/wp-content/uploads/2019/07/Rozoetal_RSS_TIG.pdf)，未核实）；交互 ProMP 的在线相位估计（Maeda 等，IJRR，[pdf](https://www.ias.tu-darmstadt.de/uploads/Team/PubGJMaeda/phase_estim_IJRR.pdf)，未核实）。
- **RL 策略的滞回切换**防止抖振（RLJ 2024，[par.nsf.gov](https://par.nsf.gov//servlets/purl/10541296)，标题和细节未核实）。
- **步态相位估计**（外骨骼）：有限状态机 vs 神经网络，状态机切换时的不连续（[2107.03746](https://arxiv.org/abs/2107.03746)，未核实）。
- **机器人模仿学习**：PHASER（无阶段时在相似状态上摆动 ✓；它的预测器会切到恢复阶段再切回 ✓）；SADP 用继续分数切换 ✓；
  StreamVLA [2602.01100](https://arxiv.org/abs/2602.01100) ✓ 的 "Lock-and-Gated"——检测到切换前子任务锁定，动机是"目标不稳定"；ABC 重做已完成的子任务 ✓。

**和我们的区别**：这些论文把阶段估计当作开环识别问题（手术、语音、视频），或只说明阶段预测器有用。我们遇到的是**反馈环**：规则过早判"已夹住"
→ 以阶段为条件的策略执行下一阶段动作 → 观测对不上 → 阶段跳回 → 来回摆动；以及抓取前悬停。没找到分析这种"策略 ↔ 估计器"相互作用，
或把 HSMM 滤波后的阶段（分类器输出当发射概率）提供给 diffusion/flow 策略的工作。

**结论**：去抖、滞回、允许回退、HMM/HSMM 前向滤波**都是标准做法**——引用，不要当贡献。**闭环失败分析**以及**在真机上测哪种修复对以阶段为条件的
视觉运动策略有效**，可能是新的。

## 4. 按阶段门控输入

**最接近的工作**

- **MS-Bot** — [2408.01366](https://arxiv.org/abs/2408.01366) ✓（HTML）。"阶段引导的动态多感官融合"：注入了阶段信息的状态 token 作为 query，
  对视觉 / 听觉 / 触觉特征做交叉注意力，模态权重随预测阶段变化。基本就是第 4 点。
- **Selective Perception for Robot** — [2602.15543](https://arxiv.org/abs/2602.15543) ✓。轻量路由器读腕部相机 + 提示词，预测每路相机的任务相关性，
  削弱用处小的视角；路由器标签由 VLM 流程生成。
- **PHASER** ✓：按阶段用 FiLM 调制编码器。
- **MoE-DP** — [2511.05007](https://arxiv.org/abs/2511.05007) ✓。在视觉编码器和 diffusion 头之间加 MoE 层；专家"动态激活以处理不同阶段"，可解释为任务基元。
- TacFiLM（[2603.14604](https://arxiv.org/abs/2603.14604)，未核实）、diffusion 策略中的 FiLM（综述 [2608.07558](https://arxiv.org/abs/2608.07558)，未核实）。

**和我们的区别**：我们的版本是用**规则**阶段（不是学出来的阶段）门控交接任务的相机 / 关节特征（腕部 vs 头部相机），属于 MS-Bot / PHASER 的变体。

**结论：已有。**只能报告新的实验发现（例如交接各阶段哪路相机重要、门控能否解决开头 复位 的混淆）；机制本身不新。

## 5. 少人工的阶段标注

- **弱监督**：标注函数 → 概率标签 → 最终模型（Snorkel，Ratner 等，[1711.10160](https://arxiv.org/abs/1711.10160)）；直接在时间序列上写启发式规则
  （ECG，[PMC8861672](https://pmc.ncbi.nlm.nih.gov/articles/PMC8861672)，未核实）；从不精确时间戳学切分（Adams & Marlin，UAI 2018，未核实）。
- **少量标注 + 对齐**：TCC（[1904.07846](https://arxiv.org/abs/1904.07846)，CVPR 2019）——最少只需一条标注视频即可做阶段分类
  （[Google 博客](https://research.google/blog/video-understanding-using-temporal-cycle-consistency-learning/)）；演示的 DTW 对齐（事件感知 DTW，米兰理工论文，未核实）；ProMP 相位估计（Maeda，见上）。
- **VLM 自动标注**：LeRobot 的 SARM 实现——子任务标注步骤 `--model` 默认 `Qwen/Qwen3-VL-30B-A3B-Instruct`
  （[LeRobot SARM 文档](https://huggingface.co/docs/lerobot/sarm)，来自搜索总结；SARM 论文本身描述的是人工标注协议 ✓）；
  StreamVLA ✓（Qwen3-VL-plus 提出子阶段和时间戳，人工核对边界——这一细节来自搜索摘要，未核实）；
  NILS（[2410.17772](https://arxiv.org/abs/2410.17772)，未核实）；*Actions as Language*（[2509.22195](https://arxiv.org/abs/2509.22195)，Gemini 标注，未核实）；拆解任务的 VLM 对比
  （[2609.17714](https://arxiv.org/abs/2609.17714)，未核实）。反复出现的结论：VLM 给的边界不准。
- **只复核分歧 / 边界的主动学习**：Su & Elhamifar，时序动作分割的主动学习，ECCV 2024（未核实）；B-ACT 以边界为中心的主动学习（[2604.15173](https://arxiv.org/abs/2604.15173)，未核实）。

**结论：已有**（各部分）。"因果规则当标注函数 → 分类器 → 人只复核边界附近规则和分类器的分歧"是个合理的工具，不是论文贡献，除非能证明它修好了第 3 点的闭环失败。

## 6. 预测的阶段回喂当输入

- **Copycat 问题** — Wen 等，NeurIPS 2020，[2010.14876](https://arxiv.org/abs/2010.14876) ✓。有历史输入、专家动作在时间上高度相关时，模仿者学会"作弊"，
  去预测专家的上一个动作；修复：对抗地去掉表示里关于上一个动作的多余信息。
- **因果混淆** — de Haan 等，NeurIPS 2019，[1905.11979](https://arxiv.org/abs/1905.11979)。信息更多反而更差，尤其是输入含历史时（来自搜索结果，没有重读）。
- **手术阶段误差级联** — [2605.16387](https://arxiv.org/abs/2605.16387) ✓（早期错误阶段向后传播）。
- **π0.5** — [2504.16054](https://arxiv.org/abs/2504.16054) ✓（HTML）。运行时模型先预测语义子任务，再预测动作；用标注的子任务标签训练。读到的部分没有讨论照抄预测的子任务。
- **ABC** ✓：推理时用外部分类器；训练时只有 20 % 的样本加子任务条件。
- **VISTA** [2602.10983](https://arxiv.org/abs/2602.10983) ✓（摘要）：世界模型 + VLA 的分层框架；搜索摘要说在子任务边界附近随机偏移目标以提高对不对齐的鲁棒性（未核实）。
  **τ0-VLA** [2608.16885](https://arxiv.org/abs/2608.16885) ✓（摘要）：高层根据执行记忆生成子任务；低层用预测 vs 真值子任务训练的细节来自搜索摘要（未核实）。

**我们目前的变体没有把预测的阶段回喂。** v6stagehead（commit d7fb1d86，`src/lerobot/policies/diffusion/modeling_diffusion.py`）=
规则阶段输入 + 辅助预测头：预测头只从关节状态和图像特征预测阶段（看不到阶段输入），交叉熵权重 0.1；阶段输入始终是规则算的 fine9；
预测结果 `stage_head_pred` 只在评测时记录。所以当前训练没有回喂，也就没有 copycat 风险。把预测的阶段回喂当输入（及其 copycat 风险）是**计划中的实验**。

**和我们的区别（针对这个计划中的实验）**：一般风险就是 copycat 问题（阶段在时间上几乎不变，"照抄输入"损失就很低）。
没找到专门针对视觉运动策略的阶段 / 子任务通道展示这一点，或比较各种修复（外部因果标注器、阶段输入 dropout、加噪 / scheduled sampling、预测头和输入之间加 HSMM 滤波）的工作。

**结论**：机制**已知**（引用 copycat 和因果混淆）；**在这个设定下的对照实验**做出来后可能是新的，但分量小——最好放进第 2 / 3 点里讲，不单独作为贡献。

## 能主张的贡献

1. **最强**：在同一数据、真机上对比阶段信息的几种用法（输入 / 与动作一起去噪的目标 / 规则输入 + 辅助预测头 / RA-BC 加权 / 学出来的记忆）——没找到先例。回喂预测阶段是计划中的补充实验，不在当前这批训练里。
2. **其次**：以阶段为条件的策略的闭环失败分析（过早判"已夹住"→ 来回摆动、悬停）和修复评测（去抖 / 滞回、允许回退、对分类器做 HSMM 滤波）——技术引自手术 / 语音，反馈环分析是新的。
3. **辅助**：双臂交接的因果、训练 / 部署一致的规则标注器（工程；最接近的是 StaKe）。
4. **不要当成新点**：阶段作输入有用（PHASER、ABC）；阶段辅助头（StaKe、SADP）；RA-BC（SARM）；按阶段门控融合（MS-Bot）；VLM / TCC / 主动学习标注。

## 建议引用

| 论文 | 用途 | 核实 |
|---|---|---|
| SARM, Chen et al., ICLR 2026, [2509.25358](https://arxiv.org/abs/2509.25358) | stage + progress reward, RA-BC | ✓ HTML |
| PHASER, Chen et al., T-Mech, [2605.29407](https://arxiv.org/abs/2605.29407) | phase as FiLM input, closed-loop predictor | ✓ HTML |
| ABC, Allshire et al., [2606.27375](https://arxiv.org/abs/2606.27375) | SARM-style subtask classifier → subtask prompt | ✓ HTML App. H |
| StaKe, [2606.26801](https://arxiv.org/abs/2606.26801) | automatic gripper-based stage labels, auxiliary heads | ✓ HTML |
| MS-Bot, Feng et al., CoRL 2024, [2408.01366](https://arxiv.org/abs/2408.01366) | stage-guided modality fusion | ✓ HTML |
| SADP, Hu & Horii, RA-L 2026, [2605.16871](https://arxiv.org/abs/2605.16871) | subgoal-conditioned DP + continuation head | ✓ abstract |
| CAMP, Wang et al., [2606.21188](https://arxiv.org/abs/2606.21188) | learned action memory | ✓ abstract |
| TF-DP, [2602.07388](https://arxiv.org/abs/2602.07388) | execution-trace conditioning, stage ambiguity | ✓ abstract |
| BEACON, [2609.22730](https://arxiv.org/abs/2609.22730) | belief-conditioned diffusion policy | ✓ abstract |
| MoE-DP, [2511.05007](https://arxiv.org/abs/2511.05007) | phase experts in DP | ✓ abstract |
| Selective Perception, [2602.15543](https://arxiv.org/abs/2602.15543) | camera-view router | ✓ abstract |
| StreamVLA, [2602.01100](https://arxiv.org/abs/2602.01100) | locked subtask, goal instability | ✓ abstract |
| π0.5, [2504.16054](https://arxiv.org/abs/2504.16054) | predicted subtask → action | ✓ HTML |
| Copycat, Wen et al., NeurIPS 2020, [2010.14876](https://arxiv.org/abs/2010.14876) | copying of fed-back inputs | ✓ abstract |
| Causal confusion, de Haan et al., NeurIPS 2019, [1905.11979](https://arxiv.org/abs/1905.11979) | history inputs hurt | search only |
| Online SPR stabilization, [2605.16387](https://arxiv.org/abs/2605.16387) | error cascades, evidence-gated transitions | ✓ abstract |
| Bourlard & Morgan 1994, *Connectionist Speech Recognition* | hybrid NN/HMM emissions | secondary |
| TCC, Dwibedi et al., CVPR 2019, [1904.07846](https://arxiv.org/abs/1904.07846) | few-label phase alignment | secondary |
| UVD, Zhang et al., NeurIPS 2023, [2310.08581](https://arxiv.org/abs/2310.08581) | label-free subgoal discovery | search only |
| OnlineTAS, NeurIPS 2024, [2411.01122](https://arxiv.org/abs/2411.01122) | streaming segmentation flicker | search only |
| VISTA [2602.10983](https://arxiv.org/abs/2602.10983), τ0-VLA [2608.16885](https://arxiv.org/abs/2608.16885) | hierarchical subtask robustness | abstract only |
