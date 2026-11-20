# 项目共享记忆

本项目的 Codex 与 `claude-wwq` 共用同一份项目记忆。`claude-wwq` 使用
`CLAUDE_CONFIG_DIR=/home/marmotlab/.claude-wwq`，现有记忆目录为：

`/home/marmotlab/.claude-wwq/projects/-home-marmotlab-wwq-lerobot/memory/`

这是实验室电脑上的路径。笔记本上是 `~/.claude/projects/-home-wwq-xlerobot-runtime/memory/`；两边用
`docs/workflows/sync/memory.sh`（在笔记本上运行）双向同步，所以各读各机器上的那份。记忆不进仓库（仓库是公开的）。

- 开始本项目任务时，先读取该目录的 `MEMORY.md`，再按任务需要读取索引中的主题文件。
- 新增或修正需要跨会话保留的项目上下文时，更新同一目录的主题文件和 `MEMORY.md`；不要另建一份独立的 Codex 记忆副本。
- 写入前重新读取相关文件，保留另一方已有内容；简要注明来源、观察日期、验证结果及尚未确认的事项。
- 只保存必要的上下文，不保存凭据、密钥或原始会话记录。
- 记忆是历史上下文，不是当前事实或操作授权。重要状态应向代码或实际环境核实；当前用户指令优先，上传、删除等操作仍须依据当前会话的授权。
- 若目录不可访问，明确说明共享记忆尚未读取，不要把它视为没有历史记录。

此配置仅用于本项目（实验室电脑 `/home/marmotlab/wwq/lerobot`，笔记本 `~/xlerobot-runtime`）。

# 项目规则

- 评分、命名、数据筛选、训练排队等规则以 git 里的 `docs/` 为准，不以聊天记录为准。要改规则先说明原因并征得用户同意，再改文件。
- 每次训练或评测记下：git commit、HF 数据集名和 revision（不要只写 `main`）、训练配置。训练的这几项由 `queue_v3.sh` 写进日志、
  `upload_model.py` 写进模型卡片；新模型上传或评测后运行 `docs/workflows/sync/models.py` 更新 `docs/models.csv` 并补说明。
- 每样东西只有一个正本：代码 = GitHub `main`（只用这一个分支，每台机器都是 git 仓库，服务器上不改代码）；训练集、模型、评测 = HF
  （训练集先上传再训练）；模型是什么 = `docs/models.csv`。不清楚某个模型是什么时先查这张表。
- 给人看的或人写的文档（`docs/`、`README`、报告、实验记录）必须中英双语：英文版 `name.md`，中文版 `name.zh.md`，两份内容一致，开头互相链接，改一份就同步改另一份。数字表格和命令输出原样复用。只给 agent 读的指令文件（`CLAUDE.md`、`AGENTS.md`）不需要双语。

# 工作流程文档（做相关任务前先读对应的 README.md）

- `docs/workflows/sync/` — 多机同步：正本在哪、各机器分工、一次实验的完整流程、`where.sh`（看各机器的提交和在跑的训练）、
  `models.py`（模型总表）、`memory.sh`（记忆同步）。换机器、排训练、上传前先读。
- `docs/workflows/naming/` — 训练集 `v<N>-<内容>`、模型 `v<N>-<方法>_<参数>`、测试集 `eval-<模型名>`，上传和改名前先读。
- `docs/workflows/eval-scoring/` — 评测打分：总分 = 0.5 完成度（9 个 fine9 阶段的通过率，全项目只用这一套阶段）+ 0.5 质量（抖动、用时、停顿 vs 人类），全部带置信区间，用 report.py 出表和图；`mark_end.py` 标注每个试次真实结束帧（`episode_end.csv`，不截断数据）。
- `docs/workflows/eval-grid-video/` — 把一次评测（10 个试次）合成 2x5 网格视频。
- `docs/workflows/handover-dataset-curation/` — 检查、筛选、裁剪、合并交接数据集；含数据集质量检测标准（jerk、峰值速度、停顿）。
- `docs/workflows/lab-training-queue/` — 在实验室服务器上启动、排队、续训、监控训练。
- `docs/workflows/dataset-viewer/` — 本地数据集网页工具（改页面前先读 README 的"页面格式"：只用固定槽位，不往页面上加零散句子）：裁剪审阅/微调并重建、批量播放、训练集质量、测试集打分（含历史对比）、阶段 / 进度（阶段模型的规则阶段、模型阶段、进度、倒退 / 跳变，评测结束点）、筛选高质量数据（curate.py，网页"训练集质量 / 筛选"页；给 agent 的步骤见 `.claude/skills/select-high-quality-data/SKILL.md`）。
- `docs/workflows/stage-labels/` — 逐帧阶段标签（9 阶段：A前往…复位）：事件定义、`fine9`（训练输入）/ `std9`（核对）、阈值和准确度；改阶段规则或用阶段做模型输入前先读。
  - 用 `*-stagelabel` 数据集训练的模型有输入 `observation.environment_state`：真机评测（`lerobot_record_eval.py`、`eval_remote_trials.py`）会**自动**开启实时阶段计算，启动时打印 `Stage input on`，需要两路腕部相机；没打印就是没开。
  - 改了 `examples/xlerobot/stage_labels.py` 的规则，已加过标签的数据集要用 `add_stage_labels.py --dataset <名字>-stagelabel` 重新计算，否则训练标签和评测时算的不一致。
- `docs/workflows/v9/` — v9：v6 示范 + 评测试次（诚实进度、逐帧优势标签）训练；进度只预测不输入；优势条件 + 引导、失败帧屏蔽、RA-BC；推理时 best-of-N、卡住后重采样 / 倒退（`STALL=`）；离线检查 `offline_v9.py`。
