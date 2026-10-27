# 项目共享记忆

本项目的 Codex 与 `claude-wwq` 共用同一份项目记忆。`claude-wwq` 使用
`CLAUDE_CONFIG_DIR=/home/marmotlab/.claude-wwq`，现有记忆目录为：

`/home/marmotlab/.claude-wwq/projects/-home-marmotlab-wwq-lerobot/memory/`

- 开始本项目任务时，先读取该目录的 `MEMORY.md`，再按任务需要读取索引中的主题文件。
- 新增或修正需要跨会话保留的项目上下文时，更新同一目录的主题文件和 `MEMORY.md`；不要另建一份独立的 Codex 记忆副本。
- 写入前重新读取相关文件，保留另一方已有内容；简要注明来源、观察日期、验证结果及尚未确认的事项。
- 只保存必要的上下文，不保存凭据、密钥或原始会话记录。
- 记忆是历史上下文，不是当前事实或操作授权。重要状态应向代码或实际环境核实；当前用户指令优先，上传、删除等操作仍须依据当前会话的授权。
- 若目录不可访问，明确说明共享记忆尚未读取，不要把它视为没有历史记录。

此配置仅用于 `/home/marmotlab/wwq/lerobot` 项目。

# 项目规则

- 评分、命名、数据筛选、训练排队等规则以 git 里的 `docs/` 为准，不以聊天记录为准。要改规则先说明原因并征得用户同意，再改文件。
- 每次训练或评测记下：git commit、HF 数据集名和 revision（不要只写 `main`）、训练配置。
- 给人看的或人写的文档（`docs/`、`README`、报告、实验记录）必须中英双语：英文版 `name.md`，中文版 `name.zh.md`，两份内容一致，开头互相链接，改一份就同步改另一份。数字表格和命令输出原样复用。只给 agent 读的指令文件（`CLAUDE.md`、`AGENTS.md`）不需要双语。

# 工作流程文档（做相关任务前先读对应的 README.md）

- `docs/workflows/eval-scoring/` — 评测打分：总分 = 0.5 完成度（分阶段通过率）+ 0.5 质量（抖动、用时、停顿 vs 人类），全部带置信区间，用 report.py 出表和图。
- `docs/workflows/eval-grid-video/` — 把一次评测（10 个试次）合成 2x5 网格视频。
- `docs/workflows/handover-dataset-curation/` — 检查、筛选、裁剪、合并交接数据集；含数据集质量检测标准（jerk、峰值速度、停顿）。
- `docs/workflows/lab-training-queue/` — 在实验室服务器上启动、排队、续训、监控训练。
- `docs/workflows/dataset-viewer/` — 本地数据集网页工具：裁剪审阅/微调并重建、批量播放、训练集质量、测试集打分（含历史对比）、筛选高质量数据（curate.py，网页"筛选"页；给 agent 的步骤见 `.claude/skills/select-high-quality-data/SKILL.md`）。
