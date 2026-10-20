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
