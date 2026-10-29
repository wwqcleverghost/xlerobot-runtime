# 数据集网页工具

English: [README.md](README.md)

一个本地网页，四个页面共用一个顶栏：**裁剪审阅**、**批量播放**、**训练集质量 / 筛选**、**测试集打分**。
数据集从 `~/.cache/huggingface/lerobot/wwqcleverghost/` 自动列出（含评测数据集，嵌套在 `<dir>/dataset/`、`<dir>/<set>/dataset/`
里的也会找到；旧名字的符号链接跳过；只有元数据、没有数据文件的目录不列出）。有 `success.csv` 的算评测数据集；有人工接管记录 `interventions.csv` 的（`*_takeover`）算 DAgger 训练数据，列在训练数据集里，不进测试集打分。顶栏显示数据集的 HF revision（本地构建的显示构建时间和 git commit）和当前代码的 git commit。

```bash
PYTHONPATH=src /home/wwq/miniconda3/envs/lerobot/bin/python docs/workflows/dataset-viewer/server.py   # 然后打开 http://localhost:8770
```

端口：`--port`，或环境变量 `PORT`（默认 8770）。只用 Python 标准库 `http.server`，不加新依赖；只监听 127.0.0.1。
在 Claude Code 里也可以用 `.claude/launch.json` 的 `dataset-viewer` 配置启动。

## 视频

直接播放数据集自己的 mp4（`videos/observation.images.<cam>/chunk-XXX/file-XXX.mp4`，h264 / yuv420p，浏览器能直接播），
按 `meta/episodes` 里每条的 `from_timestamp` 跳到这条，不转码。服务端支持 HTTP Range（原始录制的 moov 在文件尾），
每次最多回 2 MB，这样几十个播放器能共用浏览器对同一主机的 6 个连接。每个视频下面垫一张服务端解出来的静帧
（`outputs/dataset_viewer/cache/frames`），视频 seek 完才显示，所以加载、换条、循环都不闪黑。

## 1. 裁剪审阅（`#review`）

选一个 spec（`examples/xlerobot/*.json`）；"新建 spec…" 给任意一个数据集建 `examples/xlerobot/<名字>_trim.json`。
每条一张卡片，保留了 `review_page.py` 的全部功能。
分段 spec（带 `"segments"`，如 `takeover_corrections.json` = 从 4 组人工接管录像里切出的 15 段，`v6_corrections.json` = 20 段，`takeover_raw_segments.json` = 全部 39 段原始人工接管）每段一张卡片：蓝色把手改这段的
起点 / 终点（写回 `segments`），"这段不要"把序号写进 `"drop_seg": {"<ep>": [i]}`，`trim_stagnant_frames.py` 会跳过；不能加剪切段；
标注按 `"<ep>.<i>"` 存。细框 = 这段最初写入时的范围。"排序"默认按质量分从高到低：用 `report.py` 的质量分
（0.4 平顺 + 0.3 用时 + 0.3 停顿，对比人工示教），按页面加载时每张卡片的保留范围算；只有保留部分是完整任务时才算用时，
所以接管片段只按平顺 + 停顿打分。裁剪型卡片：

- 全宽时间轴：灰 = 剪掉，绿 = 保留，黄 = 停顿 > 1 秒，红线 = 突变，细框 = 规则算出的保留范围。
- 卡片进屏自动静音循环播放三路相机，进度竖线画在时间轴和曲线上；离屏的卡片释放播放器（同时只有附近几张在解码）。
- 在时间轴上按住拖动 = 拖动预览：拖的时候暂停，画面跟着指针走，松手后从那里接着播；单击 = 从那里播。
- 拖动蓝色起点/终点把手（拖动时视频跳到那一帧）；"加剪切段"；点剪切段或它的橙色把手选中，按 Delete / Backspace 删除（窄剪切段也能选中）。
- 整条不要、恢复初始（回到第一次写入前的 spec 版本 + 当前规则）、通过 / 有问题 / 备注 / 标签、导出 JSON。
  "整条不要"和训练集质量页的"强制去掉" / 人工标记是同一个决定：都记在 `dataset_labels.json`（`force` / `manual`），两页互相看得到；
  重建时这些去掉的条也会写进 spec 的 `drop`。
- 规则参数：结尾容差 `reset_tol`（v6 默认 10）、`--to_target`、突变规则。规则函数直接调用
  `trim_stagnant_frames.py` 的 `keep_range` / `reach_start` 和 `review_page.py` 的 `snaps` / `pauses` / `propose`，没有重写。
  改参数只影响 spec 里没有固定范围的条目。
- 视频下方是关节曲线：抓取臂 6 个关节各一行，实线 = 从臂 `observation.state`，虚线 = 主臂指令 `action`，
  用来看卡住、突变、复位微调；点或拖曲线可以跳到那一帧，右上角是"左臂 ⇄"（切换左/右臂）和"▾ 关节曲线"（点它折叠/展开这条，位置不变）；也可以用顶栏"折叠全部曲线"一键折叠/展开（会记住）。
- 筛选：来源、标签、只看有突变 / 有停顿 / 已修改、隐藏整条不要；"只播保留段"只循环保留的部分。
- 快捷键：A / D 上一条 / 下一条，空格 播放 / 暂停，`[` `]` 把当前播放位置设为起点 / 终点，Delete 删除选中的剪切段。
- 播放速度在顶栏，1× / 2× / 4× / 8×，所有页面的所有视频都用它（`<` `>` 或 `,` `.` 调整），会记住。

### 修改存在哪里

- 剪切点存进 spec 文件本身（进 git）：`drop`、`cut`，以及每条精确的保留范围 `"range": {"<ep>": [first, end]}`
  （`trim_stagnant_frames.py` 有这个字段时不再用头尾规则、`--to_target` 和 `start_s`，`cut` 照常生效；脚本里有自检）。
  第一次写入前，原始 spec 备份到 `outputs/dataset_viewer/spec_orig/`，"恢复初始"用它。
- 通过 / 有问题、备注、标签存在 `examples/xlerobot/dataset_labels.json`（进 git），按数据集目录名和 episode 编号，四个页面共用。
- 浏览器 localStorage 只记界面偏好（上次的 spec、选中的数据集、曲线是否折叠、格子宽度）。

### 按修改重建数据集

"按修改重建数据集…" 先把每条的精确范围写进 spec 的 `range`（手动改过的不变，其余用当前规则；突变规则建议的"整条不要"写进 `drop`），
把规则参数写进 `rule`、目标名写进 `dst`，再对每个目标在后台（`setsid`）运行
`trim_stagnant_frames.py --spec outputs/dataset_viewer/builds/<时间>_<目标>.spec.json --dst wwqcleverghost/<目标>`。
目标名相同的来源合并成一个数据集。目标已存在时要勾选覆盖，旧的改名为 `<目标>.old_<时间>`，不删除；正在构建的目标不能再启动。
页面右下角"构建日志"显示进度；每次构建的命令、git commit、日志在 `outputs/dataset_viewer/builds/`。
重建后所有条目都有固定范围，只有和当前规则结果不同的才标"已手动修改"。

## 2. 批量播放（`#batch`）

选一个或多个数据集，按页面宽度自动多列（"格子宽"滑块），每页 25 / 50 / 100 条，不同数据集分块显示。所有格子共用一个时钟同步播放 / 暂停（空格），
播完的停在最后一帧标"结束"，最长的一条播完后一起从头开始；速度用顶栏的全局速度；只看头部相机或三路相机。
按来源、标签、评测结果（成功 / 失败）、质量分区间筛选；每格显示数据集、episode 号、成功 / 失败、时长、质量分和标签。
点格子标题打开大图：三路相机 + 关节曲线 + 标注，A / D 换条。

## 3. 训练集质量（`#quality`）

- **历史表**：所有训练 / 示范数据集（v1–v6 和早期无版本前缀的），每列是该数据集各条的中位数：
  时长、中位 / p95 jerk、峰值速度、停顿占比、最长停顿、小幅来回次数，质量分带 95% 置信区间，IQR 标记条数。点行加入 / 移出对比（最多 8 个）。
- **排名分布**：选一个指标，每个数据集一条线（x = 数据集内排名，从小到大），颜色固定跟数据集走；悬停看数值，点一个点看那条的视频。
- **逐条表**：可排序；只看有 IQR 标记的；点行看视频。
- 指标来源：`lerobot_dataset_quality.py` 的 `episode_metrics` / `flag_outliers`（jerk 和峰值速度看从臂 `observation.state`，停顿看主臂指令 `action`，
  IQR 标记 = 超过 Q3 + 1.5 × IQR，时长也标过短），`eval-scoring/report.py` 的 `trials` / `quality` / `summarize`（基准 `v4-handover2-trimmed` 的中位数）。

### 筛选高质量数据（训练集质量页的下半部分）

选一个或多个数据集，每条 episode 一行，列出全部指标。每个条件一个开关和可调阈值，改了立刻显示：每个数据集按条件顺序累计
还剩多少条、每条被哪些条件去掉。点一行打开视频和关节曲线（A / D 换条）；行尾按钮手动强制保留 / 去掉
（存在 `examples/xlerobot/dataset_labels.json` 的 `force`，优先于所有条件）。

| 条件 | 定义 | 默认 |
|---|---|---|
| 突变 | `review_page.py` 的 `snaps()`：从臂一帧跳 >20，指令该帧变化 <5；>= 1 次就去掉 | 开 |
| 停顿 | `review_page.py` 的 `pauses()`：指令 <3 单位/秒，持续 >= 1 秒（开头的静止不算） | 开 |
| 小幅来回抖动 | `report.py` 的 `rev_rate` >= 基准中位数 × 2（基准 `v4-handover2-trimmed`，1.467 次/秒；v6 精选时用户同意过） | 开 |
| 从臂 jerk | `observation.state` 前 12 维三阶差分范数的中位数 × 30³ >= 基准中位数 × 2（基准 `v4-handover2-trimmed`，19941；用户已同意看从臂） | 开 |
| 从臂小卡顿 | 指令几乎不动（10 个非夹爪关节 \|Δaction\| 都 <2）而从臂某关节一帧跳 >6 的帧数 >= 15。**新指标，待用户确认** | 关 |
| 从臂卡住 | 指令在动（max\|Δaction\| >1）而从臂不动（max\|Δstate\| <0.3）连续 >= 4 帧算一次，>= 8 次。**新指标，待用户确认** | 关 |
| 清单：放置 / 重抓 / 中线 | 末帧胶带黄色像素 >= 500 / 每只手夹爪闭合 <= 2 次 / 先抓的手臂符合中线规则（`ep_score.py`、`select_strict.py`） | 关 |
| 清单：开头胶带立着 | 首帧胶带色块高/宽 >= 1.2。清单没定数值，**阈值待用户确认** | 关 |

jerk 基准用的是裁剪后的 handover2（`v4-handover2-trimmed`）。原始 `handover2` 含头尾静止段，中位数只有 14172，
用它当基准会多去掉很多条（v6-handover1/2-trimmed 只剩 48 条，而不是 61 条）。
v6-handover1/2-trimmed 上的参考数（2026-10-27）：去掉突变和停顿后 71 / 100，再加抖动和 jerk 后 61，再开小卡顿和卡住后 56。

- **全部 / 只看保留的 / 只看去掉的**：表格和起止画面共用这个切换（记在浏览器里）。
  "批量播放保留的 N 条" / "批量播放去掉的 N 条" 把当前结果（跨数据集合并）交给批量播放页，那里顶部显示"来自筛选：预设 X，共 N 条"和"返回训练集质量"，
  还能再按来源 / 标签 / 分数筛；打开大图时 A / D 只在这批里切换，标题显示保留还是去掉以及原因。
- **起止画面**（顶部"表格 / 起止画面"切换里的"起止画面"）：用来人工筛"开头胶带立着"和"胶带放进框里"。每页 20 张卡片，左边是保留范围第一帧、
  右边是最后一帧的头部相机大图（筛选页的数据集已经裁剪过，保留范围就是整条，和生成的 spec 一致）。每张三个按钮：
  「开头立着」「放进框里」「没问题」，键盘 1 / 2 / 3，A / D 或方向键换条，标完自动跳到下一张，再按一次取消。
  标记存在 `dataset_labels.json` 的 `manual`（`standing` / `basket` / `ok`）：前两个等于去掉这条（原因"人工：开头立着 / 放进框里"），
  在累计表里是最后一步"人工标记"；"没问题"只记作看过、变淡。可以只看还没看过的、只看有疑似提示的
  （末帧胶带像素 = 0 提示"疑似放进框里"，开头色块高/宽 >= 阈值提示"疑似立着"；只是提示，不会自动去掉，自动规则 standing / place 默认仍是关）。
  人工标记也进 `curate.py` 的同一个计算，所以命令行结果和网页一致。
- **预设**：一组开关和阈值存成 `presets/<名字>.json`（进 git），页面上可以切换、另存。默认预设 `default.json` 就是上表。
- **生成 spec 并构建**：填一个名字，"按筛选结果生成 spec"写 `examples/xlerobot/<名字>.json`：保留的条整条保留（`range` = 整条，
  沿用数据集现有的裁剪），其余进 `drop`，`note` 里记下日期和用到的条件，目标数据集默认同名、几个来源合成一个。
  然后"构建数据集…"用上面的重建功能在后台构建（可选 `--arm_only`）。来源已经是裁剪后的数据集，视频会再编码一次。
- 指标缓存在 `outputs/dataset_viewer/cache/scores/<数据集>.curate.json`（第一次算一个数据集要解码首尾画面，十几秒）。

### 命令行（和网页同一套函数）

```bash
PYTHONPATH=src python docs/workflows/dataset-viewer/curate.py --datasets raw-handover-20260929-trimmed raw-handover-20261001-trimmed \
  [--preset default] [--set jerk.x=2.5 --set stall.on=1 ...] [--report out.csv] [--spec <名字> [--build <新数据集> [--arm_only]]]
```

打印每个数据集按条件累计剩多少、每个条件去掉了哪些 episode、手动强制的条和合计保留数。
给 agent 用的步骤见 `.claude/skills/select-high-quality-data/SKILL.md`。

## 4. 测试集打分（`#eval`）

按 [`../eval-scoring/README.zh.md`](../eval-scoring/README.zh.md) 的规则（直接调用 `report.py`）：

- **历史表**：所有评测，新版本在上、同版本按总分排；成功率（Wilson）、完成度、质量、总分都带 95% 置信区间，阶段漏斗 `通过/进入`。点行看详情。
- **详情**：阶段漏斗（每阶段 `通过/进入`、Wilson 区间、计分用的 (通过+1)/(进入+2)）和逐试次表（人工标签、5 个阶段、平滑 / 速度 / 不停顿、质量、用时）。
- 失败试次有"▶ 失败时刻"按钮：从失败前 1.5 秒开始播，曲线上红色虚线标出失败时刻。
  失败时刻是近似值：最后一次抓住结束（胶带掉了或松开）前 0.5 秒；从没抓住过的取结束前 3 秒。

两个分数页的结果缓存在 `outputs/dataset_viewer/cache/scores/`，数据文件或打分脚本（report.py、score.py、smooth.py、
lerobot_dataset_quality.py）一改就重算。第一次打开历史表时后台线程算缺的（全部约 1 分钟），表格边算边更新，之后打开不到 1 秒。
改了 `server.py` 里 `_quality` / `_evaluation` 的输出要把 `SCORE_VERSION` 加一。

## 文件

- `server.py`：服务和 JSON API（`/api/datasets`、`episodes`、`curves`、`frame`、`review`、`review/edit`、`label`、`rebuild`、`builds`、
  `quality`、`eval`、`history`、`curate/*`，以及 `/video`）。`python server.py` 启动时先跑自检。
- `index.html`：顶栏和样式；`app.js`：四个页面（旧链接 `#curate` 会打开训练集质量页）。
- `curate.py`：筛选的指标、条件和命令行；`presets/`：筛选预设（进 git）。
- 运行时生成（不进 git）：`outputs/dataset_viewer/`（静帧和分数缓存、spec 备份、构建记录和日志）。
