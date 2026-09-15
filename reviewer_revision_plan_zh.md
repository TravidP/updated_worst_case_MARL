# CB-WCE 训练与评估工作指南

[English version](reviewer_revision_plan.md) · [历史仓库审查记录](reviewer_revision_audit_2026-09-14.md)

**方案版本：** 3

**日期：** 2026 年 9 月 15 日

**仓库：** `/home/sdc_joran/Journal/deeprl_signal_control`

**实施状态：** 本文规定修订实验的工作流程。修正实现已集成到 `agents/`、`envs/` 与 `experiments/`，统一入口为 `python main.py experiment ...`。该实现保留报告中记录的历史 C01–C10 试运行证据；可视化源代码修改后需要重新完成论文实验准入验证。原有代码入口保持不变，完整的论文训练与评估矩阵尚未启动。

**文档历史：** 2026 年 9 月 15 日早先的指南修订仅涉及文档。后续可视化更新增加命令行／本地工作台的显示选项；其定向验证见[可视化报告](docs/VISUALIZATION.md)。两份原指南及其 [SHA-256 校验索引](docs/history/reviewer_guides_20260915T085603Z/checksums.sha256) 保存在[带日期的归档目录](docs/history/reviewer_guides_20260915T085603Z/)中。

## 1 实验概览

在 5×5 网格路网和 Monaco 子路网上，比较一项基线与四种需求训练策略。每种策略均应用于 IA2C、MA2C、IQL-LR（`iqll`）和 PPO，并使用五个独立训练种子。

| 方法 ID | 每个 600 秒区间内的需求规则 | WCE 模型参数学习 |
|---|---|---|
| `baseline` | 按原有固定顺序依次使用十一种训练需求配置。 | 不适用 |
| `random_group` | 均匀随机选择一种配置，使用独热需求混合权重。 | 不适用 |
| `domain_randomization` | 从 `Dirichlet(1,…,1)` 分布采样需求混合权重。 | 不适用 |
| `fixed_wce` | 由预训练 WCE 根据交通状态选择需求混合权重。 | 禁用 |
| `online_wce` | 由同一个预训练 WCE 根据交通状态选择需求混合权重。 | 每个完整回合结束后启用更新 |

基线在整个训练过程中使用原有的顺序需求方案。不再单独设置仅使用 Uniform 配置的基线或 RARL 对比。实验包含**五种方法和四类控制器**，两者是不同的实验维度。

每个最终控制器均接受 **2,320,000 个控制器训练步**：先训练共同父模型 1,000,000 步，再追加训练 1,320,000 步。一个控制器训练步是一次用于控制器学习的全路网联合决策及其后的五秒仿真。冻结控制器仿真步与优化器更新次数分别计数。训练步数相同不代表实际耗时相同。

```text
阶段 0   准备输入，并通过 C01–C10 验证
    |
阶段 I   训练共同控制器父模型，共 1,000,000 步
    |                                      |
    |                              阶段 II   冻结父模型副本
    |                                        训练并保存 WCE
    |                                             |
阶段 III 每个控制器副本追加训练 1,320,000 步
    +-- baseline                                  |
    +-- random_group                              |
    +-- domain_randomization                      |
    +-- fixed_wce <--------- 同一预训练 WCE -------+
    +-- online_wce <-------- 同一预训练 WCE -------+
    |
阶段 IV  使用配对需求评估五种方法的最终控制器
```

WCE 针对**训练至 1,000,000 步的父模型**进行训练，而非针对延长训练后的最终基线。对于同一路网、控制器类型和训练种子，五种续训方法均继承相同的控制器状态。

## 2 环境与数据集准备

### 运行环境与输入记录

1. 在仓库根目录的 `revision` 分支上工作，记录实验使用的准确源代码提交以及任何未提交修改。
2. 以现有 `deeprlsc` 环境作为运行环境起点。审查记录识别到 Python 3.6.13 和旧版 TensorFlow 技术栈；默认 shell 中的 Python 3.13 环境并非已使用的训练环境。
3. 记录实际使用的 Python、TensorFlow、NumPy、TraCI、SUMO、操作系统、硬件、线程数和并发负载设置。审查时的 SUMO 构建版本为 `1_26_0+0455-77b9dbc222e`。
4. 记录路网文件、控制器配置、监测车道集合和有序需求清单的哈希值。
5. 建立专用修订数据集和输出位置。保留现有路网、原始 CSV、检查点及历史结果。

以下已有命令用于检查起始环境，不会启动训练：

```bash
cd /home/sdc_joran/Journal/deeprl_signal_control
conda activate deeprlsc
python --version
sumo --version
git branch --show-current
git rev-parse HEAD
git status --short
```

历史训练入口保留旧行为；修正实验统一使用 `python main.py experiment ...`，详见根目录 README_zh.md。

### 可执行准备命令与共用 Shell 变量

**目的与输入。** 先按上方命令从仓库根目录激活已有环境。已建立的运行环境为 Python 3.6.13、TensorFlow 1.12.0 和 NumPy 1.19.5。以下变量构成 Grid/IA2C 的完整示例。使用 Monaco 时，将 `CBWCE_NETWORK=monaco`，并将 `CBWCE_DATASET_ROOT=real_net_subnet/demand_groups/revised`。控制器 ID 为 `ia2c`、`ma2c`、`iqll`、`ppo`；五个论文训练种子分别运行。试运行种子 9001 不计为论文实验重复。

**试运行／论文实验准备命令。** 两种模式共用输入准备流程。`prepare` 创建归一化 CSV、场景定义和生效 INI，或检查现有副本是否一致；不会覆盖内容不同的已准备文件。`--materialize` 启动短时 SUMO 路径解析会话，创建完整交通文件，不训练控制器。因此下面两条物化命令不是只读准备检查。

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
export CBWCE_NETWORK=grid
export CBWCE_CONTROLLER=ia2c
export CBWCE_SEED=101
export CBWCE_PILOT_SEED=9001
export CBWCE_DATASET_ROOT=data_traffic/revised
export CBWCE_GATE=runs_eval/revised/verification/final_integration_20260914/gate.json

python main.py experiment prepare
python main.py experiment check --gate "$CBWCE_GATE"
python main.py experiment prepare --materialize --network grid
python main.py experiment prepare --materialize --network monaco
```

**实现脚本。** [CLI](experiments/cli.py) 调用[输入准备](experiments/prepare.py)、[需求加载](experiments/demand.py)、[场景生成](experiments/scenarios.py)与 [SUMO 环境](envs/experiment_env.py)。

**输出与检查。** 每个路网包含十一份训练 CSV、十一项已见需求定义、十二项测试定义、六项验证定义。每项生成十次实现，得到每路网 290 个、共 580 个交通文件，其中包含验证集。原始 CSV 保持不变。检查 `check` 输出中的 `inputs`、`modules`、`gate` 与 `ready`：仅有 `ready=true` 不代表准入检查通过，命令也可能正常返回但报告 gate 失败。论文训练前必须检查指定 gate。当前实现缺少已准备清单／配置时会回退到原输入；本研究应始终运行 `prepare` 并确认 `prepared=true`。

### 可选 SUMO 可视化

在 `parent`、`wce`、`continue` 或 `evaluate` 命令中添加 `--visualization`，即可打开独立的本地 SUMO 窗口。使用 `--no-visualization`，或同时省略两个选项，则保持关闭。两个选项互斥；不要使用 `--visualization false`。选项放在 `python main.py experiment <stage>` 后面；`revision.runner` 兼容入口同样支持。

```bash
# SUMO window on
python main.py experiment parent --network grid --controller ia2c \
  --seed 9001 --pilot --steps 160 --visualization

# SUMO window off (also the default when neither flag is supplied)
python main.py experiment parent --network grid --controller ia2c \
  --seed 9001 --pilot --steps 160 --no-visualization
```

针对一次预定运行，选择其中一条命令，不必两条都执行。两者均为短试运行；论文预算和学习参数保持不变。在本地工作台中，先选择 **SUMO 可视化 → 开启／关闭**，再点击 **检查并预览**。选择在切换语言后保留，并反映在生成命令和任务请求中。窗口显示在训练电脑上，不嵌入浏览器。已经运行的仪表板服务需要重启才能加载更新后的后端，然后刷新页面；不要仅为刷新界面而中断正在执行的训练任务。

| 实际生效值 | 含义 | 实现／配置来源 | 是否可配置 |
|---|---|---|---|
| 可视化默认 `false` | 无窗口运行 SUMO；`true` 打开 `sumo-gui` 并自动播放 | CLI `--visualization` / `--no-visualization`；`experiments/runner.py`、`envs/experiment_env.py`、`experiments/dashboard.py` | 每次命令／本地工作台选择；不属于 INI 字段 |

需要可运行的 `sumo-gui` 和图形桌面。Linux 中的 `DISPLAY` 必须指向可访问的 X 显示；仅有 Wayland 变量并不能为 SUMO 提供 X 显示。`check` 报告 `sumo_gui` 和 `display`，GUI 启动前检查会拒绝缺少必要条件的请求。过期或不可访问的显示仍可能在 SUMO 启动时失败。后台输入准备／路径物化会话继续无窗口运行。所选训练／评估回合使用 `--start --quit-on-end` 打开 GUI 会话；可在 SUMO 的 View Settings 中调整颜色／缩放，通过 Delay 调整播放速度。

此选项在任务启动时生效，不是在运行中切换。恢复时可以为新尝试选择任一显示模式；检查点兼容性和原有学习预算不变。**论文计时比较应关闭可视化**，因为渲染和播放延迟会增加实际耗时。所选布尔值以 `visualization` 保存到 `<run>/manifest.json`；每个 `runtime/startup_*.json` 保存实际 `visualization` 和 SUMO 启动命令。即使任务开启可视化，构造器／初始化及评估套件路径准备会话仍可能无窗口运行。排队 NPZ、交通 JSONL、检查点及结果路径保持下文描述的格式。

**本次修改后的验证状态：** 9 月 14 日的准入文件保留为历史证据，其源代码哈希已不再匹配可视化实现。论文训练前请运行 `python main.py experiment verify --workers 4`，并选择其新生成的通过记录。定向可视化检查不能代替完整八组准入验证。

### 路网与训练需求配置

| 设置 | Grid 网格路网 | Monaco 子路网 |
|---|---:|---:|
| 信号控制器数量 | 25 | 28 |
| 去重后的受控进口车道数 | 150 | 116 |
| 明确列入清单的训练需求配置数 | 11 | 11 |
| 每种配置的参考需求总量 | 3,000 veh/hour | 2,383.3333 veh/hour |
| 控制器决策间隔 | 5 秒 | 5 秒 |
| 黄灯时长 | 2 秒 | 2 秒 |
| 需求选择间隔 | 600 秒 | 600 秒 |
| 训练回合时长 | 6,600 秒 | 6,600 秒 |
| 每个完整回合的控制器训练步数 | 1,320 | 1,320 |

使用 `large_grid/data/` 下的网格路网和 `real_net_subnet/data/` 下的 Monaco 路网。记录哈希值之前，确认每个 SUMO 配置实际引用的路网文件。

为已有的十一种方向性、中心与外围之间以及 Uniform 需求配置创建归一化副本。保留 OD 比例，同时保存原始总量和归一化系数。初始训练与基线续训使用原有配置顺序：网格路网使用既有十一种配置的文件名字母排序，Monaco 使用明确指定的十一种配置顺序。将文件名按顺序写入清单，不要在每次运行时重新推断顺序。

历史网格路网加载器会发现**十二种**有效配置，其中包括 `demand_5x5_sparse.csv`。必须使用明确列出十一种配置的训练清单，避免新增测试 CSV 改变回合时长或 WCE 输出维度。修订实验中的 Monaco 同样使用明确列出的十一种配置清单。

### 数据集验证与输出组织

将训练、验证和测试数据集分开存放。检查需求率是否为有限非负值、道路边是否存在，以及路径是否满足车辆类别的通行约束并保持连通。无法生成路径的需求不得被静默丢弃。

使用唯一的运行 ID 和尝试 ID。以下为集成后的目录结构；历史文件保留原路径：

```text
config/revised/
data_traffic/revised/{train,validation,test}/
real_net_subnet/demand_groups/revised/{train,validation,test}/
runs/revised/<network>/<controller>/seed_<seed>/<stage>/<run_id>/
output_adversary/revised/                 # grid WCE
output_adversary_monaco/revised/          # Monaco WCE
output_coevolution/revised/               # grid continuations
output_coevolution_real/revised/          # Monaco continuations
runs_eval/revised/                       # evaluation and verification
output_result/revised/                   # tables and report exports
figs/revised/                            # scientific plots
```

每份运行清单记录方法、阶段、路网、控制器、随机种子流、输入哈希、准确的父检查点、训练步数预算、奖励定义和输出位置。输出记录中必须保存实际生效的值。

### 阶段与输出目录对应关系

| 根目录 | 内容 |
|---|---|
| `runs/revised/` | 两个路网：父模型和 `baseline` 续训 |
| `output_adversary/revised/` | Grid 离线 WCE |
| `output_adversary_monaco/revised/` | Monaco 离线 WCE |
| `output_coevolution/revised/` | Grid 四种比较方法续训 |
| `output_coevolution_real/revised/` | Monaco 四种比较方法续训 |
| `runs_eval/revised/` | 评估，以及 `verification/`、`preparation/`、`jobs/` 记录 |
| `output_result/revised/` | 报告目录：CSV、JSON、PNG、SVG |
| `figs/revised/` | 单独保留的图表副本；不是报告的第二个自动输出目录 |

阶段运行器通过 [experiments/protocol.py](experiments/protocol.py) 选择路径：`<stage-root>/<network>/<controller>/seed_<seed>/<stage-or-method>/<run_id>/`。阶段名称为 `parent`、`wce` 或 `evaluate`；续训使用方法 ID。自动运行 ID 包含 `pilot_` 或 `publication_`、UTC 时间戳及随机后缀。显式 `--output` 必须指向新目录。当前报告命令省略 `--output` 时，即使读取论文数据也生成 `pilot_` 前缀目录；应根据观测的清单判断数据性质，不能只看报告目录前缀。

## 3 必须完成的修正与验证

**用于论文的完整训练矩阵启动之前，必须实施并验证全部十项修正。** 下列验收检查是持续适用的要求；已观察到的结果及准确验证范围见阶段 0 的链接。按验证运行 ID 保存检查结果和证据。历史源码位置及审查发现保留在[归档审查记录](reviewer_revision_audit_2026-09-14.md)中。

### C01 正确应用评估种子

**必需行为。** 在评估重置之前设置 `env.train_mode = False`。传入预期的 rollout 索引，并记录每次仿真实际使用的 SUMO 种子。

**验收检查。** 分别请求两个不同的评估种子，确认仿真器实际收到对应种子。重复相同种子的评估必须复现其外生需求实现。

### C02 分离需求与控制器的随机性

**必需行为。** 为需求生成、控制器动作采样、WCE 采样、经验回放采样和 SUMO 建立独立随机流。评估时，在比较控制器之前，预先生成并保存完整的车辆数量、发车时间、OD 对、路径道路边序列和速度因子。各方法复用同一份完整需求文件。

**验收检查。** 更换控制器或其动作采样种子，不得改变预定交通需求文件。另行记录车辆实际插入情况，因为拥堵可能延迟车辆进入路网。各训练方法会有意选择不同需求；需求文件完全相同的要求适用于配对评估。

### C03 统一控制器奖励处理

**必需行为。** 初始训练与全部五种续训方法使用同一条控制器奖励构造路径。IA2C、PPO 和 IQL-LR 接收规定的共享路网奖励；MA2C 接收规定的邻域加权奖励。第 4 节定义的归一化仅应用一次，移除修订路径中额外的旧版归一化，包括 Monaco 非预期的额外缩放。

**验收检查。** 对于相同的路网和控制器类型，给定相同的局部排队测量，各方法必须产生相同的学习奖励向量。

### C04 一致地更新 MA2C 指纹

**必需行为。** 在初始训练、全部续训方法、冻结控制器的 WCE 训练以及评估中更新指纹。在回合边界重置指纹和循环网络状态。冻结控制器仍需更新推理状态和指纹，同时保持其已学习的模型参数固定。

**验收检查。** 检查指纹是否反映当前策略输出，并在回合之间正确重置。确认冻结控制器的模型参数保持不变。

### C05 匹配 IQL 学习频率

**必需行为。** 当经验回放包含足够样本后，每新收集二十个控制器训练转移，触发一次 IQL backward 调用。各方法保持相同的回放容量、采样规则，以及每次 backward 调用中每个智能体十次小批量更新。冻结控制器仿真不得添加回放样本或推进学习计数器。

**验收检查。** 比较五种方法等长度运行中的控制器训练转移数、backward 调用数和小批量更新数。同一路网与控制器组合内的计数必须一致。

### C06 修正 Monaco IQL 接口

**必需行为。** 使用 IQL 推理接口，不能套用 actor-critic 接口。通过经验回放实际支持的属性读取数据，不能假设存在 `.obs` 字段。评估和离线 WCE 训练使用贪心 IQL 推理；控制器学习阶段使用既定探索调度。

**验收检查。** 为 Monaco IQL 执行短运行检查，覆盖冻结控制器的 WCE 训练、续训、检查点加载和评估，确保没有函数签名或回放缓冲区属性错误。

### C07 对齐排队测量与奖励缩放

**必需行为。** 对全局去重后的受控进口车道，每秒测量一次不设上限的停止车辆数。控制器局部成本、离线与在线 WCE 奖励以及主要评估指标使用同一测量来源。在修订路径中移除网格路网的“排队加等待时间”目标和 Monaco 排队上限。应用第 4 节中的公式。

**验收检查。** 从保存的车道测量重新计算控制器成本和 WCE 奖励。只经过规定的聚合与缩放后，结果必须与日志一致。检查中应包含排队超过十辆车的车道。

### C08 恢复完整训练状态并拒绝加载失败

**必需行为。** 保存模型参数、优化器状态、调度状态、相关缓冲区、可恢复的随机状态、计数器和父检查点标识。在优化器变量创建之后构建检查点保存器。必需检查点缺失或不兼容时必须停止运行，不得静默初始化随机模型。

常规可恢复检查点在回合边界保存。父模型达到精确预算截止点时，使用正确的 bootstrap 值处理尚未更新的 on-policy 样本，即使回合未结束，也保存明确标记的阶段边界检查点。随后，全部续训方法从同一保存状态开始新的回合。若回合被中断，从最后一个完整检查点恢复，并记录丢弃的计算工作。

**验收检查。** 核查恢复后的模型参数、优化器变量、经验回放内容、调度状态、随机状态和计数器。在受控随机条件下比较下一次学习更新。确认检查点缺失和不兼容均会明确报错并停止。

### C09 拒绝不完整的评估 rollout

**必需行为。** 在修订评估路径中移除末值填充和零值填充。验证时间戳、样本数和完整的 3,600 秒时域。为失败或不完整的尝试保存明确状态，并将其排除在性能汇总之外；不得将其转换为零排队结果。

**验收检查。** 主动中断一次 rollout，确认其被标记并排除。拥堵但完成全部时域的仿真仍是有效结果。若基础设施故障导致重跑，保留失败尝试，并复用规定的检查点、需求和种子。

### C10 保持输出来源一致且可追溯

**必需行为。** 每次运行使用唯一输出目录和不可变输入清单，记录路网、需求、配置、检查点和种子标识。局部重跑必须创建独立尝试记录，不得覆盖完整运行的清单或混入无关检查点的结果。

**验收检查。** 每项结果均能追溯到实际输入和检查点。核对预期与已完成的 rollout 数，拒绝重复标识或不兼容的清单。

## 4 实际生效的控制器与 WCE 参数

下表描述当前修正实现的实际行为。`MODEL_CONFIG` 来自 `config/revised/config_<controller>_<large|real>.ini`，其中 `large` 表示 Grid。[控制器适配器](agents/controller.py)、[WCE 适配器](agents/wce.py)、[策略类](agents/policies.py)和[测量／奖励代码](experiments/core.py)决定实际行为。JSON 或 INI 中存在某字段，并不代表它是有效调参开关；部分论文实验常量也在代码中校验。不能只改一个字段就认为整个实验方案已经改变。

### 控制器参数及其含义

| 实际生效值 | 含义 | 实现／配置来源 | 是否可配置 |
|---|---|---|---|
| IA2C/MA2C 学习率 `0.0005`；PPO `0.0003`；IQL `0.0001` | 恒定学习率；未应用学习率衰减 | `MODEL_CONFIG.lr_init`; `agents/controller.py` | INI；配置改变后需重新验证 |
| IA2C/MA2C 批量：Grid `120`，Monaco `40`；PPO `120`；IQL `20` | actor-critic/PPO 为 on-policy 序列长度；IQL 为回放小批量大小 | `MODEL_CONFIG.batch_size`; `Controller.observe/flush` | INI；更新时机还受下方固定规则控制 |
| Actor-critic/PPO：wave 全连接层 `128`、wait 全连接层 `32`、LSTM `64`；MA2C 指纹全连接层 `64` | 紧凑循环策略；IQL 使用线性 Q 策略，不使用 LSTM | `num_fw`, `num_ft`, `num_lstm`, `num_fp`; `agents/recurrent.py`, `agents/policies.py` | 宽度：INI；架构：代码 |
| IA2C/MA2C：RMSProp；PPO/IQL：Adam | RMSProp 衰减 `0.99`；actor-critic/PPO epsilon 为 `1e-5`；IQL 使用 Adam 默认值 | `masked_loss`; `LRQPolicy.prepare_loss`; `rmsp_alpha`, `rmsp_epsilon` | 优化器选择：代码；所列 RMS 字段：INI |
| 控制器折扣因子 `0.99` | 回报／TD 折扣，与 MA2C 空间邻域权重不同 | `Controller.flush`; IQL `prepare_loss(..., .99)` | 代码固定；只改 INI `gamma` 对此处无效 |
| 梯度范数上限 `40` | 裁剪梯度全局范数，不裁剪排队奖励 | `MODEL_CONFIG.max_grad_norm`; controller loss code | INI |
| 熵系数 `0.01`；价值损失系数 `0.5` | 循环策略损失为 actor 损失 + `0.5 * value_coef * masked MSE` − 熵项，因此 MSE 实际乘数为 `0.25`；IQL 仅使用 TD MSE | `entropy_coef_init`, `value_coef`; `masked_loss` | INI 系数；修正适配器没有熵衰减调度 |
| PPO 裁剪范围 `0.2`；优化轮数 `4`；优势归一化 `true` | 比率限制为 `[0.8,1.2]`；四次优化使用相同的起始循环状态 | `masked_loss`, `Controller.flush`; `ppo_adv_norm` | clip／epochs：代码；优势归一化：INI |
| MA2C 邻域权重 `0.9` | 负局部排队加上加权的直接邻居负排队 | `experiments/core.py: QueueMetric.rewards` | 代码固定 |
| IQL 回放容量 `1000`；每 `20` 步学习；每个智能体执行 `10` 次小批量更新 | 循环覆盖回放；不放回抽样；回放样本充足是额外条件 | `Controller.observe/_update_iql` | 容量／频率／更新次数：代码；小批量大小：INI |
| IQL 探索率 epsilon `max(0.01, 1 − scheduler_steps / 500000)` | 仅学习阶段推进计数器，且在选动作之前推进；在 `495000` 步到达下限，续训保持 `0.01` | `Controller.epsilon/act` | 代码固定；不使用旧 INI epsilon 调度 |
| 仅使用 CPU；TF 算子内／算子间线程数 `1/1` | 控制器与 WCE 会话均禁用 GPU；BLAS／OMP 变量控制各自库线程 | `Controller.__init__`, `WCE.__init__`; setup shell variables | TF／GPU：代码；BLAS／OMP：环境变量 |

不同算法的 batch 含义不同：IA2C/MA2C 收集 on-policy 序列；PPO 将该序列复用四次；IQL 从回放中抽取小批量，其收集时钟另按二十个转移计数。算法或路网之间可以不同；同一路网／控制器下的五种方法必须保持相同参数和更新时机。

**探索调度澄清。** 旧文字将其描述为前 500,000 步衰减。实际公式在 495,000 步达到 0.01，且第一个学习动作使用已经加一的计数器。本文说明该差异，不修改调度。冻结控制器 WCE 训练和评估使用贪心 IQL 动作，不推进调度。

**保留的历史字段。** 旧 INI 中的 `reward_norm`、`reward_clip`、`TRAIN_CONFIG.total_step`、`test_interval`、`log_interval`、自动 `resume/resume_step`、学习率／熵衰减字段，以及 IQL epsilon／buffer 设置，不为修正循环提供这些控制。PPO clip／epoch 的 INI 值目前与代码常量相同，但并不驱动它们。加载由显式 CLI 检查点参数控制。模型 INI 内容仍纳入检查点签名，因此修改未使用的字段也可能导致不兼容。

### 时间、预算与归一化

| 实际生效值 | 含义 | 实现／配置来源 | 是否可配置 |
|---|---|---|---|
| 控制器周期 `5 s` = 黄灯 `2 s` + 绿灯 `3 s` | 一次联合决策；每秒仿真后测量排队，包括黄灯期间 | `envs/experiment_env.py: step/_simulate` | 修正代码固定 |
| 训练回合 `6600 s`；`11 × 600 s`；`1320` 次联合转移 | 回合与需求区间时钟；评估为 `3600 s`／`720` 次转移 | `experiments/runner.py`; `experiment_env.py` | 方案记录这些值，循环中也有固定常量 |
| 排队奖励缩放除数 `100`；禁用裁剪 | 学习奖励与 WCE 区间代价仅除以一次；评估保留车辆单位 | `experiments/core.py: QueueMetric` | 代码固定；不读取旧 `reward_norm/reward_clip` |
| 父模型 `1000000` 步；续训 `1320000` 步；离线 WCE `500` 回合 | 预算统计真实学习转移；WCE 阶段仅增加冻结仿真 | `config/revised/protocol.json`; runner goal selection | 方案预算；试运行可用 `--steps/--episodes`；论文检查还固定校验总量 |

### 排队指标与控制器奖励

令 $L$ 为固定且去重后的受控进口车道集合，$q_\ell(t)$ 为每秒采样的不设上限的停止车辆数。定义：

$$
Q(t)=\sum_{\ell\in L}q_\ell(t),\qquad
J_Q=\frac{1}{H}\sum_{t=1}^{H}Q(t).
$$

将 $J_Q$ 报告为**受控进口车道平均总排队车辆数**，单位为辆，越低越好。审查记录中的车道数为网格路网 150 条、Monaco 116 条。保存并验证准确的车道清单。SUMO 的车道停止车辆数以速度低于 0.1 m/s 为判据。[SUMO 车道数值文档](https://sumo.dlr.de/docs/TraCI/Lane_Value_Retrieval.html)

对于每个五秒的控制器转移，对每个路口的局部排队在全部五秒内取平均。IA2C、PPO 和 IQL-LR 的每个智能体接收负的全路网平均排队值。MA2C 接收负的局部平均排队值，加上以 0.9 加权的负邻居局部平均排队值。将所得学习奖励除以 100，**仅执行一次**，并禁用奖励裁剪。在修订奖励路径中不得保留额外的 Monaco 专用除数。

### WCE 参数与更新行为

| 实际生效值 | 含义 | 实现／配置来源 | 是否可配置 |
|---|---|---|---|
| Grid CNN：卷积层 `32/64`、全连接层 `128`；Monaco GCN：两层，每层宽度 `64` | Grid 使用有序 wave／wait 特征；Monaco 使用归一化 lane-wave，不包含控制器指纹 | `agents/wce.py`; active Gaussian policy classes in `agents/policies.py`; `wce_observation` | 架构／特征：代码 |
| 动作维度 `11`；高斯 logits → softmax | 独立且可恢复的 WCE 随机流生成噪声；需求混合权重非负且和为一 | `WCE.act`; `Streams` | 代码与十一需求组约定固定 |
| WCE 学习率 `0.0005`；RMSProp 衰减 `0.99`、epsilon `1e-5` | 优化器设置；WCE `prepare_loss` 中的 `0.99` 是 RMSProp 衰减，不是回报折扣 | `WCE.__init__/observe`; Gaussian policy loss | 代码固定 |
| WCE 熵系数 `0.01`；价值损失系数 `0.5`；梯度范数 `40` | 高斯 actor-critic 损失与梯度控制 | `WCE` and Gaussian policy `prepare_loss/backward` | 代码固定 |
| WCE 折扣因子 `1.0`；批量为 `11` 次宏观转移 | 对十一个区间奖励反向累加；启用学习时每完整回合更新一次 | `WCE.observe` | 代码固定；不读取控制器 INI gamma |
| 每 `600` 秒选择动作；不额外缩放或裁剪奖励 | 固定 WCE 模型参数不等于固定需求混合权重；在线与固定使用相同抽样规则 | `experiments/runner.py`; `QueueMetric.wce` | 代码固定 |

直接从规范排队测量计算 WCE 奖励，而不是累加已经转换的控制器奖励：

$$
r_{\mathrm{WCE},k}=\frac{1}{100}\frac{1}{600}\sum_{t\in k}Q(t).
$$

WCE 最大化正的排队成本；控制器通过负排队奖励进行学习。评估使用未经缩放的 $J_Q$。在十一个等时长区间和 WCE 折扣因子 1.0 下，WCE 奖励之和与该回合的平均排队成正比。

固定 WCE 与在线 WCE 必须共享相同的初始检查点、架构、观测和动作采样规则。两种方法的需求混合权重均可每 600 秒改变。只有在线 WCE 在回合之间更新模型参数。

## 5 初始基线训练

### 阶段 0 准备与验证

**前置条件：** 第 2 节输入及 C01–C10。

集成运行器已经实现修正。代码集成后，对全部八种路网与控制器组合执行新的短运行检查。覆盖五种续训方法和冻结控制器的 WCE 路径。使用不属于论文实验种子集合的试运行种子，记录每项修正的验证证据，并在完整训练之前解决失败问题。试运行结果仅作为调试证据，不纳入论文性能结果。

**输出：** 经验证的输入清单，以及表明全部必需检查均已通过的验证记录。当前[集成报告](docs/INTEGRATION_REPORT.md)链接了已接受的试运行证据及对应源代码版本的准入记录。实现或实验输入发生变化时，必须重新验证准入条件。

### 阶段 0 命令参考与已接受证据

**目的／前提：** 输入准备后建立与源代码对应的 C01–C10 准入记录。**输入：** 当前代码及数据／配置哈希，不需要父模型检查点。**试运行命令：** verifier 以试运行预算检查八个路网／控制器组合。**论文实验前提命令：** `check --gate` 检查所选证据，不启动训练。

下方链接给出 9 月 14 日的最终 gate；仅当 `check` 报告 `gate: passed` 时复用。gate 失效或需要重新验证时运行 `verify`，然后用输出的 gate 路径替换占位值。仅阅读本指南不需要重跑验证矩阵。

```bash
python main.py experiment check --gate "$CBWCE_GATE"
python main.py experiment verify --workers 4

export CBWCE_GATE='REPLACE_WITH_THE_NEW_VERIFICATION_GATE_JSON'
python main.py experiment check --gate "$CBWCE_GATE"
```

**脚本：** [experiments/verify.py](experiments/verify.py)、[修正测试](tests/test_corrections.py)、[工作流测试](tests/test_workflow.py)，使用同一个修正运行器。**输出：** `runs_eval/revised/verification/<run_id>/gate.json`、测试日志及每个路网／控制器的案例目录。**完成条件：** 八组案例和全部测试通过，所选 gate 与源代码、输入及证据哈希一致。

**已有证据：** [集成报告](docs/INTEGRATION_REPORT.md)、[八组案例 gate](runs_eval/revised/verification/integrated_20260914/gate.json)、[已接受最终 gate](runs_eval/revised/verification/final_integration_20260914/gate.json)及[定向验证范围](runs_eval/revised/verification/final_integration_20260914/validation_scope.json)。八组试运行包含 40 次续训、八次恢复检查、64 条完整评估和八次故意中断。随后的身份／套件方法标识／计时／报告修正通过 18 项测试、七项 HTTP 检查及一条附加完整评估。定向界面修正后没有再次重跑整个八组矩阵，学习器／环境／配置逻辑未改变。这些是已有记录，不是本次文档更新新执行的测试。

### 阶段 I 训练共同父控制器

**必须持续执行的检查：** 训练期间保持 C02–C08 和 C10 生效。

使用主训练种子 `101, 202, 303, 404, 505`。对每个路网、控制器和种子组合：

1. 使用修订后的排队奖励和固定配置，初始化全新控制器。
2. 按原有顺序依次使用十一种归一化需求配置进行训练。
3. 在**恰好 1,000,000 个控制器训练步**时停止，必要时允许最后一个回合未完成。
4. 使用正确的 bootstrap 值处理最后一个未满的 on-policy 批次。在该次更新之后保存；按算法需要保留 IQL 经验回放和调度状态。
5. 保存完整共同父检查点、累计计数器、输入哈希、随机种子流和阶段耗时。

**输出：** 40 个独立父检查点。由同一个训练父模型派生五次续训，不能替代五个独立训练的父模型。

此父模型不是最终基线结果。在阶段 III 中，基线将继续训练至与四种对比方法相同的最终预算。

### 阶段 I 命令、脚本与保存状态

**目的与必需输入：** 使用有序十一需求组清单和生效 INI，从头训练共同父模型。不提供 `--parent`、`--wce` 或 `--resume`。检查 C02–C08／C10，以及路网／控制器／种子选择。

**试运行命令：** 160 个真实学习转移。对于 120 步 rollout，一个完整批次后还有 40 个真实样本，通过带掩码的部分批次处理。该试运行检查机制，不是论文所需的已训练父模型。

```bash
python main.py experiment parent --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --pilot --steps 160 --checkpoint-every 1
```

**论文实验命令：** 精确运行 1,000,000 个学习转移；不传入 `--steps` 或 `--episodes`。

```bash
python main.py experiment parent --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" --gate "$CBWCE_GATE"
```

**脚本：** `main.py` → [experiments/cli.py](experiments/cli.py) → [experiments/runner.py](experiments/runner.py)；学习由 [agents/controller.py](agents/controller.py) 实现，测量由 [envs/experiment_env.py](envs/experiment_env.py) 实现。

**输出：** `runs/revised/<network>/<controller>/seed_<seed>/parent/<run_id>/`。本试运行的最终检查点为 `checkpoint_000000160`，论文实验为 `checkpoint_001000000`。检查该运行的 `result.json`：要求 `status=complete`、正确的 `learning_steps` 和显式 `checkpoint` 字段。CLI 也输出返回的检查点路径。将该准确路径复制到下面相应变量；`REPLACE_...` 是特意保留的不可直接执行占位值，不是检查点名称。仅设置自己实际训练过的模式。

```bash
export CBWCE_PARENT_PILOT='REPLACE_WITH_COMPLETED_PILOT_PARENT_CHECKPOINT'
export CBWCE_PARENT='REPLACE_WITH_COMPLETED_PUBLICATION_PARENT_CHECKPOINT'
python main.py experiment check --checkpoint "$CBWCE_PARENT"
```

**完成条件：** 父模型包含模型与优化器变量、缓冲区、计数器和随机状态。不能用历史纯权重检查点代替，也不能把一个种子的五次续训当成五个独立父模型。

## 6 针对冻结控制器训练 WCE

### 阶段 II 训练并保存共同 WCE

**必须执行的检查：** C02–C04、C06–C08 和 C10。C05 的计数器必须确认冻结控制器未进行学习。

对每个阶段 I 父模型：

1. 加载准确的 **1,000,000 步父检查点**，冻结控制器已学习的模型参数。
2. 禁用控制器优化、回放插入和学习调度推进。继续正确更新推理状态与 MA2C 指纹，并在回合之间重置状态。
3. 生成 WCE 训练经验时，IA2C/MA2C/PPO 使用策略采样动作，IQL 使用贪心动作。
4. 训练 WCE **500 个回合**，共 **5,500 个宏观转移**。每个宏观转移包含一个 600 秒区间，即 120 个冻结控制器仿真步。
5. 保存最终 WCE 检查点、准确的父模型标识、计数器和耗时。验证控制器模型参数始终不变。

**输出：** 40 个预训练 WCE 检查点。每次 WCE 训练增加 **660,000 个冻结控制器仿真步**，这些步数均不计入控制器训练预算。

`fixed_wce` 和 `online_wce` 使用同一个已保存的 WCE 作为初始模型。其余三种方法独立运行时不需要训练 WCE。

### 阶段 II 命令、脚本与保存状态

**目的与输入：** 针对对应的冻结父模型学习有挑战性的需求混合。所选父模型必须匹配路网、控制器和训练种子。论文 WCE 必须从一百万步父模型开始；下方试运行从 160 步测试父模型开始。禁用控制器优化、回放插入和学习调度推进，同时保留推理状态／指纹更新。

**试运行命令：** 两个回合、22 个宏转移、2,640 个冻结控制器仿真步。

```bash
python main.py experiment wce --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --parent "$CBWCE_PARENT_PILOT" --pilot --episodes 2 --checkpoint-every 1
```

**论文实验命令：** 500 回合、5,500 个宏转移、660,000 个冻结控制器仿真步。

```bash
python main.py experiment wce --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --parent "$CBWCE_PARENT" --gate "$CBWCE_GATE"
```

**脚本：** 共用 `experiments/runner.py`、[agents/wce.py](agents/wce.py)及控制器／环境适配器。**输出：** Grid 为 `output_adversary/revised/<network>/<controller>/seed_<seed>/wce/<run_id>/`；Monaco 使用 `output_adversary_monaco/revised/`。本试运行最终检查点后缀为 `000002640`，论文实验为 `000660000`。WCE 检查点包也包含冻结控制器；其控制器训练步数分别仍为 160 或 1,000,000。

从已完成运行的 `result.json` 选择准确检查点，然后设置：

```bash
export CBWCE_WCE_PILOT='REPLACE_WITH_COMPLETED_PILOT_WCE_CHECKPOINT'
export CBWCE_WCE='REPLACE_WITH_COMPLETED_PUBLICATION_WCE_CHECKPOINT'
```

**完成条件：** 控制器参数／计数器保持不变，WCE 更新次数为两次或 500 次，父检查点哈希对应实际使用的控制器。固定与在线续训必须使用与该父模型对应的同一个预训练 WCE。

## 7 基线续训与对比训练

### 阶段 III 训练五种续训方法

**必须执行的检查：** C02–C08 和 C10；各方法使用同一控制器交互与更新路径。

1. 对每个路网、控制器和种子组合，从阶段 I 完全相同的模型、优化器、调度、经验回放及已记录训练状态创建五个控制器副本。开始新的回合，并以一致方式重置循环网络状态。
2. 对所选方法应用第 1 节的需求规则。
3. 收集**恰好 1,320,000 个追加控制器训练步**，相当于 1,000 个完整修订回合。在**累计 2,320,000 步**时停止。
4. 同一路网与控制器组合内，各方法保持相同的控制器奖励构造、批次调度、IQL 更新频率、PPO 优化轮次、检查点策略和失败处理。
5. 保存各方法精确最终预算对应的检查点及父模型标识。记录控制器训练步数、优化器调用数、小批量更新数、WCE 决策与更新数，以及实际耗时组成。

**方法行为：** `baseline` 延续原有顺序需求方案。`random_group` 以相同概率从十一种配置中选择一种，并使用独热向量。`domain_randomization` 采样总和为一的非负需求混合权重，使用 `Dirichlet(1,…,1)` 权重组合所有配置的 OD 需求率。由于各训练配置的需求总量相同，两种方法均保持相同的预期需求总率，同时改变空间分配。

`fixed_wce` 根据状态进行推理，但不存储 WCE 学习转移，也不执行 WCE 优化器更新。`online_wce` 记录 WCE 转移，并在回合边界收集满十一个宏观转移后更新。其续训阶段包含 11,000 个 WCE 学习转移和 1,000 次回合更新。

**输出：** 200 个最终控制器检查点：40 个延长训练后的基线，以及四种对比方法合计 160 个控制器。训练一百万步的父模型不作为额外对比方法。

### 阶段 III 五种方法的命令

**目的／输入：** 从同一个选定父模型分出五份完整控制器状态。固定／在线方法还需要同一个 WCE，且其父模型哈希必须一致；其他方法不加载 WCE。不同方法可以选择不同需求，但控制器架构、奖励路径和更新时机保持一致。

**试运行命令：** 每次选择一条。每条增加 2,640 个学习步，从 160 步父模型达到 2,800 步。固定／在线试运行覆盖两个完整回合，可检查冻结和更新的区别。

```bash
python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --method baseline --parent "$CBWCE_PARENT_PILOT" \
  --pilot --steps 2640 --checkpoint-every 1

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --method random_group --parent "$CBWCE_PARENT_PILOT" \
  --pilot --steps 2640 --checkpoint-every 1

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --method domain_randomization --parent "$CBWCE_PARENT_PILOT" \
  --pilot --steps 2640 --checkpoint-every 1

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --method fixed_wce --parent "$CBWCE_PARENT_PILOT" \
  --pilot --steps 2640 --checkpoint-every 1 --wce "$CBWCE_WCE_PILOT"

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --method online_wce --parent "$CBWCE_PARENT_PILOT" \
  --pilot --steps 2640 --checkpoint-every 1 --wce "$CBWCE_WCE_PILOT"
```

**论文实验命令：** 每次选择一条。每条精确增加 1,320,000 个学习步，最终达到 2,320,000 步。这些命令不是自动实验队列。

```bash
python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --method baseline --parent "$CBWCE_PARENT" \
  --gate "$CBWCE_GATE"

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --method random_group --parent "$CBWCE_PARENT" \
  --gate "$CBWCE_GATE"

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --method domain_randomization --parent "$CBWCE_PARENT" \
  --gate "$CBWCE_GATE"

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --method fixed_wce --parent "$CBWCE_PARENT" \
  --gate "$CBWCE_GATE" --wce "$CBWCE_WCE"

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --method online_wce --parent "$CBWCE_PARENT" \
  --gate "$CBWCE_GATE" --wce "$CBWCE_WCE"
```

**脚本：** 全部命令共用 `experiments/runner.py` 和 `agents/controller.py`；仅需求选择与 WCE 更新不同。**输出：** baseline 使用 `runs/revised/`；其他四种使用 Grid 的 `output_coevolution/revised/` 或 Monaco 的 `output_coevolution_real/revised/`，后接 `<network>/<controller>/seed_<seed>/<method>/<run_id>/`。

**完成条件：** `result.json` 记录正确的累计学习步数与 `stage_simulation_steps`。固定 WCE 参数不变；论文在线续训增加 1,000 次 WCE 更新，从预训练 500 次达到累计 1,500 次。`checkpoint_001320000` 表示本阶段追加步数，不是累计学习步数。评估每种方法的准确最终控制器，不要把父模型当作等预算基线。

### 检查点选择、停止与恢复

默认 `--checkpoint-every 10` 每十个完整回合及阶段结束保存；`--checkpoint-every 1` 每个完整回合保存。父模型还会在精确预算截止点，对部分 on-policy 批次正确 bootstrap／加掩码后保存。1,000,000 步父模型包含 757 个完整的 1,320 步回合，另加 760 个转移；对于 120 步批次，最后剩 40 个真实样本。填充项不产生学习损失，也不增加环境转移。

从检查点状态和 `result.json` 读取计数器，不能只看文件名。保留 `checkpoint_*` 上级运行目录中的来源 `manifest.json`；只移动检查点目录会丢失当前加载器要求的来源身份。保留完整包，不要仅保存 TensorFlow 权重。只读取受信任的本地产生的 pickle 状态。

停止后保留中断尝试并关闭其拥有的 SUMO 连接。恢复从最后一个完整检查点加载，在新输出目录中开始新的回合。检查点之后的计算被丢弃，但原尝试／计时记录保留。不支持恢复回合中间的 SUMO 状态。没有完整检查点时，应重启该阶段。阶段、方法、路网、控制器、种子、目标预算、父模型／WCE 身份必须相同；原阶段总预算不是恢复后再追加的预算。

**在线 WCE 的试运行和论文恢复示例：**

```bash
export CBWCE_RESUME_PILOT='REPLACE_WITH_COMPLETE_SAME_STAGE_PILOT_CHECKPOINT'
python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --method online_wce --parent "$CBWCE_PARENT_PILOT" --wce "$CBWCE_WCE_PILOT" \
  --pilot --steps 2640 --checkpoint-every 1 --resume "$CBWCE_RESUME_PILOT"

export CBWCE_RESUME='REPLACE_WITH_COMPLETE_SAME_STAGE_PUBLICATION_CHECKPOINT'
python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --method online_wce --parent "$CBWCE_PARENT" --wce "$CBWCE_WCE" \
  --gate "$CBWCE_GATE" --resume "$CBWCE_RESUME"
```

父模型恢复使用 `parent --resume ...`，不传父模型／WCE 参数；离线 WCE 恢复使用 `wce --parent ... --resume ...`，不传预训练 `--wce`。试运行应保留原 `--steps` 或 `--episodes`。不能重复使用中断运行的 `--output`。显式选取完整检查点；工作台“最后检查点”指所选任务内的检查点，不是全目录最新文件搜索。

## 8 评估、计时与完成检查

### 阶段 IV 固定评估输入

**必须执行的检查：** C01、C02、C04、C06–C10。评估期间禁用控制器和 WCE 的模型参数学习。

针对每个路网，在十一种已见需求配置和十二种新场景上评估准确的 2,320,000 步控制器检查点。每个场景使用十次配对 rollout，仿真时域为 3,600 秒，初始路网为空，不延长仿真以排空剩余车辆。主要平均指标包含启动阶段。IA2C/MA2C/PPO 使用采样动作，IQL 使用贪心动作，每次 rollout 重置独立策略随机流。

评估前生成并保存完整需求文件。对于给定场景和到达实现，在所有方法与控制器种子间复用同一需求，并使用规定的 SUMO 种子。将实际插入、待发车辆和残留交通作为结果记录。在此共同评估中，WCE 不对留出的测试需求进行自适应调整。

### 阶段 IV 命令、脚本与输出层次

**目的／前提：** 使用冻结交通文件评估一个显式选定的最终控制器。每条轨迹必须满足 C01／C02／C09／C10。采用已完成续训的 `result.json.checkpoint`，并保持路网／控制器／训练种子一致。只设置适用的变量：

```bash
export CBWCE_FINAL_PILOT='REPLACE_WITH_SELECTED_COMPLETED_PILOT_CONTINUATION_CHECKPOINT'
export CBWCE_FINAL='REPLACE_WITH_SELECTED_COMPLETED_PUBLICATION_CONTINUATION_CHECKPOINT'
```

**试运行命令：** 对 1.25× 高峰进行一次完整 3,600 秒评估，模型可以只有试运行学习步数。

```bash
python main.py experiment evaluate --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --parent "$CBWCE_FINAL_PILOT" --pilot --suite test --scenario peak_1.25 --rollouts 1
```

**论文实验命令：** 控制器必须精确具有 2,320,000 个学习步，每场景十次实现。该命令评估一个控制器，不是整个研究。CLI 评估会检查检查点身份／预算，但不像训练阶段一样强制要求／校验 `--gate`，因此先执行显式 gate 检查。本地启动器也要求论文评估具有有效 gate。

```bash
python main.py experiment check --gate "$CBWCE_GATE"
python main.py experiment evaluate --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --parent "$CBWCE_FINAL" --gate "$CBWCE_GATE" --suite all --rollouts 10
```

**套件选择：** `--suite seen` 为十一种已见需求；`--suite test` 为十二种新场景；`--suite validation` 为六种独立验证场景；`--suite all` 仅包含 seen + test。`--scenario` 选择套件内的准确 ID，例如 `peak_1.25` 或 `peak_1.5`。试运行 `--rollouts` 可为 1–10，非试运行套件必须为十次。验证集示例：

```bash
python main.py experiment evaluate --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --parent "$CBWCE_FINAL_PILOT" --pilot --suite validation --rollouts 1
```

**单交通文件试运行／重试：** 不传 `--suite`，显式提供完整交通文件及对应的配对仿真／策略种子。下例对应套件 rollout 索引 0（到达种子 51001、SUMO 种子 61001）。其他重复应使用原尝试清单中的交通文件、种子和检查点。直接评估器默认策略种子 `71001` 不等于套件派生策略种子，因此配对重试不能依赖该默认值。

```bash
export CBWCE_ARTIFACT="$CBWCE_DATASET_ROOT/test/artifacts/peak_1.25_51001.json"
export CBWCE_POLICY_SEED="$(python -c 'import os; from experiments.core import digest; print(int(digest(["evaluation-policy", int(os.environ["CBWCE_PILOT_SEED"]), 0])[:8], 16))')"
python main.py experiment evaluate --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --parent "$CBWCE_FINAL_PILOT" --pilot --artifact "$CBWCE_ARTIFACT" \
  --sumo-seed 61001 --policy-seed "$CBWCE_POLICY_SEED"
```

**脚本：** [experiments/cli.py](experiments/cli.py) 枚举套件，通过 [experiments/scenarios.py](experiments/scenarios.py) 生成／复用交通文件。[experiments/runner.py](experiments/runner.py) 运行冻结控制器，导出测量和行程摘要；[experiments/core.py](experiments/core.py) 校验准确时间戳。所选续训的方法由来源清单推导，因此即使不显式传入 `--method`，评估 `online_wce` 也会正确标记结果。

**输出与完成条件：**

```text
runs_eval/revised/<network>/<controller>/seed_<seed>/evaluate/<run_id>/
  suite.json
  suite_result.json                         # 仅完整套件结束时写入
  demand_runtime/                           # 路径解析的启动／行程记录
  <seen|test|validation>/<scenario>/rollout_01/attempt_001/
    manifest.json
    environment.json
    progress.jsonl
    rollout.npz
    rollout.jsonl
    rollout.controls.jsonl
    rollout_summary.json
    result.json
    runtime/
```

一个所选 `all` 套件包含 230 条完整轨迹。每条要求 `status=complete`、`sample_count=3600`、准确的 1…3600 时间戳、正确检查点／需求哈希及实际仿真种子。单文件评估直接在其运行目录下保存轨迹文件，不包含套件层次。失败尝试保留 `result.json`，若已有测量则保留 `incomplete_attempt.*`，不能生成有效性能摘要行。

训练 `--resume` 不是跳过已完成评估的套件恢复功能。新的套件尝试从所选场景／轨迹的起点重新执行。若只重试某条失败实现，使用其单交通文件与准确配对种子，并在评估根目录内指定新的 `--output`。不能覆盖 `attempt_001`，也不能把重复有效尝试当作额外重复样本。

### 新场景与种子分配

<a id="traffic-generation"></a>

#### 1. 从各路网的十一种 OD 需求配置开始

验证与测试交通是**由已有 OD 需求率配置构造的合成场景**，不是新采集的实测交通，也不是由训练后的 WCE 生成。每条 OD 记录包含起点道路边、终点道路边，以及车辆／小时的请求需求率。两个路网分别应用相同的生成流程：

| 路网 | 原始配置来源 | 准备后数据集根目录 | 参考总需求率 |
|---|---|---|---:|
| Grid | `data_traffic/demand_<name>.csv` | `data_traffic/revised/` | 3,000 veh/hour |
| Monaco | `real_net_subnet/demand_groups/<name>.csv` | `real_net_subnet/demand_groups/revised/` | 2,383.3333 veh/hour |

对于每种原始配置，[original_profiles](experiments/demand.py) 将所有 OD 需求率乘以 `reference_total / original_total`。这样保留其 OD 比例，同时使同一路网的十一种配置具有相同总需求率。[prepare](experiments/prepare.py) 写入独立的归一化训练 CSV 和有序哈希清单，原始 CSV 不变。使用的道路网络分别是 `large_grid/data/exp.net.xml` 与 `real_net_subnet/data/in/most.net.xml`。

**顺序很重要：** Grid 按十一种配置名称的字母顺序排列；Monaco 使用 `experiments/demand.py` 中的 `ORDER`：`N_to_S`、`S_to_N`、`W_to_E`、`E_to_W`、`NW_to_SE`、`SE_to_NW`、`SW_to_NE`、`NE_to_SW`、`Periphery_to_Center`、`Center_to_Periphery`、`Uniform`。解释十一维混合权重时应读取对应的 `train/manifest.json`。两个路网复用相同数字种子，并不代表具有相同 OD 分配、路径或车辆计划。

#### 2. 构造固定的 3,600 秒场景定义

[definitions](experiments/scenarios.py) 创建时序计划，[block_rows](experiments/scenarios.py) 将每个区间转换为 OD 需求率。每个场景完整覆盖 `[0,3600)` 秒且没有间隙。以下评估区间长度可以不同于 WCE 训练中的 600 秒区间。

| 类别 | 测试场景／生成种子 | 验证场景／生成种子 | 构造与用途 |
|---|---|---|---|
| OD 重分配 | `redistribution_0.25`、`redistribution_0.5`、`redistribution_0.75` / `41001–41003` | `redistribution_0.35`、`redistribution_0.65` / `31001–31002` | 对 Uniform 配置的 OD 需求率乘以具有所列对数空间标准差的对数正态因子，再归一化至参考总量；在固定负载下测试新的空间分配。 |
| 凸组合混合 | `mixture_1`、`mixture_2`、`mixture_3` / `41004–41006` | `mixture_1`、`mixture_2` / `31003–31004` | 每个场景抽取一个 `Dirichlet(1,…,1)` 向量混合十一种配置，并在整个 3,600 秒内保持该混合不变。 |
| 时间切换 | `switch_300`、`switch_900`、`switch_1200` / `41007–41009` | `switch_450` / `31005` | 从 `N_to_S` 开始，按所列秒数与 `W_to_E` 交替，至 3,600 秒结束；保持总需求率，测试新的时间变化。 |
| 高峰强度 | `peak_1.1`、`peak_1.25`、`peak_1.5` / `41010–41012` | `peak_1.15` / `31006` | `[0,1200)` 使用参考率 Uniform；`[1200,2400)` 乘以所列倍数；`[2400,3600)` 恢复参考率，测试更高强度。 |

OD 重分配中，设归一化 Uniform 需求率为 $u_i$，路网参考总需求率为 $R$：

$$
z_i\sim\operatorname{Lognormal}(0,\sigma),\qquad
r_i=R\frac{u_i z_i}{\sum_j u_j z_j}.
$$

零需求率仍为零，不会因此创建新的 OD 连接。这里 sigma 是底层正态分布的标准差，不是交通数量的变异系数。混合场景中，OD 对 $i$ 的需求率为 $r_i=\sum_{g=1}^{11}w_g r_{g,i}$；缺少的 OD 记录贡献为零，重复 OD 对累加，且 $\sum_g w_g=1$。混合属于 WCE 可以访问的需求族，因此应称为组合泛化，不能自动认定为分布外需求。

时间切换和高峰计划是确定性的：其 `generation_seed` 是记录标识，不用于随机选择切换／高峰参数。OD 重分配和混合类别才实际使用该种子生成随机因子／权重。测试设置目前写在 `experiments/scenarios.py`；验证设置与种子列表记录在 [protocol.json](config/revised/protocol.json)。训练或调参前应冻结生成后的定义。

#### 3. 将需求率转换为完整车辆计划

场景定义还不是车辆列表。[artifact_for](experiments/scenarios.py) 创建新的 `RandomState(arrival_seed)`，并按顺序在所有区间间传递该随机流。对起始时间为 $s$、持续 $d$ 秒的区间中每个正需求率 OD 对，[materialize](experiments/demand.py) 执行：

1. 通过 SUMO `findRoute(..., vType='type1')` 在该路网中解析路径，保存有序道路边。无法到达或端点不一致的路径会被拒绝，即使本次车辆抽样数可能为零也要先验证。环境缓存重复 OD 对的路径；控制器不选择这些路径。
2. 从 $N\sim\operatorname{Poisson}(r\,d/3600)$ 抽取车辆数。需求率表示期望值，不是严格固定的车辆数量。
3. 在区间内均匀抽取出发时间，加上均值为 0、标准差为 **2 秒**的正态扰动，裁剪至 `[s+0.01, s+d−0.01]`，并保留两位小数。因此最终出发分布包含扰动与裁剪，并非未处理的均匀样本。
4. 从均值 **1.0**、标准差 **0.1** 的正态分布抽取每辆车的 `speed_factor`，若不为正则重新抽取。该量是无量纲速度因子，不是以 m/s 表示的目标速度。
5. 保存带区间前缀的唯一车辆 ID、出发时间、OD 端点、完整路径和速度因子；按出发时间与 ID 排序。

例如 Grid `peak_1.25` 在三个 20 分钟区间内分别请求 3,000 → 3,750 → 3,000 veh/hour。一小时车辆数的期望为 **3,250**，但已有 `peak_1.25_51001.json` 实际安排了 **3,335** 辆车。Monaco 相应为约 2,383.3333 → 2,979.1666 → 2,383.3333 veh/hour，期望为 **2,581.9444** 辆，对应已保存文件安排 **2,632** 辆。这些是已有输入中的计划数量，不是性能结果，也不是保证能实际插入路网的数量。

#### 4. 区分种子用途并复用配对交通文件

| 种子用途 | 数值 | 作用 |
|---|---|---|
| 场景生成 | 测试 `41001–41012`；验证 `31001–31006` | 空间因子／混合权重，或确定性时序计划的标识 |
| 到达实现 | 每个路网的每个场景使用 `51001–51010` | 泊松车辆数、出发时间扰动及速度因子；每场景十份完整文件 |
| SUMO 评估 | `61001–61010`，按 rollout 索引配对 | 固定计划交通后的仿真器随机性；物化使用种子为 `61001` 的空路网路径解析会话 |
| 控制器策略 | `experiments/cli.py` 根据训练种子与 rollout 索引推导 | 仅用于动作抽样，不重新生成交通文件 |

到达种子列表在不同场景、数据划分和路网之间复用；当前实现**没有**为验证与测试到达分配完全不重叠的随机流。它们的场景定义／参数与文件相互分离。OD 记录或需求率改变时，相同种子可以产生不同车辆计划。公平比较应在同一路网／场景／到达实现下，为所有方法和控制器种子复用完全相同的交通文件哈希。即使计划交通相同，拥堵仍可能改变实际插入与完成情况。

验证用于开发与参数选择；最终测试结果不得用于选择参数或检查点。测试中的 `mixture_1`、`mixture_2` 与验证集同名场景不同，因为种子与划分目录不同。不要将验证／测试文件放入训练输入。十一项 `seen` 场景各将一种归一化训练配置重复一小时，并使用留出的到达实现；`--suite all` 指十一项已见加十二项测试，不含验证集。

#### 5. 生成、定位并检查文件

以下命令准备输入；物化会启动 SUMO 解析路径，但不训练控制器。它们是使用说明，不是本次文档更新期间执行的命令：

```bash
python main.py experiment prepare
python main.py experiment prepare --materialize --network grid
python main.py experiment prepare --materialize --network monaco
```

在任一准备后数据集根目录下：

```text
train/<profile>.csv
train/manifest.json
validation/validation_scenarios.json
validation/artifacts/<scenario>_<arrival_seed>.json
test/test_scenarios.json
test/seen_scenarios.json
test/artifacts/<scenario>_<arrival_seed>.json
```

每路网包含 **60 验证 + 120 测试 + 110 已见 = 290 份交通文件**，两个路网**共 580 份**。已见文件实际位于 `test/artifacts/`，但带有 `scenario.split = seen`。每次生成尝试还会在 `runs_eval/revised/preparation/<run_id>/manifest.json` 写入文件索引。数据路径不包含控制器类型／方法，因为它们是共用的外生输入。

场景 JSON 包含 `id`、`network`、`split`、`family`、`generation_seed`、`horizon` 和 `blocks`。交通 JSON 还包含路网／配置／场景哈希、`arrival_seed`、`vehicles` 及自身内容 `hash`；每辆车包含 `id`、`depart`、`origin`、`destination`、`edges` 和 `speed_factor`。这些是保存为 JSON 的完整车辆计划，不是每个测试场景各生成一个新 CSV。评估时，`envs/experiment_env.py` 通过 TraCI 插入已保存的路径和车辆。

`prepare` 拒绝内容冲突的已准备文件。已有交通文件只有在哈希、唯一车辆 ID 及预期元数据匹配后才会复用。这些复用检查不会重新执行所有可能的路径／结构验证；路径解析与时序检查在生成阶段执行。输入改变时，应保留旧文件并为方案／输出位置建立新版本，而非覆盖或静默复用。本次文档更新期间，对**全部 580 份已有交通文件**进行了只读内容哈希／ID 检查，并确认其场景定义与当前定义一致；没有重新生成交通文件，也没有启动 SUMO 仿真。

`Real_Life_Monaco` 在路网／路径来源问题解决前不纳入主测试套件：审查描述了 272 条边，而当前子网有 270 条边，并且一个正需求 OD 对未通过静态连通性检查；不得静默丢弃该流量。已有 `demand_5x5_noisy` 配置约为 13,787–17,379 veh/hour，应单独作为极端负载分析。不要因为有效测试场景造成拥堵而删除它们。

### 每次 rollout 的记录与不确定性

为每个完整 rollout 导出一条汇总记录，包括路网、控制器、方法、训练种子、场景与数据划分、检查点哈希、需求哈希、全部种子、策略模式、时域、样本数和实际耗时。保留排队与交通时间序列、车道测量和行程记录。

记录平均排队、排队积分和峰值排队；按车辆时间加权的平均速度；预定、已插入、已完成、待发和剩余车辆数；车辆传送、碰撞及执行失败；以及已完成行程的旅行时间、等待时间、时间损失和发车延迟，并明确各指标的统计分母。完全没有车辆的 rollout，其平均速度应标记为不可用，不能人为赋值为零。不完整尝试按照 C09–C10 单独保留并可识别。

对于每个训练种子和场景，分别计算十个有效 rollout 的 $J_Q$，然后报告均值和样本标准差。对于每个路网、控制器和方法组合，在每个训练种子内，以相等场景权重平均十二个新场景的估计值。报告得到的五个种子级数值、均值、样本标准差和 95% t 置信区间：

$$
\bar J\pm 2.776\,\frac{s}{\sqrt{5}}.
$$

进行方法比较时，先计算五个匹配训练种子的差值，再计算差值的区间。不得将逐秒样本视为独立重复，也不得将不同控制器类型或路网混入同一个五种子估计。分别报告已见配置表现和四类新需求场景结果。探索性的“已测试最差场景均值”定义为：先对训练种子与 rollout 求平均，再取各场景均值的最大值。区间描述的是给定固定测试套件条件下的不确定性。[Agarwal 等关于 RL 评估的研究](https://arxiv.org/abs/2108.13264)

### 高峰需求热力图

对于预先指定的 1.25× 和 1.50× 高峰，在相同的 `[1200, 2400)` 秒窗口内聚合车道排队。生成五种方法的绝对值地图，以及 `online_wce − comparator` 差值地图，并对匹配的训练种子和 rollout 实现取平均。

同一路网与场景内使用统一的绝对值色标；差值色标以零为中心且对称。保持相同几何形状和空间范围，明确单位，未监测道路用灰色表示。负差值表示在线 WCE 的排队更低。将监测范围标注为受控进口车道，并核对地图聚合值与相同排队测量的一致性。不得为不同控制器选择不同高峰窗口或需求场景。

### 训练耗时与实验数量

使用单调时钟测量初始控制器训练、离线 WCE 训练、续训、初始化与重置、推理、控制器与 WCE 优化、检查点与日志开销，以及失败或丢弃的工作。分项计时相加时避免重叠；阶段总耗时与组成明细分别报告。记录硬件、线程数和并发负载。

对于 `baseline`、`random_group` 和 `domain_randomization`，独立训练成本包括父模型训练与续训。对于 `fixed_wce` 和 `online_wce`，还需计入离线 WCE 训练，即使整个实验实际为两者复用了同一个预训练 WCE。不得由控制器训练步数相同推断实际耗时相同。

| 工作项 | 数量 |
|---|---:|
| 共同父模型训练运行 | 40 |
| 离线 WCE 训练运行 | 40 |
| 包含基线的续训运行 | 200 |
| 最终评估 rollout | 46,000 |

评估数量为 `2 × 4 × 5 × 5 × 23 × 10 = 46,000`：路网数 × 控制器类型数 × 训练种子数 × 方法数 × 场景数 × rollout 次数。试运行、验证运行和失败尝试另计，并单独记录。

### 阶段 V 报告与本地工作台

**目的／前提：** 将显式选定的完整记录转换为科学表格／图表。使用有效清单、原始车道测量和原交通文件；报告生成会校验其来源。生成报告时控制器不学习。

**试运行命令：** 下例使用已有试运行证据创建新报告。仅在示例目标尚不存在时运行，否则选择另一个新名称。

```bash
python main.py experiment report \
  --input runs_eval/revised/peak_validation \
  --input runs_eval/revised/verification/integrated_20260914/grid_iqll/baseline \
  --input runs_eval/revised/verification/integrated_20260914/monaco_iqll/baseline \
  --output output_result/revised/pilot_report_20260915_example
```

**论文实验命令模板：** 采用选定的论文实验尝试目录。为其他方法、训练种子、父模型或 WCE 运行重复添加 `--input`。报告器按各行来源清单分类，`report` 不接受 `--pilot`。不要选择包含同一比较单元的多个有效重试的大范围目录。

```bash
export CBWCE_EVAL_RUN='REPLACE_WITH_ONE_SELECTED_EVALUATION_SUITE_OR_ATTEMPT_DIRECTORY'
export CBWCE_TRAIN_RUN='REPLACE_WITH_SELECTED_COMPLETED_CONTINUATION_RUN_DIRECTORY'
export CBWCE_REPORT_DIR='output_result/revised/REPLACE_WITH_NEW_REPORT_ID'
python main.py experiment report --input "$CBWCE_EVAL_RUN" \
  --input "$CBWCE_TRAIN_RUN" --output "$CBWCE_REPORT_DIR"
```

**脚本：** [experiments/reporting.py](experiments/reporting.py) 收集记录、检查配对、统计摘要并导出排队曲线和高峰图。**输出：** 全部报告 CSV／JSON／PNG／SVG 位于选定报告目录，该目录必须是新的。重复／未配对观测将显式失败并写出 `rejected.json`；其他被拒绝摘要另列表。没有轨迹摘要的失败尝试保留在原失败记录中，不一定出现在被拒绝摘要表中。

**完成条件：** 检查有效轨迹数、预期／缺失单元、配对需求／SUMO／策略种子，以及热力图车道总量。论文 95% 区间需要五个规定训练种子均完整；现有子集均值不是完整研究估计。热力图使用五种方法共同拥有的 `(seed, arrival_seed)` 配对交集；论文使用前必须检查配对数和完整性。`figs/revised/` 是另行保留的图表副本，不是 `report` 自动写入的第二个目录。

**已有试运行输出：** [试运行表格与导出](output_result/revised/pilot_integrated_final_report/)、[dashboard JSON](output_result/revised/pilot_integrated_final_report/dashboard.json)、[图表副本](figs/revised/pilot_integrated_final_report/)。其中包含 20 条高峰轨迹，不是完整论文实验。训练曲线来自抽样进度，不是平滑回合统计或评估性能曲线。

**工作台命令：** 同一工作台选择其中一个端口，不同时运行两条：

```bash
python main.py experiment dashboard
python main.py experiment dashboard --port 8766
```

按所选端口打开 [English](http://127.0.0.1:8765/) 或 [中文](http://127.0.0.1:8765/zh.html)。[experiments/dashboard.py](experiments/dashboard.py) 使用当前 Python 可执行文件启动任务，一次一个。选择阶段、试运行／论文模式、路网、控制器、种子、方法及明确兼容的检查点，先预检查再启动。停止保留中断尝试；恢复创建另一尝试。服务重启不会自动恢复训练。任务 request／start／finish／recovery JSON 和 `console.log` 位于 `runs_eval/revised/jobs/<job_id>/`；普通 CLI 默认把控制台输出写到终端，除非另行捕获。

绘图时打开 Results，选择“载入本地结果导出”，读取生成的 `dashboard.json`。筛选路网／控制器／试运行状态，并明确选择训练记录或高峰地图。[英文 HTML](docs/site/dist/index.html)、[中文 HTML](docs/site/dist/zh.html)和[私有 Sites 指南](https://cb-wce-training-workspace.loyal-bowl-4834.chatgpt.site)提供说明及导出结果。只有本地服务可启动训练；本地导入报告不会自动重新发布私有网站。

### 论文实验完成检查表

- [ ] C01–C10 每项均有通过的验证记录及支持证据。
- [ ] 每个路网与控制器类型均有五个独立父模型，并保存完整状态。
- [ ] 每个 WCE 均指向正确的冻结 1,000,000 步父模型。
- [ ] 五种续训方法均在 2,320,000 个控制器训练步完成，且控制器更新频率一致。
- [ ] 冻结控制器与固定 WCE 的模型参数保持不变；在线 WCE 的模型参数在学习过程中发生变化，并记录了规定的更新。
- [ ] 配对比较使用匹配的测试需求文件和种子，并已验证训练与测试分离。
- [ ] 只有完整 rollout 进入汇总，所有失败尝试均可查。
- [ ] 在规定车道集合与缩放下，排队奖励、评估指标和热力图一致。
- [ ] 对每个路网、控制器和方法分别报告 rollout 变异性与训练种子不确定性。
- [ ] 运行数量、检查点标识、不可变清单和计时记录相互一致。

### 双语术语

| English | 简体中文 |
|---|---|
| Baseline training | 基线训练 |
| Controller-learning step | 控制器训练步 |
| Frozen-controller simulation step | 冻结控制器仿真步 |
| Training episode | 训练回合 |
| Demand group | 交通需求组 |
| Demand-mixture weights | 需求混合权重 |
| Model parameters | 模型参数 |
| Domain randomization | 域随机化 |
| Checkpoint | 检查点 |

英文和中文指南定义同一套流程。更新任一版本时，必须同步方法 ID、配置键、路径、数值、修正标识和公式。

## 9 实现脚本索引与数据文件字典

| 脚本 | 职责 |
|---|---|
| [main.py](main.py) → [experiments/cli.py](experiments/cli.py) | 公共 `experiment` 命令、参数解析与套件调度 |
| [experiments/protocol.py](experiments/protocol.py) / [protocol.json](config/revised/protocol.json) | 共用设置与默认输出根目录；生效 INI 位于 `config/revised/` |
| [experiments/prepare.py](experiments/prepare.py) / [experiments/demand.py](experiments/demand.py) | 明确训练顺序、归一化配置、哈希与交通实现 |
| [experiments/scenarios.py](experiments/scenarios.py) | 已见／测试／验证定义与配对交通文件 |
| [experiments/runner.py](experiments/runner.py) | 统一父模型／WCE／续训／评估交互循环 |
| [agents/controller.py](agents/controller.py) / [agents/recurrent.py](agents/recurrent.py) | 策略动作、指纹、带掩码批次、更新时机和循环状态 |
| [agents/wce.py](agents/wce.py) / [agents/policies.py](agents/policies.py) | WCE 适配器与已有策略架构 |
| [envs/experiment_env.py](envs/experiment_env.py) / [experiments/core.py](experiments/core.py) | SUMO 步进、排队／奖励测量、随机流和运行记录 |
| [experiments/checkpoint.py](experiments/checkpoint.py) | 完整检查点包、完整性、恢复与来源运行身份 |
| [experiments/reporting.py](experiments/reporting.py) | 科学数据收集、统计、训练曲线、高峰图和导出 |
| [experiments/dashboard.py](experiments/dashboard.py) | 本地预检查、启动／状态／停止、运行目录和报告 |
| [experiments/verify.py](experiments/verify.py) / [tests/test_corrections.py](tests/test_corrections.py) / [tests/test_workflow.py](tests/test_workflow.py) / [tests/test_visualization.py](tests/test_visualization.py) | 集成准入与确定性正确性测试 |

历史 `main.py train/evaluate`、对抗／协同进化脚本和评估脚本保留原路径及行为。`revision.*` 是兼容包装，不是另一套修正实现。本研究使用上方 `experiment` 接口。

### 输入、训练与评估记录

Grid 的 `<dataset>` 为 `data_traffic/revised`，Monaco 为 `real_net_subnet/demand_groups/revised`。`<run>` 为阶段运行目录，`<attempt>` 为单次评估目录。JSONL 每行一个完整 JSON 对象，JSON 则是一个完整文档。CSV 有表头；成本组件等嵌套报告值在 CSV 单元格内以 JSON 编码。

| 文件／生成者 | 位置／格式 | 字段与单位 | 用途 |
|---|---|---|---|
| 训练 CSV／`prepare` | `<dataset>/train/<profile>.csv` · CSV | `origin_edge`、`dest_edge`、`veh_per_hour`；单位为车辆／小时，每行一个 OD | 控制器／WCE 需求输入 |
| 训练清单／`prepare` | `<dataset>/train/manifest.json` · JSON list | 有序条目：`name`、`source`、`source_hash`、`original_total`、`scale`、`prepared`、`prepared_hash`；索引中的 `rows` 为 null | 核对原流量、归一化因子、顺序与输入完整性 |
| 场景定义／`prepare` | `<dataset>/test/{seen,test}_scenarios.json`; `<dataset>/validation/validation_scenarios.json` · JSON lists | `id`、`network`、`split`、`family`、`generation_seed`、`horizon`、`blocks`；时间单位秒，乘数／权重无量纲 | 固定空间／时间场景构造 |
| 交通文件／场景物化器 | `<dataset>/<test\|validation>/artifacts/<scenario>_<arrival_seed>.json` · JSON | 路网／配置／场景哈希、`horizon`、`arrival_seed`、`scenario`、`vehicles`、`hash`；车辆字段 `id`、`depart`、`origin`、`destination`、`edges`、`speed_factor` | 完整计划交通；车辆数等于 `vehicles` 长度；出发时间为秒，速度因子无量纲；已见需求文件也位于 test |
| 运行身份／`RunRecord` | `<run>/manifest.json` · JSON | 阶段／方法／种子、包含 `visualization` 的 CLI 输入、父检查点路径／哈希、训练配置、配置／源代码哈希、运行环境信息、`manifest_hash` | 不可变实际输入，将结果与来源关联 |
| 环境身份／运行器 | `<run>/environment.json` · JSON | `assets`、`lanes`、`nodes`、`node_lanes`、`neighbors`、`configuration` | 车道顺序与路网身份；包含复制设置及历史字段 |
| 完成记录／`RunRecord.finish` | `<run>/result.json` · JSON | `status`、`manifest_hash`、`wall_seconds`、`component_seconds`；训练另含 `checkpoint`、`learning_steps`、`stage_simulation_steps`、`backward_calls`、`minibatch_updates_per_agent`、`wce_updates`；失败含错误／已完成计数 | 权威完成／成本记录；耗时单位为实际秒 |
| 进度／运行器 | `<run>/progress.jsonl` · JSONL | `stage`、`episode`、`simulation_steps`、`learning_steps`、`goal`、`mean_queue`、`wce_updates`、`wall_seconds` | 每 120 个联合转移一条；排队均值覆盖最近 600 秒仿真；不是每次优化一条，也不是回合均值 |
| 需求决策／运行器 | `<run>/demand_decisions.jsonl` · JSONL | `episode`、`block`、十一项 `weights`、`wce_reward`、`completed_seconds`、`scheduled_vehicles`、`traffic_hash` | 每区间一条；episode／block 从零编号；父模型最后不完整区间的 WCE 奖励为 null；训练记录哈希／数量，而不是完整车辆文件 |
| 车道数组／`export_episode` | `<run>/episode_0001.npz` or `<attempt>/rollout.npz` · compressed NumPy archive | `time`：形状 `(T,)`，秒 1…T；`lanes`：形状 `(L,)`，有序车道 ID；`queue`：形状 `(T,L)`，各车道停车数 | 完整训练回合 T=6600；评估 T=3600；Grid L=150／Monaco L=116；部分父模型回合／不完整记录可更短 |
| 交通序列／`export_episode` | Same episode/rollout stem + `.jsonl` · JSONL | `time`、`queue`、`active`、`speed_sum`、`inserted`、`completed`、`pending`、`teleports`、`collisions` | 每秒仿真一条；数量为车辆数，`speed_sum` 为活动车辆速度之和，速度单位 m/s |
| 学习奖励序列／`export_episode` | Same episode/rollout stem + `.controls.jsonl` · JSONL | `time`、按控制器节点顺序排列的 `learner_rewards` 向量 | 每五秒仿真一条；已按控制器类型转换并除以 100 |
| 轨迹结果／评估器 | `<attempt>/rollout_summary.json` · JSON | 排队指标、速度及分母、计划／插入／完成／剩余／待出发车辆、事件、已完成行程指标、需求哈希、实际 SUMO 种子、检查点身份 | 排队单位车辆；积分排队为车辆·秒；速度 m/s；行程／等待／时间损失／出发延误为秒。方法／训练／策略种子关联 manifest 读取 |
| 运行时记录／环境 | `<run>/runtime/startup_<attempt>.json`, `trips_<attempt>.xml`, SUMO assets | 启动命令、实际 `visualization`、种子／版本来源；SUMO `tripinfo` 车辆记录 | 从启动命令确定匹配行程文件；已完成行程均值排除未完成行程并给出分母 |

`simulation_steps` 表示五秒联合转移，不是秒。进度中的 `episode` 从一开始，需求决策中的 `episode` 从零开始；NPZ 文件名从 `episode_0001` 开始。最后的部分回合可能在两次进度写入之间完成，因此完成状态以 `result.json` 和检查点计数器为准。离线 WCE 的控制器训练步曲线保持常数，而仿真步数与 WCE 更新数增加。

完整评估交通文件保存全部车辆路径／出发时间／速度因子。方法之间配对的是计划需求；拥堵可改变实际插入、待出发车辆和完成行程。训练需求使用独立随机流逐区间生成并记录哈希；哈希本身不是已保存的车辆计划。

### 检查点包与编号

| 论文阶段 | 最终目录 | 控制器训练步数 |
|---|---|---|
| 父模型 | `checkpoint_001000000` | 1,000,000 |
| 离线 WCE | `checkpoint_000660000` | 仍为 1,000,000 |
| 续训 | `checkpoint_001320000` | 2,320,000 |

```text
<run>/manifest.json
<run>/checkpoint_<nine-digit-stage-steps>/
  manifest.json
  runner.pkl
  controller/
    variables.index
    variables.data-00000-of-00001
    checkpoint
    state.pkl
  wce/                         # 仅阶段包含 WCE 时存在
    variables.index
    variables.data-00000-of-00001
    checkpoint
    state.pkl
```

检查点清单记录 `version`、模型 `signatures`、文件哈希、`parents` 与包 `hash`，最后写入以标记完整性。TensorFlow 文件包含模型和优化器变量。控制器 `state.pkl` 保存真实学习／调度／更新计数、pending／replay 数据、回放游标、随机流和循环状态；WCE 状态保存待处理宏转移、宏转移／更新计数和随机流。`runner.pkl` 保存阶段／方法、目标、阶段步数、回合、初始学习计数及运行器随机流。pickle 是 Python 专用状态，不是通用 CSV；通过检查点管理器加载，不要仅根据文件名重建模型。

### 报告输出与计时解释

| 报告目录内的文件名 | 含义 |
|---|---|
| `rollouts.csv` | 每方法／训练种子／场景／到达实现一条有效观测；含指标与来源 |
| `scenarios.csv` | 每训练种子／场景的十轨迹均值、样本标准差及完整性；试运行行保留标识 |
| `seeds.csv` | 每个规定论文训练种子的等权场景均值，seen／test 分开 |
| `comparison.csv`, `paired_differences.csv` | 五种子均值／标准差／95% 区间，以及在线减比较方法的配对差值；缺失区间保持不可用 |
| `demand_family_summary.csv`, `worst_tested_scenario.csv` | 满足规定完整记录后才生成论文需求家族统计与最差测试场景 |
| `computation_costs.csv`, `standalone_pipeline_costs.csv` | 阶段耗时，以及父模型 + 续训 + 必需离线 WCE 成本；父记录不可用时流程总耗时不可用 |
| `rejected.csv`; `rejected.json` on duplicate/pairing failure | 被拒绝的现有摘要；未生成摘要的失败还需检查原尝试记录 |
| `dashboard.json` | 轨迹、场景／种子比较、曲线、地图、家族、最差情况和成本；本地 HTML 导入格式 |
| `training_<index>.png/.svg` | 保存的进度排队与控制器训练步曲线；若选择冻结 WCE 阶段，其控制器步数保持常数 |
| `<network>_<controller>_<scenario>_<pilot\|publication>_<method>.png/.svg` | 高峰绝对地图；差值图名称使用 `online_minus_<comparator>` |

`wall_seconds` 是本次尝试通过单调时钟记录的阶段总耗时。当前 `component_seconds` 在相关操作发生时包含 `reset`、`demand_selection`（含 WCE 推理）、`demand_generation_insertion`、`controller_inference`、`simulation_measurement`、`controller_learning`、`wce_learning` 和 `checkpoint`。这些是已测量部分，不是完整分解：最终 flush／日志及其他开销未全部单独计时。不能宣称已有独立完整的日志或 WCE 推理计时账本，也不能用组件之和代替总耗时。按上文要求另列失败／中断尝试和共用离线 WCE 成本。

### 不训练模型，读取现有轨迹

下面的只读示例使用已有 Grid/IQL 试运行。NPZ 时间戳表示每个一秒区间的终点；`time > 1200` 且 `time <= 2400` 对应物理窗口 `[1200,2400)`。无需启动新实验。
```python
import json
from pathlib import Path
import numpy as np

attempt = Path("runs_eval/revised/verification/integrated_20260914/grid_iqll/eval_baseline")
with np.load(str(attempt / "rollout.npz"), allow_pickle=False) as data:
    lane_ids = data["lanes"]
    time = data["time"]
    queue = data["queue"]
    assert len(time) == 3600
    mean_total_queue = float(queue.sum(axis=1).mean())
    peak_window = (time > 1200) & (time <= 2400)
    mean_peak_lane_queue = queue[peak_window].mean(axis=0)
with (attempt / "rollout.jsonl").open() as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
summary = json.loads((attempt / "rollout_summary.json").read_text())
assert np.isclose(mean_total_queue, summary["mean_queue"])
vehicle_seconds = sum(row["active"] for row in rows)
mean_speed = (sum(row["speed_sum"] for row in rows) / vehicle_seconds
              if vehicle_seconds else None)
print(queue.shape, mean_total_queue, mean_speed)
```
