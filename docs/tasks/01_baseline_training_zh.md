# 01 初始 baseline 训练任务

[单机自动编排与英文控制台](05_local_orchestration_zh.md)

控制器训练及验证均记录每路口、每控制步的平均即时 reward，控制台另画原始平均排队和历史验证 best。

> 任务要求；默认正式训练 seed=101，开发 seed=9001。运行状态与证据见 [实施记录](../implementation_status.md)。本文不表示正式训练已完成。

## 训练多少回合，什么时候停

| 单个任务 | 训练回合数 | 停止时点 |
|---|---:|---|
| 开发调参候选（seed 9001） | 固定 1000 | 24 个候选各自跑满 1000 回合，再按开发验证分数选参 |
| 正式 baseline 父模型（seed 101） | 1000–5000 | 第 0 回合做初始验证，以后每 100 回合验证一次。最早第 1000 回合早停，最晚第 5000 回合结束 |

**1 个 episode = 6600 个仿真秒 = 11 个需求块 = 1320 次联合控制决策。** 单个正式父模型共训练 132 万至 660 万个联合控制步，验证另计。要估算实际要跑几小时，可在目标机器上计时同一地图、算法和并发数下两个相邻常规验证结束点之间的 100 回合。用这段实际耗时乘以剩余回合数再除以 100，分别按最早可停点和预算上限估算。这样会把周期性验证和 checkpoint 写盘计入。

早停看验证集上的平均排队。相对上一次实质改善时的分数，累计下降至少达到 `max(参考分数 × 1%, 0.1 辆)` 才重置计数。连续 10 次验证都没有达到阈值，且已训练至少 1000 回合，就停止并选历史验证最优模型。**这 10 次可以在第 1000 回合之前开始累计。** 例如上次实质改善在第 800 回合，第 900–1800 回合的十次验证都未达到阈值，就在第 1800 回合停。更小的改善仍可更新 best。[训练协议](../../config/protocol.json) · [停止判断](../../experiments/selection.py)

## 目标、环境与交付

训练6个父控制器：Grid／Monaco × MA2C／PPO／DQL × seed 101。初始baseline使用11个原始归一化需求profile的固定顺序；后续baseline续训是另一个阶段，以相同停止规则下的 baseline 续训作为主对照。

Grid为5×5、25个受控路口；Monaco为当前修订版子网、28个受控路口，不切换到另一套修复网络。每回合6600秒＝1320个全路网联合控制步；SUMO步长1秒，控制周期5秒，已有相位转换逻辑分2秒黄灯区间和3秒绿灯区间，同相位保持按环境逻辑处理。每600秒注入下一需求块，不清空车辆和排队。Grid基准3000 veh/h，Monaco 2383.3333 veh/h；实际车辆数是随机实现。

State使用本地与邻居wave；Grid来自检测区域车辆数，Monaco来自车道车辆数，norm_wave=5、clip_wave=2。MA2C邻居wave乘coop_gamma=0.9，并加入邻居上一策略概率向量去掉最后一项的fingerprint。PPO/DQL也观察邻居wave，但无fingerprint。不输入等待时间。Action为每路口可用相位列表的离散索引，由既有环境转换成信号状态，不假设所有路口动作维度相同。

| 参数 | MA2C | PPO | DQL |
|---|---:|---:|---:|
| 优化器／学习率起点 | RMSProp／5e-4 | Adam／3e-4 | Adam／1e-4 |
| gamma | 0.99 | 0.99 | 0.99 |
| wave编码／LSTM | 128／64 | 128／64 | 128／64 |
| 额外输入 | 邻居fingerprint→64 | 无fingerprint | 无fingerprint |
| 数据与更新 | 连续120步，更新一次 | 连续120步，整序列更新4轮 | replay均匀抽64个终点，各最多16步历史 |
| 更新间隔 | 每120联合控制步 | 每120联合控制步 | warm-up 5000步后每4步 |
| entropy／PPO ratio clip | 0.01／— | 0.01／0.2 | — |
| 梯度范数上限 | 40 | 40 | 10 |
| Grid奖励除数 | 2000 | 3000 | 3000 |
| Monaco奖励除数 | 200 | 1000 | 1000 |

MA2C/PPO为独立actor与critic；保留episode内LSTM状态，需求切换时不reset。MA2C采用n-step return，不引入GAE。DQL采用Huber TD loss、独立online/target、容量50000联合转移的均匀replay；target每1000个环境学习步同步，有限窗口从零循环状态开始，padding由mask排除。无Double/Dueling/prioritized replay。初训epsilon从1在500000步内线性降至0.01；评估greedy。MA2C/PPO评估保持带固定policy seed的采样。

奖励只在learner边界除一次并裁剪到[-2,2]；MA2C为本地负排队加0.9倍邻居负排队，PPO/DQL为复制到各路口的全局负排队。控制器取5秒控制端点排队，WCE／评估累计每秒排队，不能混为同一采样频率。MA2C/PPO在有效样本上做advantage normalization；PPO ratio clip与梯度裁剪分别报告。

## 实际接口与参数量

| 地图／算法 | 每路口state维度 | 每路口动作数 | online可训练参数（全网） | 每路口参数范围 |
|---|---|---|---:|---|
| grid / ma2c | 26/36/46 | 5 | 3,511,190 | 137,990–142,086 |
| grid / ppo | 18/24/30 | 5 | 2,647,830 | 104,070–107,142 |
| grid / dql | 18/24/30 | 5 | 1,327,165 | 52,165–53,701 |
| monaco / ma2c | 6–50 | 2/3/4/5/6 | 3,848,314 | 133,571–143,045 |
| monaco / ppo | 5–34 | 2/3/4/5/6 | 2,900,986 | 100,547–108,101 |
| monaco / dql | 5–34 | 2/3/4/5/6 | 1,452,638 | 50,306–54,148 |

Grid MA2C 的角点、边缘非角点、内部路口分别有 2、3、4 个邻居。每个路口有 6 个本地 wave 输入，每个邻居提供 6 个 wave 和 4 个上一策略概率，因此输入维度分别为 26、36、46。

维度范围不表示其中每个整数都出现。完整逐路口state/action/fingerprint与参数量见[机器可读清单](model_inventory.json)，可用`python scripts/inspect_models.py --output <新JSON路径>`复核。DQL target与online等量，但target不训练；Grid／Monaco replay数组容量50000时分别为263250000／198600000字节，未包含TF和SUMO。

## 开发调参：24个候选是什么

所有命令均在项目根目录执行；运行前先完成后文“本地环境与正式启动”中的解释器与路径环境设置。

正式训练前，开发seed 9001对每个“地图×算法”运行四种参数组合，每个1000完整回合，共24个开发任务、24000回合，不能与6个正式父任务混算。

| 算法 | 学习率两档 | 另一个参数两档 | 每地图候选数 |
|---|---|---|---:|
| MA2C | 5e-4、2.5e-4 | entropy 0.01、0.001 | 4 |
| PPO | 3e-4、1.5e-4 | entropy 0.01、0.001 | 4 |
| DQL | 1e-4、3e-4 | replay 50000、20000 | 4 |

每组按开发验证best平均排队选择，精确同分使用候选tie_rank较小者（默认起点优先）。选出六套参数后，写入版本化配置、重新验收并冻结正式campaign。五种续训沿用对应配置。开发manifest不得使用正式选模或最终测试的随机流。

```bash
# 在项目根目录；两个开发设计种子须显式指定，与正式设计隔离
"$CBWCE_PYTHON" main.py validation-manifest --network grid --design-seed "$CBWCE_DEV_GRID_SEED" --output "$CBWCE_DATA_ROOT/grid_dev.json"
"$CBWCE_PYTHON" main.py validation-manifest --network monaco --design-seed "$CBWCE_DEV_MONACO_SEED" --output "$CBWCE_DATA_ROOT/monaco_dev.json"
"$CBWCE_PYTHON" main.py local prepare-tuning --output "$CBWCE_RUN_ROOT/tuning_queue" --validation-grid "$CBWCE_DATA_ROOT/grid_dev.json" --validation-monaco "$CBWCE_DATA_ROOT/monaco_dev.json"
"$CBWCE_PYTHON" main.py local run --campaign "$CBWCE_RUN_ROOT/tuning_queue" --max-parallel 1
"$CBWCE_PYTHON" main.py local select-tuning --campaign "$CBWCE_RUN_ROOT/tuning_queue" --output "$CBWCE_DATA_ROOT/selected_configs"
"$CBWCE_PYTHON" main.py local prune-tuning --campaign "$CBWCE_RUN_ROOT/tuning_queue" --selection "$CBWCE_DATA_ROOT/selected_configs"
```

选参命令要求24个候选各完成1000回合；导出六份INI和selection.json，不静默覆盖源配置。将胜出INI纳入`config/revised`后重新运行验收。开发任务显式标记pilot，只承担开发职责，不能作为正式父模型进入下游。

选参导出成功后，运行上面的 `prune-tuning`。它重新核对 24 个完整候选的验证分数、六个胜者及导出 INI 的 SHA-256，再清理这 24 个候选的 checkpoint（含 DQL replay）。保留全部候选配置、训练和验证日志、结果、选参清单及六份 INI。清理回执在队列目录的 `tuning_cleanup.json`，记录清理前的文件字节数；中断后可重跑同一命令。需要用开发任务导出资源测量时，请在清理之前运行 `export-resources`。

## 本地环境与正式启动

使用独立Python 3.10环境，按`packaging/requirements.lock`安装并校验：TF CPU 2.15.1、NumPy 1.26.4、SUMO/TraCI/sumolib 1.26.0及完整锁定依赖。以下为执行模板，不表示当前机器已匹配：

```bash
python3.10 -m venv .venv
export CBWCE_PYTHON="$PWD/.venv/bin/python"
"$CBWCE_PYTHON" -m pip install --require-hashes -r packaging/requirements.lock
export CBWCE_ASSETS_ROOT="$PWD/assets"
export CBWCE_DATA_ROOT="$(dirname "$PWD")/cbwce_data"
export CBWCE_RUN_ROOT="$(dirname "$PWD")/cbwce_runs"
export CBWCE_CACHE_ROOT="$(dirname "$PWD")/cbwce_cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_ENABLE_ONEDNN_OPTS=0
export PATH="$PWD/.venv/bin:$PATH"
export CBWCE_RUNTIME_LOCK="$CBWCE_DATA_ROOT/runtime.local.json"
"$CBWCE_PYTHON" main.py runtime-lock --output "$CBWCE_RUNTIME_LOCK"
"$CBWCE_PYTHON" scripts/verify.py --extended --output "$CBWCE_RUN_ROOT/verification"
```

SUMO需要本机TraCI端口。正式环境锁记录解释器、依赖、SUMO二进制SHA-256、系统/CPU及有效线程；完整gate需与当前环境和代码一致。缺失／开发版依赖只能做开发检查，不伪造容器锁或用pilot绕过正式门禁。

```bash
"$CBWCE_PYTHON" main.py validation-manifest --network grid --design-seed "$CBWCE_VALIDATION_GRID_SEED" --frozen --output "$CBWCE_DATA_ROOT/grid_validation.json"
"$CBWCE_PYTHON" main.py validation-manifest --network monaco --design-seed "$CBWCE_VALIDATION_MONACO_SEED" --frozen --output "$CBWCE_DATA_ROOT/monaco_validation.json"
"$CBWCE_PYTHON" main.py local prepare --output "$CBWCE_RUN_ROOT/formal_queue" --validation-grid "$CBWCE_DATA_ROOT/grid_validation.json" --validation-monaco "$CBWCE_DATA_ROOT/monaco_validation.json" --gate "$CBWCE_RUN_ROOT/verification/gate.json"
```

正式设计种子必须明确选定；冻结后不重新生成。prepare只生成队列，不启动训练；默认包含6父模型、6WCE、30续训及690个逐场景评估任务，每个续训分支测试截至停止的历史best。可加`--training-only`先只准备42个训练任务。所有变量都指向本次新目录，既有运行不覆盖。

## 验证、选模与停止

控制器使用提前生成、按profile名称绑定的11个DR混合块，各600秒，固定顺序连续运行6600秒；只在整个序列开始清空路网和循环状态。常规只用固定实现0。开发与正式选模使用不同manifest和随机流，最终test不参与调参。

主分数是每秒所有受控进口车道排队总数的全程平均，越低越好；Grid为150条、Monaco为116条，不能称作全路网所有车道。指标不缩放、不裁剪，同分保留较早模型。

父模型触发早停时，曾有实质改善记为 `early_stopped`，从未有实质改善记为 `no_learning`。达到预算上限时优先判断早停；未触发早停且曾有实质改善记为 `max_budget_reached`。只有early_stopped上游自动放行下游，其余需检查。提前停止不是已证明收敛。

## 保存与频次

建议保留以下摘要频次。每个需求块持续 **600 个仿真秒（120 个控制步）**，结束时向同一个压缩日志追加一行。这样可以比较 11 种需求各自的排队情况。一次连续运行到 1000 回合产生 11,000 条块记录和 1000 条回合记录；到 5000 回合分别为 55,000 条和 5000 条，均写入同两个日志文件。模型和优化器的大文件按下表的 recovery、best 规则单独保存。

| 内容 | 写入频次 | 保留要求 |
|---|---|---|
| `manifest.json` | 新建运行时写一次 | 记录地图、算法、种子、配置和源码身份；恢复时读取并校验 |
| `blocks.jsonl.gz` | 每个 600 仿真秒的需求块结束时 | 权重、traffic hash、车辆数、原始排队和聚合诊断；每回合11条 |
| `episodes.jsonl.gz` | 每完整回合 | 排队、吞吐、耗时、步数；trainer RSS不是整个任务内存 |
| `validation.jsonl.gz` | episode 0、每100回合、阶段停止时 | 保留每次总分、区间分数、候选哈希及停止诊断；同一终点不重复验证 |
| 完整 recovery | 每10回合、每次验证、正常分段／结束 | 最新两个；参数、优化器、replay、RNG、计数和patience |
| best | 验证分数严格改善时 | 当前完整best与推理快照；同分保留较早者 |
| 验证候选快照 | 每次验证 | 最近五个轻量候选；不能用轻量快照恢复训练 |
| 阶段成果 | 阶段结束 | final、被下游引用的祖先、必要manifest和摘要；归档先校验 |

训练不默认保存逐秒车道、每步reward、每回合NPZ或tripinfo。训练摘要的completed沿用SUMO原始到达计数；正式评估用TripInfo区分正常完成和移除，不能直接将训练吞吐摘要当作最终完成率。正常信号请求在回合边界保存；再次强制中断丢弃不完整回合，从完整恢复点重跑。记录按attempt区分，离线导出时剔除回滚后的重复区间。`CBWCE_DATA_ROOT`存共享输入，`CBWCE_RUN_ROOT`存运行，`CBWCE_CACHE_ROOT`存可重建缓存；另一块磁盘／NAS才构成独立备份。

## 本地多任务安排

开发阶段最多24个独立候选；配置冻结后为6个独立父任务。先测Grid/DQL及Monaco/MA2C等不同资源类，再据实测并行，不能把所有任务一次启动。父任务完成时可放行对应下游，不必等待另外五个父模型。

使用独立进程、输出目录、SUMO连接和随机流。统一使用本地队列，不能在训练队列外另开一批评估绕过资源上限。每任务默认TF intra/inter及BLAS/OpenMP/MKL线程均为1、CPU运行。

网页工作流按一任务一槽位并行，调参上限 12，正式队列上限 24，并受本机 CPU、内存与磁盘预算限制。详见[网页编排](05_local_orchestration_zh.md)。以下为直接使用独立 campaign 命令时的测量步骤。

先单任务测整进程树，再对已测量的任务比较2、4并发。资源类未测量、或DQL未覆盖replay填满时，调度器只允许该任务单独运行。预留至少20%总内存，并满足各活跃任务实测峰值之和×1.3不超过预算；同时检查CPU及写入新旧checkpoint的磁盘余量。默认磁盘余量10 GiB，可显式调高。队列每5秒采集RSS/CPU，状态变更立即保存。

```bash
"$CBWCE_PYTHON" main.py local run --campaign "$CBWCE_CAMPAIGN" --max-parallel 1
"$CBWCE_PYTHON" main.py local status --campaign "$CBWCE_CAMPAIGN"
# 停止队列后导出实际测量，再在足够覆盖的资源类上增加并发
"$CBWCE_PYTHON" main.py local export-resources --campaign "$CBWCE_CAMPAIGN" --output "$CBWCE_RESOURCES"
"$CBWCE_PYTHON" main.py local resume --campaign "$CBWCE_CAMPAIGN" --max-parallel 2 --resources "$CBWCE_RESOURCES"
```

`CBWCE_CAMPAIGN`是包含campaign.json的目录，`CBWCE_RESOURCES`是新的JSON文件路径。5秒采样可能漏掉极短峰值，1.3倍只是运行余量；不能用短pilot RSS冒充满replay资源证明。本地默认不按时长分段；可加`--segment-seconds`。正常分段自动续跑，连续三次无进度会停止；异常任务记录failed，不无限重试。重复运行目录由文件锁保护。

## 验收与代码依据

验收包括两地图实际state/action维度、网络参数量、奖励只缩放一次、有效样本mask、更新计数、验证不改变训练、完整恢复和轻量拒绝恢复。记录6个best父模型及状态；no_learning／预算耗尽单独报告。

依据：[控制器](../../agents/controller.py)、[DQL](../../agents/dql.py)、[环境](../../envs/experiment_env.py)、[配置](../../config/revised/)、[调参](../../experiments/tuning.py)、[队列](../../experiments/campaign.py)。英文：[对应任务](01_baseline_training_en.md)。
