# 03 五种续训与断点恢复任务

[单机自动编排与英文控制台](05_local_orchestration_zh.md)

Online 分支分别画控制器与 WCE 的训练奖励，仅控制器参与外生需求验证。固定 WCE 仅推理，不产生 WCE 学习曲线。

> 任务要求；默认正式训练 seed=101，开发 seed=9001。运行状态与证据见 [实施记录](../implementation_status.md)。本文不表示正式训练已完成。

## 目标与分叉方法

两地图×三算法×seed101×五方法，共30个续训分支。每个地图／算法的五分支从同一完整best父状态开始；fixed和online还从同一个完整best离线WCE开始。正式上游必须成功退出、完整校验且early_stopped。

| 方法 | 每600秒需求生成 | Controller更新 | WCE更新 |
|---|---|---|---|
| baseline | 原11组固定顺序，独热权重 | 是 | 无 |
| random_group | 11组中均匀随机抽一组 | 是 | 无 |
| domain_randomization | Dirichlet(1,…,1)混合11组 | 是 | 无 |
| fixed_wce | 已选WCE在线观察并采样需求权重 | 是 | 否，参数冻结 |
| online_wce | 同一初始WCE在线观察并采样 | 是 | 每完整回合一次 |

fixed冻结的是参数，仍根据当前状态和自身RNG输出动作，不是预先固定一条需求序列。各分支环境、state/action/reward、网络与优化配置沿用[baseline任务](01_baseline_training_zh.md)；WCE结构沿用[WCE任务](02_wce_training_zh.md)。不会额外加入等待时间或扩大网络。

## 新增预算与RL更新频次

每回合6600秒＝1320个联合控制步，五分支使用相同的回合上下限与停止规则，实际训练步数可不同。一个学习步是全网5秒控制，不乘路口数，也不等于一次optimizer更新。

| 项目 | MA2C | PPO | DQL |
|---|---|---|---|
| 更新触发 | 每120个连续控制步 | 每120步，四轮整序列 | warm-up后每4步 |
| 每路口数据量 | 120个有效样本 | 同一120样本重复四轮 | 均匀replay抽64个终点，各最多16步历史 |
| 完整1320步回合更新次数 | 11 | 44 | warm-up结束后330 |
| 记忆 | episode内LSTM64 | episode内LSTM64 | 有限16步窗口LSTM64 |

Online WCE另在11宏步后更新一次；fixed WCE零次。保持不同算法原有batch与梯度阈值，不为了数字一致改算法；同一算法五分支必须一致。报告交互量、优化次数、耗时及整个进程树内存，不声称同交互量就是同计算量。

DQL分叉保留online/target、replay和优化器，epsilon从0.1在132000新增步内降到0.01；分叉只执行一次。恢复保留该日程及其origin，不能重新变成0.1。其余控制器继续父状态的优化器和学习调度。

## 选模、停止与恢复

父模型作为续训episode0的best候选，允许续训后没有改善。控制器始终用固定外生11×600秒DR序列、实现0验证；online不按变化对手上的训练reward选模。每100回合验证，最少1000、最多5000回合；patience=10，阈值max(anchor×0.01,0.1)，取消额外确认。

阶段结束时，从第 0 回合到实际停止回合的全部验证中选平均排队最低的控制器，冻结后测试。同分保留较早的 checkpoint。入选的可能是起始父模型，也可能是第 1000 回合后才出现的最佳模型。分别记录入选回合和实际完成回合。

完整checkpoint保存参数、优化器、replay／pending、循环推理历史、RNG、调度、累计步数和patience。Online保存同一时刻controller与WCE，不能拼接两个各自best。Fixed保存冻结WCE的采样状态；仅保存冻结参数不足以复现续训。`--resume`使用原目录latest完整checkpoint，首次分叉使用parent/wce best；两者不能混淆。

```bash
"$CBWCE_PYTHON" main.py continue --network grid --controller dql --seed 101 --method online_wce --parent "$CBWCE_PARENT_BEST" --wce "$CBWCE_WCE_BEST" --validation "$CBWCE_DATA_ROOT/grid_validation.json" --gate "$CBWCE_RUN_ROOT/verification/gate.json" --output "$CBWCE_RUN_ROOT/manual_grid_dql_online"
# 原目录恢复：同一命令加 --resume；baseline/random_group/domain_randomization不传 --wce
```

## 保存与频次

| 内容 | 写入频次 | 保留要求 |
|---|---|---|
| `manifest.json`、来源与父模型身份 | 运行开始 | 不修改既有身份；数据、缓存、运行结果放项目外 |
| `blocks.jsonl.gz` | 每600秒需求块 | 权重、traffic hash、车辆数、原始排队和聚合诊断；每回合11条 |
| `episodes.jsonl.gz` | 每完整回合 | 排队、吞吐、耗时、步数；trainer RSS不是整个任务内存 |
| `validation.jsonl.gz` | episode 0、每100回合、阶段停止时 | 保留每次总分、区间分数、候选哈希及停止诊断；同一终点不重复验证 |
| 完整 recovery | 每10回合、每次验证、正常分段／结束 | 最新两个；参数、优化器、replay、RNG、计数和patience |
| best | 验证分数严格改善时 | 当前完整best与推理快照；同分保留较早者 |
| 验证候选快照 | 每次验证 | 最近五个轻量候选；不能用轻量快照恢复训练 |
| 阶段成果 | 阶段结束 | final、被下游引用的祖先、必要manifest和摘要；归档先校验 |

训练不默认保存逐秒车道、每步reward、每回合NPZ或tripinfo。训练摘要的completed沿用SUMO原始到达计数；正式评估用TripInfo区分正常完成和移除，不能直接将训练吞吐摘要当作最终完成率。正常信号请求在回合边界保存；再次强制中断丢弃不完整回合，从完整恢复点重跑。记录按attempt区分，离线导出时剔除回滚后的重复区间。`CBWCE_DATA_ROOT`存共享输入，`CBWCE_RUN_ROOT`存运行，`CBWCE_CACHE_ROOT`存可重建缓存；另一块磁盘／NAS才构成独立备份。

保存入选控制器的推理快照供测试，并保留完整 best 供后续训练使用。下游引用的祖先固定保留。归档包含引用清单，以及分支仍在使用的父模型和 WCE。

## 本地多任务安排

30分支按依赖释放。每个父完成后，baseline/random_group/DR可与离线WCE并行；fixed/online等待对应WCE。优先让不同地图／算法交错运行，不强制等一组五分支全部结束。Online资源类单独实测，包含两个可训练模型和验证worker。

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

## 验收与将来扩展种子

检查五分支父哈希一致、fixed/online WCE哈希一致、模型结构和超参数一致、首次DQL分叉及恢复日程正确；比较不中断和恢复后的下一更新／最终张量，验证停止时选中了整个验证历史中的 best。异常、无学习和预算耗尽保持可见。

以后需要202／303时，新建扩展campaign，保持科学配置及旧目录身份，不改旧manifest：

```bash
"$CBWCE_PYTHON" main.py local prepare --output "$CBWCE_RUN_ROOT/extended_queue" --extend "$CBWCE_RUN_ROOT/formal_queue" --training-seeds 101 202 303 --validation-grid "$CBWCE_DATA_ROOT/grid_validation.json" --validation-monaco "$CBWCE_DATA_ROOT/monaco_validation.json" --gate "$CBWCE_RUN_ROOT/verification/gate.json"
```

同一run root下，已有成功receipt和有效身份／checkpoint的seed101任务被复用；新增两个种子需各完成完整矩阵，不只补效果好的方法。科学代码、输入或运行环境变化后不能冒充同一批次恢复。

依据：[runner](../../experiments/runner.py)、[DQL分叉](../../agents/dql.py)、[checkpoint](../../experiments/checkpoint.py)、[停止规则](../../experiments/selection.py)、[本地队列](../../experiments/campaign.py)。英文：[对应任务](03_continuation_training_en.md)。
