# 02 离线 WCE 训练任务

[单机自动编排与英文控制台](05_local_orchestration_zh.md)

控制台显示每回合 11 个块的平均原始／learner reward 及单实现验证点。原始 WCE reward 是块平均排队，越高越好；较小的正改善也可刷新 best。

> 任务要求；默认正式训练 seed=101，开发 seed=9001。运行状态与证据见 [实施记录](../implementation_status.md)。本文不表示正式训练已完成。

## 前置条件与交付

每个“地图×控制器×seed101”的best父模型训练一个离线WCE，共6个。父模型必须通过正式环境／源码／输入检查，退出成功、状态early_stopped，且引用完整best checkpoint；WCE与父模型通过哈希绑定。no_learning／预算耗尽不自动放行。

父控制器仅推理，冻结网络、优化器和replay。环境沿用baseline的路网、6600秒回合、5秒控制及600秒需求间隔。WCE不直接控制信号，不改变地图或总基准流量，而是选择十一种归一化需求的混合。

## WCE的state、action、reward与网络

| 项目 | Grid | Monaco |
|---|---|---|
| 输入 | 25节点×6 wave＝150 | 28节点×11补齐wave＝308 |
| 编码 | 5×5×6；3×3 CNN 32 same→3×3 CNN 64 valid→flatten→FC128 | 对称含自环、度归一化邻接；GCN64→GCN64→节点max pooling→FC128 |
| 输出 | Gaussian μ11、可训练log_std11、V1，共享编码 | 同左 |
| 可训练参数 | 95671 | 14679 |
| 奖励除数 | 3000 | 1000 |

wave归一化／截断与控制器一致，按明确节点顺序补齐；没有等待时间或fingerprint输入。策略采样`z=μ+std×Normal(0,1)`，`weights=softmax(z)`，十一项非负且和为1，再按profile名称顺序混合。log_std初值-0.5。WCE策略不是Dirichlet，也不是LSTM。

每600秒取一次动作，reward为该区间每秒受控进口车道全网排队总数的平均，越高越好。奖励在learner边界除以表中数值并裁剪[-2,2]；不改变评估原始指标。

| 学习参数 | 值 |
|---|---|
| 优化器／学习率 | RMSProp／5e-4，constant |
| gamma／entropy／value coefficient | 1.0／0.01／0.5 |
| RMSProp alpha／epsilon | 0.99／1e-5 |
| 梯度范数上限 | 40 |
| 更新频次 | 11个宏步＝完整6600秒回合后更新一次 |
| Advantage normalization | true；11样本内总体std，epsilon=1e-8；只标准化actor advantage |

critic return和奖励尺度保持原定义。日志保存标准化前后advantage均值／标准差、loss、entropy、梯度和reward clipping。完整回合共11次需求动作、一次WCE更新，父控制器零次优化。

## 验证、停止与执行

在episode0、每100回合和阶段停止时，冻结候选WCE与父控制器，用1 个固定外生随机实现（索引 0）运行完整6600秒，最大化平均排队。**验证中WCE必须实际产生动作**，不能用控制器的固定DR序列替代。固定的是按区间／OD／随机量类型索引的随机流和需求约束，候选WCE产生的车辆轨迹可不同。

阶段最低预算为 500 回合，上限为 3000 回合。连续 10 次常规验证无实质改善才触发早停。第 0 回合建立基线，此后每 100 回合验证一次，因此最早在第 1000 回合触发早停。严格改善保存best，实质改善阈值max(anchor×0.01,0.1)，同分保留较早者。无学习与预算耗尽单独报告；只有early_stopped的同一个best WCE供fixed／online使用。

```bash
# 队列会自动解析并固定父best；手动单任务时须提供完整best路径
"$CBWCE_PYTHON" main.py wce --network grid --controller ppo --seed 101 --validation "$CBWCE_DATA_ROOT/grid_validation.json" --gate "$CBWCE_RUN_ROOT/verification/gate.json" --parent "$CBWCE_PARENT_BEST" --output "$CBWCE_RUN_ROOT/manual_grid_ppo_wce"
# 在原目录断点恢复，其他身份参数相同，添加 --resume
```

## 保存与频次

| 内容 | 写入频次 | 保留要求 |
|---|---|---|
| `manifest.json`、来源与父模型身份 | 运行开始 | 不修改既有身份；数据、缓存、运行结果放项目外 |
| `blocks.jsonl.gz` | 每600秒需求块 | 权重、traffic hash、车辆数、原始排队和聚合诊断；每回合11条 |
| `episodes.jsonl.gz` | 每完整回合 | 排队、吞吐、耗时、步数；trainer RSS不是整个任务内存 |
| `validation.jsonl.gz` | episode 0、每100回合、阶段停止时 | 保留每次总分、区间分数、候选哈希及停止诊断；同一终点不重复验证 |
| 完整 recovery | 每10回合、每次验证、正常分段／结束 | 最新两个；WCE参数、优化器、RNG、计数、patience和父策略采样状态 |
| best | 验证分数严格改善时 | 当前完整best与推理快照；同分保留较早者 |
| 验证候选快照 | 每次验证 | 最近五个轻量候选；不能用轻量快照恢复训练 |
| 阶段成果 | 阶段结束 | final、被下游引用的祖先、必要manifest和摘要；归档先校验 |

训练不默认保存逐秒车道、每步reward、每回合NPZ或tripinfo。训练摘要的completed沿用SUMO原始到达计数；正式评估用TripInfo区分正常完成和移除，不能直接将训练吞吐摘要当作最终完成率。正常信号请求在回合边界保存；再次强制中断丢弃不完整回合，从完整恢复点重跑。记录按attempt区分，离线导出时剔除回滚后的重复区间。`CBWCE_DATA_ROOT`存共享输入，`CBWCE_RUN_ROOT`存运行，`CBWCE_CACHE_ROOT`存可重建缓存；另一块磁盘／NAS才构成独立备份。

完整离线WCE恢复还保存冻结父控制器的采样RNG。冻结权重不表示策略不采样；漏掉这部分会改变恢复后的训练交通。WCE本身的RNG、pending、宏步和update计数随完整checkpoint保存。父模型只引用、不重复复制。fixed 和 online 分支继承截至停止时验证平均排队最高的离线 WCE。

## 本地多任务安排

6个离线WCE分别依赖对应父模型；父模型完成后即可启动，不等整批。每个资源类须计入冻结父控制器、可训练WCE、SUMO以及验证期间的额外推理进程。每次 WCE 验证仅执行一次实现 0，外层并行不同任务。

使用独立进程、输出目录、SUMO连接和随机流。统一使用本地队列，不能在训练队列外另开一批评估绕过资源上限。每任务默认TF intra/inter及BLAS/OpenMP/MKL线程均为1、CPU运行。

网页工作流按一任务一槽位并行，调参上限 12，正式队列上限 24，并受本机 CPU、内存与磁盘预算限制。详见[网页编排](05_local_orchestration_zh.md)。以下为直接使用独立 campaign 命令时的测量步骤。

先单任务测整进程树，再对已测量的任务比较2、4并发。资源类未测量时，调度器只允许该任务单独运行。离线WCE只加载父控制器推理图，不加载DQL replay，因此该阶段不要求满replay测量。预留至少20%总内存，并满足各活跃任务实测峰值之和×1.3不超过预算；同时检查CPU及写入新旧checkpoint的磁盘余量。默认磁盘余量10 GiB，可显式调高。队列每5秒采集RSS/CPU，状态变更立即保存。

```bash
"$CBWCE_PYTHON" main.py local run --campaign "$CBWCE_CAMPAIGN" --max-parallel 1
"$CBWCE_PYTHON" main.py local status --campaign "$CBWCE_CAMPAIGN"
# 停止队列后导出实际测量，再在足够覆盖的资源类上增加并发
"$CBWCE_PYTHON" main.py local export-resources --campaign "$CBWCE_CAMPAIGN" --output "$CBWCE_RESOURCES"
"$CBWCE_PYTHON" main.py local resume --campaign "$CBWCE_CAMPAIGN" --max-parallel 2 --resources "$CBWCE_RESOURCES"
```

`CBWCE_CAMPAIGN`是包含campaign.json的目录，`CBWCE_RESOURCES`是新的JSON文件路径。5秒采样可能漏掉极短峰值，1.3倍只是运行余量；不能用短pilot RSS冒充完整训练及单实现验证的资源证明。本地默认不按时长分段；可加`--segment-seconds`。正常分段自动续跑，连续三次无进度会停止；异常任务记录failed，不无限重试。重复运行目录由文件锁保护。

## 验收与代码依据

核对Grid150／Monaco308输入、动作权重和、总需求守恒、父参数不变、每回合恰好一次更新、max方向选模、每次验证只运行实现 0和恢复后的下一次采样／更新一致。成果为6个WCE最佳身份、绑定父哈希、验证历史和明确停止状态。

依据：[WCE包装与恢复](../../agents/wce.py)、[Gaussian CNN/GCN](../../agents/policies.py)、[训练循环](../../experiments/runner.py)、[验证worker](../../experiments/validation.py)、[WCE配置](../../config/revised/)。英文：[对应任务](02_wce_training_en.md)。
