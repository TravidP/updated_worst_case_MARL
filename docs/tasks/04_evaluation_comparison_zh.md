# 04 最终评估、数据保存与比较任务

[单机自动编排与英文控制台](05_local_orchestration_zh.md)

> 任务要求；默认正式训练seed=101，每场景保留10个评估实现。英文：[对应任务](04_evaluation_comparison_en.md)。实际执行证据见[实施记录](../implementation_status.md)，不把开发检查称为正式结果。

## 目标、准入与数量

比较两张地图 × 三种控制器 × 五种续训方法。每个分支只测试截至停止时验证平均排队最低的控制器。30 个模型 × 23 个场景，合计 690 个场景任务。每场景使用 10 个配对实现，共 6,900 次主评估，规定的审计另计。

模型必须已结束训练、有完整身份与可校验的选定checkpoint。正式运行须通过本地／容器runtime lock；父／WCE下游门禁仍执行early_stopped要求。测试只加载冻结控制器，不在线训练controller或WCE，不根据test挑checkpoint。初始父模型可做开发诊断，但主比较的baseline是相同停止规则下的baseline续训。

保留原测试的场景语义，使用新版独立manifest与随机流，不混合旧IA2C/IQLL、旧单seed数据或旧artifact。对每个network/scenario/realization，所有方法共享交通artifact、SUMO seed和策略随机流身份。训练seed不改变这套评估配对；MA2C/PPO按固定policy seed采样，DQL greedy。

## 23个场景的定义

每条rollout从空路网开始，完整3600秒、1秒统计、5秒控制（720次联合控制决策），没有warm-up或额外排空尾段。车辆可以跨块保留，需求切换不重置LSTM或路网。

| 类别 | 三个测试值／场景 | 生成规则 |
|---|---|---|
| Seen | 11个原始profile | 每个单独运行3600秒，按各地图基准总率归一化 |
| Mixture | mixture_1／2／3 | generation seed 41004／41005／41006产生Dirichlet(1,…,1)权重，按名称绑定11组，整小时固定 |
| OD redistribution | σ=0.25／0.5／0.75 | generation seed 41001／41002／41003；对Uniform正OD乘LogNormal(0,σ)，再归一化恢复总率；零OD保持零 |
| Switching | 300／900／1200秒 | generation seed 41007／41008／41009；每块从11组抽取，相邻不重复；分别12／4／3块 |
| Peak | 1.1／1.25／1.5 | generation seed标签41010／41011／41012；[0,1200)基准、[1200,2400)倍率、[2400,3600)恢复 |

Mixture是未见组合／插值，不称为严格分布外外推；peak改变需求强度；redistribution保持OD支持集；switching改变已见需求的时间排列。300秒是三档中较快的切换，不把900／1200秒描述为秒级扰动。两地图profile顺序不同，不能直接拿另一地图的权重数组位置解释需求名称。

测试manifest使用`test_suite_v1`，保存23场景精确定义、profile名称顺序／哈希、network和detector哈希、生成器哈希。arrival／SUMO／policy seed由manifest哈希、场景ID、实现索引0–9和流名称派生。车辆artifact包含route table、车辆出发时刻与speed factor，压缩共享；同一输入只生成一次。代码严格检查完整场景定义，删场景或换参数不能冒充原测试。

## 执行步骤

1. 冻结全部训练配置、验证设计和模型选择规则；正式campaign已自动写入两地图测试manifest。
2. 核对训练终止状态、best 推理快照、模型哈希、source/runtime 及 validation/test 哈希。记录入选回合和实际完成回合，便于解释各分支的训练时长差异。
3. 按模型和场景分任务，每任务顺序运行10次。每完成一条就原子保存压缩分片；失败记录原因，恢复校验已完成项后只补缺失项。
4. 对每个模型的seen_Uniform和peak_1.5第一个实现追加完整审计；与常规模式的所有确定性指标一致才通过。
5. 全部单元齐全后生成报告；缺场景、实现、方法或预期训练种子均报错，不静默当作完整实验。

```bash
# 单独生成测试清单；formal_queue中已生成的清单优先复用
"$CBWCE_PYTHON" main.py test-manifest --network grid --output "$CBWCE_DATA_ROOT/grid_test.json"
# 单场景任务；不传--scenario则评估完整23场景；恢复加--resume
"$CBWCE_PYTHON" scripts/evaluate_suite.py --run "$CBWCE_SELECTED_RUN" --manifest "$CBWCE_DATA_ROOT/grid_test.json" --selection best --scenario peak_1.5 --output "$CBWCE_RUN_ROOT/manual_eval_peak"
# 从 campaign 收集全部截至停止的 best 测试结果
"$CBWCE_PYTHON" scripts/report.py --campaign "$CBWCE_RUN_ROOT/formal_queue" --output "$CBWCE_RUN_ROOT/report_single_seed"
```

手动评估不要与本地训练队列同时绕过资源上限；正式批量采用同一个队列。输出目录必须新建或显式resume，禁止覆盖旧结果。缺失文件、损坏哈希和身份不匹配不通过重新采样掩盖。

## 指标与保存方案B

主指标`mean_queue`＝3600秒内每秒受控进口车道排队总数的平均，单位vehicles，越小越好。Grid150条、Monaco116条；不是每路口均值，也不是物理路网所有车道的统计。

| 内容 | 计算／写入频次 | 保存形式与要求 |
|---|---|---|
| mean/integrated/peak/p95 queue、queue variance | 每秒累计，rollout结束汇总 | 原值，不缩放不裁剪；integrated≈3600×mean |
| 最后600秒趋势 | 最后600个全网排队样本线性斜率 | vehicles/s；仅诊断，不控制选模 |
| mean speed、scheduled/inserted/arrived/completed/removed、pending/remaining、teleports/collisions | 每秒累计，结束汇总 | 保存速度分母active_vehicle_seconds；completion_rate=completed/inserted，分母0为null |
| 分钟曲线 | 每60秒 | 60点，区间mean和peak；不长期重复保存全程逐秒车道 |
| 车道窗口统计 | 在线累计 | 三个互不重叠的1200秒窗口：pre_peak、peak、post_peak；另存全程lane_full_mean |
| 完成车辆行程指标 | rollout结束、关闭SUMO后流式解析 | duration、waitingTime、timeLoss、departDelay的count/sums/means；成功保存后删除临时XML |
| 正式rollout分片 | 每完成一次 | `rows/<scenario>_<index>.json.gz`，含模型／测试／指标身份和payload哈希 |
| 完成索引 | 场景任务完成 | `result.json`列出分片SHA-256、数量和状态；manifest记录场景范围 |
| 完整审计 | 每个模型Uniform／peak_1.5实现0 | 压缩NPZ：逐秒车道、5秒动作和reward；另存审计哈希证明 |
| 汇总报告 | 全矩阵验收后 | CSV与summary.json，不能把失败项填0 |

指标schema为`evaluation_metrics_v2`。旧代码窗口0是全程，不能解释为峰前；新schema明确给出窗口名称、边界和样本数[1200,1200,1200]，旧缓存不复用。正式评估只临时保留3600个全网总排队float64供精确p95计算，约28.8KB；训练仍不保留完整逐秒矩阵。

四个行程指标分别是实际进入路网至到达的时间、速度≤0.1m/s的交通等待、相对理想行驶的时间损失、计划出发至实际插入的延迟。只统计在窗口内完成且未提前移除的车辆；保存完成样本分母，零完成时means为null。SUMO原始到达数保存为arrived；它可能包括以teleport结束的移除记录。completed改为TripInfo正常完成数，removed单独报告，completion_rate只用completed/inserted。TripInfo全部非负arrival记录数必须与原始arrived一致，否则报错。未完成和未出发车辆通过remaining/pending报告，不能因慢车未完成而把完成车辆均值变小解释为整体更好。定义依据：[SUMO TripInfo](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html)。

交通文件在`CBWCE_DATA_ROOT/<network>/traffic/`按内容身份共享；评估缓存放`CBWCE_CACHE_ROOT/final_evaluations/`，可重建；正式分片和审计在run root，不能只存在缓存里。模型、test、source/runtime或指标schema变化都使缓存失效。归档保留选定模型引用、测试manifest、交通哈希、分片、报告和checksums；先验哈希再备份至独立磁盘/NAS。

正式训练期间保留恢复点。30 个分支的 best 全部评估完成后，对已确定只用于推理的分支，归档每个分支入选的推理 checkpoint、上游引用、配置和日志。相同 checkpoint 按哈希保留一份。校验归档可读取、所有引用可解析后，再清理该分支无需恢复且没有被引用的完整状态及 replay。

## 对比策略与单种子解释

按地图和控制器分别比较，不将两地图的原始排队值混成一个排名。每种方法对baseline；online另对random_group、DR、fixed。差值定义`method−comparator`，负值更好；相对改善`100×(comparator−method)/comparator`，comparator均值0时为null。

每场景10对给出均值、样本SD、差值、相对改善、胜／平次数和固定随机种子10000次paired bootstrap 95%区间。场景族、seen/test内部按场景等权，对每个实现先汇总场景再配对；区间条件于这套固定场景和已经训练好的模型，不代表对所有可能需求的置信度。

当前`n_training_seeds=1`，`training_seed_sd=null`（CSV空白），不填0、不报告跨训练seed置信区间。十次评估衡量车辆／SUMO／策略采样随机性，不能称十个独立训练实验。以后扩展campaign明确列出全部种子，报告要求每个种子的完整矩阵，再计算跨seed均值和样本SD。

输出包括seeds.csv、comparison.csv、paired_differences.csv、paired_training_seeds.csv、auxiliary_metrics.csv和summary.json。评估 manifest 和 seeds.csv 分别用 best_episode、completed_episodes 记录入选回合和实际完成回合。车辆行程跨rollout按sum/count汇总；同时报告完成样本数。主排名依旧用排队，辅以吞吐、pending和teleport检查，避免把未插入或被移除的车辆当作改善。默认不增加显著性检验，也不改旧结果网站。

## 本地多任务安排

网页工作流按一任务一槽位并行，调参上限 12，正式队列上限 24，并受本机 CPU、内存与磁盘预算限制。详见[网页编排](05_local_orchestration_zh.md)。以下为直接使用独立 campaign 命令时的测量步骤。

30个入选模型×23场景＝690个场景任务；各自顺序完成10实现，任务与训练共用队列上限。单任务先测，再对已覆盖资源类比较2、4并发；默认CPU和每库1线程。评估任务依赖对应续训结束，相同模型哈希和完整评估身份可复用缓存。

先生成／复用共享交通，再并行不同模型或场景。原子交通写入和缓存锁防止并发写坏；独立临时目录和TraCI连接隔离SUMO。每5秒记录进程树RSS/CPU，预留至少20%内存并对测得峰值加1.3倍余量，检查审计及临时tripinfo空间。未经测量的资源类单独运行。

```bash
"$CBWCE_PYTHON" main.py local status --campaign "$CBWCE_RUN_ROOT/formal_queue"
"$CBWCE_PYTHON" main.py local resume --campaign "$CBWCE_RUN_ROOT/formal_queue" --max-parallel 2 --resources "$CBWCE_RESOURCES"
```

同一run root仅一个队列调度器可拥有资源锁；重启先核对活动worker和退出回执。评估失败保留已完成分片；排除失败原因后显式以`--resume`恢复对应评估目录，不按结果好坏挑选重跑。

## 附录：短时封路／路口阻断可行方案，尚不实施

| 候选 | 能测试的问题 | 实现与限制 |
|---|---|---|
| 单车道或短路段暂时限制通行 | 局部容量丢失、绕行及队列外溢 | 保存原权限并恢复；区分已在路段内车辆与后续进入车辆；改变权限未必令已有车辆立即停车 |
| 指定路口短时全红 | 信号服务短暂停止后的恢复 | 事件层在指定秒覆盖动作，期间记录controller请求与实际执行；结束恢复正常信号转换，避免被下一次5秒控制立即覆盖 |
| 临时限速／容量下降 | 更温和、可恢复的瓶颈 | 强度单独定义；不能把限速与硬封闭视为同一个实验 |

建议第一轮先研究单路口全红，因为可保持原有OD、路线和网络结构；这是待讨论建议，不是已冻结实验。数秒事件与5秒决策周期要分开，在每秒仿真层触发；可讨论5／10／20秒等候选，但本次不确定正式时长。地点、开始时刻、持续时间、是否允许绕行和恢复判据都需后续确定。

每个事件版本与无事件版本使用相同交通和随机实现配对，冻结事件manifest，观察局部／全网附加排队AUC、峰值、吞吐损失和解除后的恢复时间。记录事件开始/结束、受影响车道、实际信号／权限、pending和teleports；全红不等于路段物理封闭，改路径的实验应单列，不混入当前23场景主比较。

技术依据：[SUMO道路／车道限制](https://sumo.dlr.de/docs/Simulation/Rerouter.html#closing_a_lane)、[TraCI信号控制](https://sumo.dlr.de/docs/TraCI/Change_Traffic_Lights_State.html)。后续实现须针对锁定的SUMO版本实测恢复及车辆行为，不通过删除需求、跳过不可达OD或扩大teleport来制造改善。

## 验收与代码依据

验收23场景定义、名称绑定、需求总量、随机流配对、连续3600样本、60分钟点、窗口边界、行程分母、分片哈希、恢复不重复、完整矩阵及单seed空标准差。比较审计与精简输出的所有确定性指标；源码／环境／指标变更必须失效旧缓存。

依据：[场景](../../experiments/scenarios.py)、[worker](../../experiments/validation.py)、[指标](../../experiments/metrics.py)、[tripinfo](../../experiments/tripinfo.py)、[可恢复评估](../../experiments/evaluation.py)、[统计报告](../../experiments/reporting.py)、[队列](../../experiments/campaign.py)。
