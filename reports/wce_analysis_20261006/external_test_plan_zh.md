# Hangzhou 与 Monaco 外部需求：定位、独立测试及结果站点导入方案

日期：2026-10-06。本文是实施方案；本次完成文件定位和静态检查，没有启动训练、正式 rollout 或修改结果网站。英文版：[external_test_plan_en.md](external_test_plan_en.md)。机器可读检查：[external_demand_audit.json](external_demand_audit.json)。

## 1. 这两组在哪里，当前能否直接测试？

| 项目 | Grid / Hangzhou 候选 | Monaco / MoST 重建需求 |
|---|---|---|
| 本地需求文件 | [data_traffic/demand_5x5_sparse.csv](../../data_traffic/demand_5x5_sparse.csv) | [real_net_subnet/demand_groups/Real_Life_Monaco.csv](../../real_net_subnet/demand_groups/Real_Life_Monaco.csv) |
| 旧版 Group 12 入口 | [eval_signal_controllers.py](../../eval_signal_controllers.py)，列表最后一项 | [eval_signal_controllers_real.py](../../eval_signal_controllers_real.py)，列表最后一项 |
| 原始数据证据 | 未找到本地杭州 4×4 原始输入及其 5×5 映射脚本 | [real_net/data/in/most_0.rou.xml](../../real_net/data/in/most_0.rou.xml) |
| 当前路网 | [large_grid/data/exp.net.xml](../../large_grid/data/exp.net.xml) | [real_net_subnet/data/in/most.net.xml](../../real_net_subnet/data/in/most.net.xml) |
| 正需求 OD 数量 | 140 | 14 |
| CSV 总需求 | 2,983 veh/h | 2,383.333332 veh/h |
| 有向连接图静态检查 | 140/140 可达 | 13/14 可达 |
| 正式外部测试的前置条件 | 证明杭州来源；完成 SUMO 车辆类别与路线检查 | 先解决不可达 OD 与 272/270 边路网审计不一致 |

论文原句位于 [paper/main.tex](../../paper/main.tex) 第 591 行。上述 Grid 文件是旧评估器指定的 Group 12，因此是最直接的本地候选；**旧列表、文件名称和约 3,000 veh/h 的总流量，都不能独立证明它来自杭州**。

杭州 4×4 原始数据可从 CoLight 作者仓库的 [data/Hangzhou/4_4](https://github.com/wingsweihua/colight/tree/master/data/Hangzhou/4_4) 查找。其中有 `roadnet_4_4.json`、`anon_4_4_hangzhou_real.json`、`anon_4_4_hangzhou_real_5734.json` 和 `anon_4_4_hangzhou_real_5816.json`。仍需确定论文究竟使用哪一个版本，不能任意选择后声称复现了原实验。

Monaco 上游是 [MoSTScenario 作者仓库](https://github.com/lcodeca/MoSTScenario)。本地证据能证明 CSV 是从本地 `most_0.rou.xml` 重建的需求，但还需补充上游版本、下载来源和本地截取过程，才能完成端到端溯源。

当前 revised 主评估使用 11 个 seen + 12 个生成 test 场景；这两个旧 Group 12 **没有作为同名外部场景进入该主评估定义**。论文所写的“11+1”与现有主评估的“11+12”应在论文中明确区分。

## 2. 两组数据分别需要处理什么？

### 2.1 Grid：先补齐杭州映射证据

建议先检查现有备份、历史生成脚本和实验记录，寻找原始映射；找到后复现输出并比对现有 CSV 的 SHA-256：`1a2bdc51cf5656cf7ea348bdf8fee0271345feeac9f9ee732de02500ab71b9c0`。

如果旧映射确实无法恢复，建立一个明确标记为新版本的映射流程：

1. 固定上游 commit、原始 roadnet/flow 文件、时间窗口及文件哈希。
2. 从原始路线提取进入/离开边界、方向、OD 数量和时间结构。
3. 对 4×4 与 5×5 边界位置做坐标归一化，构建显式 `edge_mapping.csv`。记录映射的一对多分配、合并、转向和不可映射情况。
4. 预先规定质量守恒及流量缩放规则，记录每一步前后总流量、OD 数、方向占比与时间分布。保持当前 5×5 路网不变。
5. 每条映射后正需求 OD 用当前 SUMO 路网实际寻路，保存完整路由；不丢弃正流量来绕过错误。
6. 固定 `mapping_manifest.json`、转换脚本版本、CSV/路线/artifact 哈希。

这是一种待实施的新映射设计，**不能倒推为论文原来使用的方法**。在溯源完成前，可以把现有 CSV 用作 `Grid sparse external candidate` 调试场景；正式页面暂不命名为已验证的 `Hangzhou-derived`。

第一轮保留 2,983 veh/h 原生流量；可选增加 3,000 veh/h 归一化版本，缩放因子为 `3000/2983 ≈ 1.00570`。分别命名、分别统计，从而区分空间分布变化与总流量变化。原生版本用于复现已存在的本地 CSV，归一化版本用于与主训练负荷更直接比较。

### 2.2 Monaco：先修复需求与路网的一致性

证据文件：

- [scenario_metadata.json](../../real_net_subnet/demand_groups/scenario_metadata.json)：记录来源及 14 OD。
- [Real_Life_Monaco_audit_summary.json](../../real_net_subnet/demand_groups/Real_Life_Monaco_audit_summary.json)：历史审计用了增加 `10180#0`、`10180#1` 的 272 边集合。
- [Real_Life_Monaco_route_audit.csv](../../real_net_subnet/demand_groups/Real_Life_Monaco_route_audit.csv)：88 个原始 flow 的投影记录。
- [reviewer_revision_audit_2026-09-14.md](../../reviewer_revision_audit_2026-09-14.md)：此前已指出该 profile 不应未经修复加入主测试。

本次按当前 270 条非内部边及 SUMO `<connection>` 建有向图，发现 `-10051#2 → 10043` 不可达，流量 `108.333333 veh/h`，约占总需求 **4.55%**。历史审计的 `88/88 ok` 不代表当前网络的路线也有效。

建议优先尝试在**不修改当前网络资产**的前提下，重新做与子网边界一致的需求投影。必须说明受影响 flow 如何对应到新的合法入口/出口，保留流量与时间质量并记录差异。若不存在有物理意义的合法投影，则不能把删掉该 OD 的版本当作原始需求测试。

另一条路径是建立独立的修复路网版本，重新核查路口、车道、相位及状态维度，再训练与该路网兼容的模型。当前 controller checkpoint 的 signature 包含网络资产哈希；直接改路网会导致旧模型不兼容。这个路径属于新的路网实验，不能与冻结旧模型的零样本需求泛化测试混为一谈，也不应关闭哈希检查强行加载。

Monaco 还应区分两种测试：

| 场景 | 输入与生成方式 | 能回答的问题 |
|---|---|---|
| `monaco_most_od_stationary_v1`，建议首轮 | 14 OD CSV，3600 s 恒定流率，物化随机车辆 | 对 MoST 重建 OD 空间分布的泛化能力 |
| `monaco_most_temporal_v1`，建议第二轮 | 修复后的 88 flow，保留原始 begin/end、分段强度与路线约束 | 对原始分时需求结构的泛化能力 |

本地 `most_0.rou.xml` 有 88 flow，时间段覆盖 0–3300 s；审计中每段通常为 300 s。CSV 将这些贡献按 3600 s 汇总成 14 个小时 OD 流率。**恒定 CSV 再采样会改变时间结构及路径选择，不等同于原始路线逐条重放**。分时测试应保留 3300–3600 s 无新增源 flow 的尾段，不擅自补成平稳流量；总评估仍为 3600 s。

## 3. 公平的独立测试设计

先建立两个分开的 supplementary campaign，沿用 seed 101 的现有最终模型，冻结 policy，禁止在这些外部场景上训练 WCE、微调 controller 或选择 checkpoint。

| 配置 | 建议 |
|---|---|
| 网络 | Grid 与 Monaco 分开运行、分开统计 |
| Controller | IA2C、MA2C、IQLL、PPO |
| Training method | baseline、random_group、domain_randomization、fixed_wce、online_wce |
| 模型选择 | 对应 `runs_eval/revised/publication_seed101_v1/<network>/<controller>/<method>/suite.json` 的精确 `parent` |
| 学习预算 | 现有最终模型累计 2,320,000 learning steps，不根据新测试改选模型 |
| 评估时长 | 每次 3600 s；1 s 日志，5 s 控制，共 720 次控制决策 |
| 初始化 | 每次空路网，包含启动期；主指标不增加排空尾段 |
| 动作方式 | 保持当前 IA2C/MA2C/PPO 采样、IQL 贪心的评估行为 |
| Arrival seeds | 51001–51010 |
| SUMO seeds | 61001–61010 |
| Policy seed | 沿用 `int(digest(['evaluation-policy',101,i])[:8],16)`，`i=0..9` |
| 车辆随机实现 | 同一网络、场景、rollout 下，全部 20 个 controller/method 组合复用同一完整 artifact |
| 运行环境 | 按原训练/评估 manifest 匹配 Python、TensorFlow、SUMO 版本并记录差异 |

单场景每个网络：`4 controllers × 5 methods × 10 rollouts = 200` 次。两组首轮共 **400 次**。Grid 归一化或 Monaco 分时版本每增加一个场景，额外增加 200 次。先独立完成 Grid，Monaco 一致性问题解决后再运行其 campaign；两个网络无需互相等待。

当前 revised 训练输入显式限制为 11 个 profile。正式执行前仍需逐个核查最终模型的训练 manifest，确认外部需求未参与训练或选模。旧版 Grid loader 曾可能自动包含 sparse profile，因此旧模型的 Group 12 结果不能直接当作严格的未见分布测试。

## 4. 实施步骤与输出目录

### 阶段 A：数据准备及准入

建立独立外部 manifest，字段包括：`campaign_id`、`network`、`scenario_id`、`split=external`、`family`、`source_files`、`source_hashes`、`mapping_version`、`network_hash`、`total_rate`、`temporal_schedule`、`route_validation`、`horizon`、`arrival_seed`、`artifact_hash`。所有正需求 OD 必须通过边存在、车辆类别兼容、完整路径及 SUMO 实际加载检查。

本次静态检查只使用有向连接图，**尚未证明车道权限、实际 TraCI 路由及仿真加载通过**。这些检查必须在正式物化前完成。

现有 [experiments/scenarios.py](../../experiments/scenarios.py) 只支持 seen/test/validation；现有 `--suite` 也没有 external。建议新增隔离的 supplementary 物化/批量调度脚本，复用 [experiments/demand.py](../../experiments/demand.py) 的 `materialize/save_artifact/check_artifact`，避免变更已冻结的主训练集合。脚本名称是方案建议，当前还未实现。

### 阶段 B：完整物化与冒烟检查

每个场景生成 10 份完整车辆 artifact，包含 departure、OD、全路由和 speed factor。先一次性生成，再给所有方法复用；不能让方法各自随机生成路线。

做短程路线加载检查和每个 controller 至少一次完整评估冒烟检查，重点确认模型兼容、日志维度、策略冻结及 3600 个样本。冒烟结果单独存放；如重跑，记录新的 attempt，不覆盖失败记录，不根据性能决定是否重跑。

### 阶段 C：分别运行正式评估

建议布局：

```text
runs_eval/revised/external_group12_seed101_v1/
  campaign.json
  artifacts/grid/<scenario>_<arrival_seed>.json
  artifacts/monaco/<scenario>_<arrival_seed>.json
  grid/<controller>/<method>/external/<scenario>/rollout_01/attempt_001/
  monaco/<controller>/<method>/external/<scenario>/rollout_01/attempt_001/
```

现有单次评估 CLI 能接收完整外部 artifact。下面是**在物化与路线检查完成后**可用的调用模板，尖括号需替换；不要加当前不支持的 `--suite external`：

```bash
python main.py experiment evaluate \
  --network grid --controller ppo --method online_wce --seed 101 \
  --parent <EXACT_FINAL_CHECKPOINT_FROM_SUITE_JSON> \
  --artifact <VALIDATED_COMPLETE_EXTERNAL_ARTIFACT_JSON> \
  --sumo-seed 61001 --policy-seed <POLICY_SEED_FOR_INDEX_0> \
  --output <NEW_EXCLUSIVE_ROLLOUT_ATTEMPT_DIRECTORY>
```

Monaco 替换 network、checkpoint 和 artifact。publication 训练需要 verification gate；当前代码的 `evaluate` 分支不要求该训练 gate，但仍检查模型来源、checkpoint 完整性、资产签名及最终学习步数。新增调度器应预检这些条件并保存源代码哈希。

### 阶段 D：独立分析

主要指标用现有 `mean_queue`：每秒全网络 queue 之和在完整 3600 s 上的平均，越小越好。辅助报告 integrated/peak queue、mean speed、inserted、completed、remaining、pending、teleports、collisions；速度指标缺失时保留缺失说明，不填零。

每个 controller 内，按相同 rollout 做方法与 baseline、online_wce 与 fixed_wce 的配对差异。报告十次均值、标准差、配对差异置信区间与胜出次数；收益定义为 `100 × (baseline - method)/baseline`，baseline=0 时百分比记为不可定义。多项正式显著性检验应处理多重比较。

排队更小要结合 completed/pending/teleports 判断，避免把未成功插入车辆或被 teleport 清除误解释为控制更好。分别展示两个网络；原生/归一化、恒定/分时版本分别比较，避免把需求生成方式差异混入算法结论。

十次 rollout 来自**一个训练 seed**，只能估计车辆、SUMO 与策略采样的评估随机性，不能支持“跨训练随机种子稳定优于所有算法”的结论。若要检验这一点，再增加完整多训练 seed 实验，并保持每 seed 相同预算。

## 5. 如何导入 evaluation results site？

目前不能把 200 次结果简单复制到现有主目录后直接导出：

| 当前代码限制 | 所在位置 | 必要调整 |
|---|---|---|
| 固定每网络 4600 rollouts、23 scenarios、460 groups | [export_network_data.py](../../docs/evaluation_workbook/grid_results_site/export_network_data.py) | 根据 campaign manifest 推导矩阵及数量，同时检查每组严格 10 次 |
| split 只允许 seen/test | 同一 exporter；前端 `populateSplits` | 新 campaign 支持 external，选项来自 catalog |
| 前端要求 summary.length=460 | [dist/app.js](../../docs/evaluation_workbook/grid_results_site/dist/app.js) 的 `loadNetwork` | 校验 catalog 的实际 groups 数及 expectedRollouts |
| coverage 文案固定 23 场景及主实验总数 | 前端语言配置/状态文案 | 按选择的 campaign/network 显示数量 |

建议网站增加 **Evaluation set** 选择器：`Main v7` / `External Group 12`，仍保留独立的 Grid / Monaco 网络选择器。不要把测试集名称当成新的物理 network，也不要覆盖主实验数据。

建议新的静态数据目录：

```text
dist/data/supplementary/group12_v1/grid/
dist/data/supplementary/group12_v1/monaco/
```

每个目录沿用 catalog、metrics_summary、rollouts 和 series 的现有格式，补充来源/映射/路网哈希/测试状态。注册表记录 campaign/network/base/expectedRollouts/expectedGroups。首轮每网络 200 rollouts、20 groups、1 scenario；主实验仍每网络 4600、460、23。全站完成后共有 `9200 + 400 = 9600` rollouts，但页面应明确区分主实验和补充实验。

正式导入前必须检查：矩阵完整、无重复 seed、每组十次、3600 连续时间样本、NPZ 维度及非负有限队列、同 rollout 需求哈希与 SUMO seed 配对、同 controller 跨方法 policy seed 配对、精确 checkpoint 哈希、route/network/provenance 版本一致。继承原 exporter 的验证，不仅放宽数量。

先向临时静态目录导出并验证，再更新注册表；Grid 完成时允许单独发布，Monaco 显示 `pending route validation`。失败尝试单独展示原因或保留审计，不能混入完成结果平均值。

用浏览器验证两个 evaluation set 与两个 network 的切换、外部场景列表、20 条方法曲线、十次样本统计、CSV 下载、双语标签、来源说明及独立 URL 状态。回归检查原主数据的 catalog/metrics/series 哈希和数值保持一致。若 8878 当前服务的是另一份旧静态快照，先核对服务进程的实际目录，再更新其对应产物。

## 6. 推荐执行顺序及论文表述

1. 补齐 Grid 来源并完成实际路线检查；先做 Grid 原生流量 200 次测试。
2. 同时解决 Monaco 投影与当前网络一致性；通过后做 Monaco 恒定 OD 200 次测试。
3. 扩展隔离 exporter 与网站 evaluation set，分别导入已验收结果。
4. 需要更接近原数据时，追加 Grid 归一化和 Monaco 分时版本，各额外 200 次。
5. 依据结果更新论文：主实验 11 seen + 12 generated tests；这些单独称 supplementary external-demand tests。

来源未验证时 Grid 应写 `an external sparse OD profile`；有完整上游和映射证据后才能写 `a Hangzhou-derived demand profile mapped to the 5×5 grid`。Monaco CSV 测试建议写 `an OD demand profile reconstructed from the local MoST route file`，保留分段结构的测试写 `a temporally preserved MoST-derived demand scenario`。这些表述与实际输入匹配，不把汇总 OD 再采样称为未经转换的真实车辆轨迹重放。
