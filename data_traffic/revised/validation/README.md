# Grid validation traffic / Grid 验证交通

[Full generation explanation](../../../reviewer_revision_plan.md#traffic-generation) · [完整中文生成说明](../../../reviewer_revision_plan_zh.md#traffic-generation) · [Generator source](../../../experiments/scenarios.py)

These inputs use Grid OD profiles from `data_traffic/demand_<name>.csv`, normalized to **3,000 veh/hour** per profile. The original CSVs remain unchanged.

输入来自 Grid 的 `data_traffic/demand_<name>.csv`，每种配置归一化至 **3,000 veh/hour**；原始 CSV 保持不变。

These are predefined synthetic scenarios, not new measured traffic or WCE-generated evaluation demand. Use validation for development; keep test results held out from parameter/checkpoint selection. Each scenario spans 3,600 seconds.

这些是预先定义的合成场景，不是新实测交通或 WCE 生成的评估需求。验证用于开发；测试结果不用于选择参数／检查点。每个场景为 3,600 秒。

| Family / 类别 | Settings / 参数 | Generation seeds / 生成种子 |
|---|---|---|
| OD redistribution / OD 重分配 | lognormal σ = 0.35, 0.65 | 31001–31002 |
| Convex mixtures / 凸组合混合 | 2 × Dirichlet(1,…,1) | 31003–31004 |
| Temporal switching / 时间切换 | N_to_S ↔ W_to_E; 450 s | 31005 |
| Peak / 高峰 | Uniform × 1.15 during [1200,2400) s | 31006 |

`validation_scenarios.json` contains six definitions. `artifacts/<scenario>_<arrival_seed>.json` contains **60** complete vehicle schedules: six scenarios × ten arrival seeds. Example: `artifacts/peak_1.15_51001.json`.

`validation_scenarios.json` 保存六项定义；`artifacts/` 保存 **60** 份完整车辆计划，即六场景 × 十个到达种子。例如 `artifacts/peak_1.15_51001.json`。

Redistribution changes Uniform OD proportions and renormalizes the total; mixtures use one fixed vector for the full hour. Counts are Poisson with mean `rate × block_seconds / 3600`. Departures are uniform within a block plus normal jitter (SD 2 s), clipped inside the block and rounded to 0.01 s. Positive speed factors are drawn from `Normal(1,0.1)` (mean 1, SD 0.1) by rejection sampling. Full routes are resolved in the corresponding SUMO network before saving.

重分配改变 Uniform 的 OD 比例后恢复总需求率；混合在一小时内使用固定向量。车辆数按均值为 `rate × block_seconds / 3600` 的泊松分布抽取。出发时间为区间内均匀样本加标准差 2 秒的正态扰动，经区间内裁剪后保留到 0.01 秒。正速度因子通过拒绝抽样从 `Normal(1,0.1)`（均值 1、标准差 0.1）获得。保存前在对应 SUMO 路网中解析完整路径。

Arrival seeds `51001–51010` generate ten artifacts per scenario; SUMO evaluation uses `61001–61010`. Arrival seeds are reused across splits; validation/test definitions and paths remain separate. Reuse an exact traffic hash across compared controllers/methods. Each JSON contains scenario metadata, hashes and `vehicles` (`id`, `depart`, `origin`, `destination`, `edges`, `speed_factor`). Actual insertion can differ under congestion.

到达种子 `51001–51010` 为每场景生成十份文件；SUMO 评估使用 `61001–61010`。到达种子在不同划分之间复用，但验证／测试定义与目录分离。控制器／方法之间复用完全相同的交通哈希。JSON 保存场景元数据、哈希及 `vehicles`（`id`、`depart`、`origin`、`destination`、`edges`、`speed_factor`）；拥堵可导致实际插入情况不同。

From the repository root / 从仓库根目录执行：

```bash
python main.py experiment prepare
python main.py experiment prepare --materialize --network grid
```

Preparation preserves matching existing files and rejects conflicts. Materialization resolves routes through SUMO; it does not train controllers. Do not overwrite frozen artifacts or move validation/test files into training directories.

准备流程保留匹配的已有文件并拒绝冲突；物化通过 SUMO 解析路径，不训练控制器。不要覆盖冻结文件或将验证／测试文件移入训练目录。
