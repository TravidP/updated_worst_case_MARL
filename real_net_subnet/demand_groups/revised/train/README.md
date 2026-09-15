# Monaco train traffic / Monaco 训练交通

[Full generation explanation](../../../../reviewer_revision_plan.md#traffic-generation) · [完整中文生成说明](../../../../reviewer_revision_plan_zh.md#traffic-generation) · [Generator source](../../../../experiments/scenarios.py)

These inputs use Monaco OD profiles from `real_net_subnet/demand_groups/<name>.csv`, normalized to **2,383.3333 veh/hour** per profile. The original CSVs remain unchanged.

输入来自 Monaco 的 `real_net_subnet/demand_groups/<name>.csv`，每种配置归一化至 **2,383.3333 veh/hour**；原始 CSV 保持不变。

Eleven `<profile>.csv` files contain `origin_edge,dest_edge,veh_per_hour`. `manifest.json` stores their order, source/prepared hashes and normalization factors. Grid uses alphabetical profile order; Monaco uses `ORDER` in `experiments/demand.py`. Use this order when reading eleven-element mixture weights. Validation/test artifacts are excluded from this training manifest.

十一份 `<profile>.csv` 使用 `origin_edge,dest_edge,veh_per_hour` 列；`manifest.json` 保存顺序、原始／准备后哈希和归一化因子。Grid 按配置名称字母序，Monaco 按 `experiments/demand.py` 中的 `ORDER`；解释十一维混合权重时必须使用该顺序。训练清单不包含验证／测试文件。

From the repository root / 从仓库根目录执行：

```bash
python main.py experiment prepare
```

Preparation preserves matching existing files and rejects conflicts. Materialization resolves routes through SUMO; it does not train controllers. Do not overwrite frozen artifacts or move validation/test files into training directories.

准备流程保留匹配的已有文件并拒绝冲突；物化通过 SUMO 解析路径，不训练控制器。不要覆盖冻结文件或将验证／测试文件移入训练目录。
