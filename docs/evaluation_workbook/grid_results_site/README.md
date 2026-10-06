# Current snapshot / 当前快照

Grid and Monaco are both imported: 9,200 rollouts, 920 series.
两套路网均已导入。[Restore data / 数据恢复](RESTORE.md) · [Archive metadata](release.json).
The older workflow notes below describe export operations, not missing Monaco data.

# Grid / Monaco Results Site · 双语结果站点

本地、无外部依赖、只读的 Protocol v7 结果分析站点。当前包含完整 Grid 4,600 条 rollout；Monaco 数据接口已经预留，只有完整导入并通过验证后才会在页面启用。

## 启动站点

```bash
python3 docs/evaluation_workbook/grid_results_site/server.py
```

打开 <http://127.0.0.1:8878/>，使用 `Ctrl+C` 停止服务。若端口被占用：

```bash
python3 docs/evaluation_workbook/grid_results_site/server.py --port 8879
```

## 图表操作

- 总览及左右双窗口共 3 张 Canvas 图支持鼠标滚轮或 `＋/−` 按钮缩放。
- 按住任意图表左右拖拽，所有图同步到相同时间视窗。
- `↺` 恢复完整 1–3,600 秒视窗。
- `PNG` 下载当前图表、当前缩放范围和当前平滑效果。
- 平滑强度为 `0–0.95`：`0` 是原始数据，数值越大曲线越平滑。
- 总览中的五种方法可独立勾选；至少保留一种。
- “下载当前筛选 CSV”导出当前勾选方法的原始 CSV，避免平滑结果覆盖原始实验数据。

## 重新生成 Grid 数据

```bash
python3 docs/evaluation_workbook/grid_results_site/export_grid_data.py --force
```

## Monaco 完成后的自动导入

Monaco 的 4,600 条 rollout 全部完成后执行：

```bash
python3 docs/evaluation_workbook/grid_results_site/export_network_data.py --network monaco
```

该命令只读取：

```text
runs_eval/revised/publication_seed101_v1/monaco
```

它不会启动训练、SUMO、evaluation 或正式报告。导入前会验证：

- 4 controllers × 5 methods × 23 scenarios × 10 rollouts = 4,600；
- 每条 rollout 恰好 3,600 个时间点；
- arrival seeds 为 51001–51010；
- demand hash、SUMO seed 和 policy seed 配对一致；
- 无失败、NaN 或负 queue；
- 时序 queue 与 `rollout_summary` 对账一致。

验证成功后，数据写入 `dist/data/networks/monaco/`，并原子更新 `dist/data/networks.json`。刷新页面后，“Monaco 城市路网”会自动从禁用状态变为可选择状态。已有导出需要明确覆盖时使用：

```bash
python3 docs/evaluation_workbook/grid_results_site/export_network_data.py --network monaco --force
```

## 数据文件

- `dist/data/series/`：Grid 的 460 个时序 CSV，每个 3,600 行。
- `dist/data/metrics_summary.csv`：Grid 的 460 行分组指标。
- `dist/data/rollout_metrics.csv`：Grid 的 4,600 行 rollout 指标。
- `dist/data/catalog.json`：Grid 双语 catalog。
- `dist/data/networks.json`：页面自动读取的路网注册表。
- `dist/data/networks/monaco/`：未来 Monaco 导入位置。

语言选择保存在浏览器 `localStorage`。切换语言不会重置路网、controller、split、scenario、方法选择、缩放范围或平滑强度。

## 多算法与双窗口比较

1. **01 结果筛选**：Controller 使用复选框，可同时选择 IA2C、MA2C、IQLL、PPO。
2. **02 多算法平均排队总览**：展示所选 Controller × 所选方法；颜色表示方法，线型表示 Controller。
3. **03 所选算法的完整指标对比**：每个所选 Controller 展示全部五种方法的原始 10-run 统计，不受总览方法勾选影响。每个 Controller 内分别计算各指标最佳均值，以绿色和 ★ 标注，并列最佳全部标注；核对指标不参与最佳值评选。
4. **04 双窗口自定义对比**：桌面横排窗口 A/B，每个窗口都可从全部 20 种 Controller × Method 组合独立多选，移动端上下排列。顶部 Controller 和方法选择不改变这两个窗口的选择。
5. **05 口径与完整性**：保留指标说明和 Monaco 导入入口。

两个窗口共享顶部路网、split 和 scenario，以及时间视窗、Y 轴范围和平滑强度。切换语言会保留所有选择。当前筛选 CSV 包含总览所选组合的原始时序数据。
