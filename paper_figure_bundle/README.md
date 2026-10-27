# 论文图片、绘图程序与源数据

整理日期：2026-10-20。来源：项目根目录中的 `paper/` 及其相关实验文件。

本文件夹收集了 `paper/` 内的全部 9 张 PNG 图片、论文 PDF 与 LaTeX、6 张实验结果图对应的 3 个绘图程序、两套实验的原始 rollout 数据和汇总表。所有收集的源文件均为复制，原文件保持不变。程序和数据保持项目中的相对目录，方便查找和重新绘图。

## 1. 图片与程序、数据的对应关系

| 图片（均在 `paper/`） | 内容 | 直接绘图程序 | 直接输入与来源情况 |
| --- | --- | --- | --- |
| `OverviewFlowChartPaper.png` | 分布鲁棒多智能体强化学习训练框架示意图 | 未找到 | 非实验曲线，未找到可编辑原稿或直接生成代码。不能把项目中其他 CNN 结构图程序当作它的来源。 |
| `5x5_grid.png` | 5×5 合成路网及交叉口示意 | 未找到 | 附有 `supporting_sources/large_grid/data/` 中的路网 XML、`build_file.py` 和已有路网 PDF；属于相关材料，尚未证明能直接输出这张 PNG。 |
| `monaco_controller_classification.png` | Monaco 路网与控制器方向分类 | 未找到 | 附有 `supporting_sources/real_net_subnet/` 中路网、控制器坐标和分类 CSV。另有 `controller_classifications.png`，它与论文图片不是同一文件，不能视为已确认的直接来源。 |
| `Absolute_Heatmap_Comparison_optimized.png` | 5×5：12 个需求组、4 种算法的队列和速度热力图 | `heatmap_optimized_unseen.py` | `runs_eval/signal_controller_benchmark/full_performance_comparison.xlsx` |
| `Absolute_Heatmap_Comparison_real_optimized.png` | Monaco：队列和速度热力图 | 同上 | `runs_eval/signal_controller_benchmark_real/full_performance_comparison_real.xlsx` |
| `Combined_Performance_Comparison.png` | 5×5：各需求组上的性能分布箱线图 | `boxplot_comparison_2in1.py` | 5×5 的同一 Excel |
| `Combined_Performance_Comparison_real.png` | Monaco：性能分布箱线图 | 同上 | Monaco 的同一 Excel |
| `5x5_Grid_Stacked.png` | 5×5：指定最差需求组的 3600 秒队列/速度曲线 | `Worst_case_Multi_plot.py` | `runs_eval/signal_controller_benchmark/{ia2c,ma2c,iqll,ppo}/*_raw.csv` 中被配置选中的文件 |
| `Monaco_City_Stacked.png` | Monaco：指定最差需求组的队列/速度曲线 | 同上 | `runs_eval/signal_controller_benchmark_real/{ia2c,ma2c,iqll,ppo}/*_raw.csv` 中被配置选中的文件 |

最后 6 张图与原项目 `runs_eval/ieee_comparisons_stacked/`、`runs_eval/manual_comparisons/`、`runs_eval/manual_comparisons_real/` 中同名输出的 SHA-256 完全一致。这确认了输出文件的对应关系；不代表已重新执行程序并获得逐像素一致的新图。

## 2. 文件夹结构

```text
paper_figure_bundle/
  README.md
  paper/                           论文图片、main.tex、论文 PDF
  Worst_case_Multi_plot.py          原始时序曲线绘图程序
  heatmap_optimized_unseen.py       原始热力图程序
  boxplot_comparison_2in1.py        原始箱线图程序
  requirements.txt                 重新绘图所需 Python 库
  run_figures.py                    在本文件夹运行的绘图入口
  runs_eval/
    signal_controller_benchmark/       5×5 汇总表、清单和原始 CSV
      ia2c/ ma2c/ iqll/ ppo/
    signal_controller_benchmark_real/  Monaco 汇总表、清单和原始 CSV
      ia2c/ ma2c/ iqll/ ppo/
  supporting_sources/              路网、交通需求、上游参考代码
  stacked_plot_inputs.csv          时序图实际选取的 32 条输入映射
  latex_image_references.csv       main.tex 中全部有效图片引用及缺失情况
  file_manifest.csv                每个复制文件的原路径、包内路径、字节数、SHA-256
  raw_data_checks.csv              原始 CSV 行数、rollout 数、时域均值
  raw_vs_workbook.csv              原始 CSV 均值与 Excel 绘图输入值的差异
  collection_report.json           收集和检查结果
```

`supporting_sources/` 保留原项目层级。它提供来源参考，不是完整训练或仿真运行包。未收集模型权重及全部训练依赖，因此不能仅凭本文件夹从头重新训练智能体。已有的 6 张统计图无需重新运行 SUMO 或训练模型。

## 3. 原始实验数据怎么读

两套实验各有 4 个算法 × 2 个控制器版本 × 12 个需求组 × 2 个指标 = 192 个原始 CSV，合计 384 个。

例如：

```text
ia2c_marl_group02_demand_w_to_e_queue_raw.csv
ppo_retrained_group01_s_to_n_speed_raw.csv
```

- `ia2c`、`ma2c`、`iqll`、`ppo`：四种算法，`iqll` 在论文图中写作 IQL-LR。
- `marl`（5×5）或 `real`（Monaco）：基线控制器；`retrained`：DR 重训练控制器。
- `group00` 至 `group11`：程序内部从 0 开始编号，论文的 Group 1 至 Group 12 要加 1。
- `queue_raw`：队列长度，单位为车辆数；`speed_raw`：平均车速，单位为 m/s。
- 每个 CSV 的 `t` 是仿真秒数，`rollout_0` 至 `rollout_9` 是 10 次评估。已检查每个文件有 3600 行，时间为 1 至 3600。

队列口径需要注意：评估程序优先记录 `total_queue`，缺失时回退到 `avg_queue`，评估汇总清单标记为 `total_queue`。因此不能仅根据论文图注中的 “per intersection” 将现存队列数据认定为每路口均值；如需换算，应先确认环境实际记录口径。速度取评估日志中的 `avg_speed_mps`。

时序图在每一秒对 10 个 rollout 求均值画线，以该秒 rollout 的最小值和最大值画阴影；阴影不是置信区间。基线为虚线，重训练为实线。最差组由 `Worst_case_Multi_plot.py` 的 `PLOT_CONFIGS` 手工指定，程序并不会自动重新挑选最差组。队列和速度也可能选不同组。`stacked_plot_inputs.csv` 列出每条线的算法、版本、指标、论文组号和实际 CSV 路径。

## 4. 汇总表、热力图和箱线图怎么读

两份 `full_performance_comparison*.xlsx` 均为 15 行、25 列：前两行为表头，第 3 至 14 行为 12 个需求组，最后一行为整体汇总。第 A 列为内部组号。

| 算法 | 队列：基线 / DR / 变化百分比 | 速度：基线 / DR / 变化百分比 |
| --- | --- | --- |
| IA2C | B / C / D | E / F / G |
| MA2C | H / I / J | K / L / M |
| IQL-LR | N / O / P | Q / R / S |
| PPO | T / U / V | W / X / Y |

热力图直接读取这两份 Excel 的 12 个组数据。代码提供的 `*.xlsx - Sheet1.csv` 优先输入在当前项目中不存在，因此收集包使用 Excel。最后一组用红框标注为 unseen。

箱线图的每组样本是 12 个需求组各自的汇总值，并不是 36000 个逐秒 rollout 值。图中文字的均值和变化百分比直接读取 Excel 最后一行。变化率按 `(DR - 基线) / 基线 × 100` 理解，队列下降为负值，速度上升为正值。百分比列存储的是百分数值，例如 `-83.46` 表示 `-83.46%`。

`summary_averages.csv` 保存评估输出汇总；`horizon_averages_from_raw.csv` 为已有时域汇总。尤其 Monaco 的这个文件还包含右侧手工展开表，不能当作完全整洁的单表直接使用。收集包按原样保留这些文件，未修正或覆盖。

`raw_data_checks.csv` 的时域均值计算为：每秒先对 rollout 求平均，再对所有时间点求平均。`raw_vs_workbook.csv` 的 `difference` 为该值减去 Excel 对应值，可用于核对汇总表与原始数据是否一致。Excel 是现存绘图程序的直接输入，不应未经核对便用其他汇总文件替换。

本次对全部 384 个原始 CSV 的比较中，绝对差值均不超过 0.011，未发现超出该阈值的差异。5×5 汇总表中的组数据通常只保留两位小数。

## 5. 需求分组与路网数据

| 论文组号 | 内部编号 | 需求含义 |
| --- | --- | --- |
| 1–4 | 00–03 | 北→南、南→北、西→东、东→西 |
| 5–8 | 04–07 | 西北→东南、东南→西北、西南→东北、东北→西南 |
| 9–10 | 08–09 | 周边→中心、中心→周边 |
| 11 | 10 | Uniform，均匀需求 |
| 12 | 11 | 未见需求：5×5 为 `demand_5x5_sparse`；Monaco 为 `Real_Life_Monaco` |

- `demand_manifest.csv`：内部组号、需求名称及原始需求文件路径。里面的 Linux 绝对路径来自原实验机器，当前电脑不能直接照用；对应需求文件已收集到 `supporting_sources/data_traffic/` 和 `supporting_sources/real_net_subnet/demand_groups/`。
- `controller_manifest.csv`：算法与控制器模型配置的原始记录。记录中的模型路径不表示包内包含模型权重。
- 需求 CSV 中的 `origin_edge`、`dest_edge`、`veh_per_hour` 表示起点路段、终点路段和每小时车辆需求。它们是仿真输入，与评估输出的逐秒队列 CSV 不同。
- `large_grid/data/build_file.py`：生成合成路网 XML 的程序；`.net.xml` 为 SUMO 路网，`.nod.xml`、`.edg.xml`、`.con.xml`、`.tll.xml`、`.typ.xml` 分别描述节点、边、连接、信号灯和道路类型；`.rou.xml` 是车辆/路线配置，`.sumocfg` 是仿真配置。
- `real_net_subnet/data/in/most.net.xml` 与 `subnetwork_meta.json`：Monaco 子路网与控制器、物理交叉口等元数据。
- `controller_classification.csv`：控制器 ID、物理路口、坐标、相对中心位置、角度、内外环、方向、象限和邻居数。`controller_groups.csv` 记录组归属，`group_edge_summary.csv` 记录组规模，`scenario_metadata.json` 提供需求场景信息。这些是分类图的相关数据，尚未确认论文图片的直接渲染程序。

## 6. 重新绘制已有的 6 张实验结果图

在此文件夹内打开终端：

```powershell
python -m pip install -r requirements.txt
python run_figures.py --check
python run_figures.py
```

`run_figures.py` 自动切换到包目录，建立输出目录，并在内存中将原程序的 Linux `/tmp/matplotlib` 缓存路径改成本地缓存；三个归档绘图源文件保持原样。

输出位置：

```text
runs_eval/manual_comparisons/              5×5 热力图、箱线图
runs_eval/manual_comparisons_real/         Monaco 热力图、箱线图
runs_eval/ieee_comparisons_stacked/         两张时序曲线图
```

`paper/` 中的原始图片不被绘图入口覆盖。不同字体、库版本和渲染环境可能影响图像外观。`requirements.txt` 列出依赖但未锁定版本。当前收集环境缺少完整绘图库，未实际执行重新绘图；已执行复制哈希、全部原始 CSV 结构、汇总表结构及选中输入路径检查。

## 7. 已知缺失及论文引用路径问题

`paper/main.tex` 中不少引用指向 `Images/Graphs/`，而实际图片直接位于 `paper/`。原 LaTeX 已按原样收集，没有替用户修改论文。若要重新编译，应先调整相对路径。

`monaco_controller_classification.pdf` 没有找到，但同名 PNG 已收集。`map.jpeg`、`baseline.png`、`retrained.png` 以及作者照片 `shuwei.jpeg`、`Joran.png`、`Arda.png`、`Bayu.png`、`Omer.png`、`Saeed.jpg` 在本次项目工作目录中未找到。完整有效引用及状态见 `latex_image_references.csv`。论文 PDF 中若包含这些图，它们也只是嵌入结果，不等于已找到原始素材、绘图代码或底层数据。

本次检索覆盖当前项目文件，并查看了相关图片/分类文件的 Git 历史路径。未发现框架图、合成路网 PNG 和 Monaco 分类图的直接生成代码，故这三项保留为“未找到”，不以相关程序替代直接来源。

## 8. 文件来源与校验

`file_manifest.csv` 的 `source_path` 相对于原项目根目录，`bundle_path` 相对于本文件夹。所有清单中的文件均经过源文件和复制文件的 SHA-256 一致性检查。新写的 README、运行入口和检查报告不属于原始源文件清单。

收集程序位于原项目 `scripts/collect_paper_figures.py`，包内另保留了一份供参考；它需要在完整原项目的 `scripts/` 下运行。检查报告及差异表描述的是收集当时的快照，未来改动数据后需重新核对。
