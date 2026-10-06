# 终点奖励与原生训练监控——方案版本 5

[English](TRAINING_MONITORING.md) · [完整方案](../reviewer_revision_plan_zh.md)

## 生效训练设置

Grid 与 Monaco 均使用控制器终点奖励：每个五秒动作结束后，IA2C/PPO/IQL 接收 `-Q(t+5)`。MA2C 接收负局部终点排队加上 0.9 加权的负邻居终点排队。**控制器奖励不归一化、不裁剪。** WCE 保持 `mean(Q over 600 seconds)/100`。评估对无上限的每秒排队值取平均。三者共享测量来源，但时间聚合／缩放有意不同。

两个路网的 IA2C、MA2C、PPO 批量均为 120 个转移。IA2C/MA2C 按顺序收集；PPO 将同一批数据复用四个优化轮次。IQL 保持 20 个样本的小批量、1,000 条回放容量，每二十个学习转移为每个智能体执行十次小批量更新。其余参数不变。相比历史 40 步批量，Monaco IA2C/MA2C 的更新频率降为三分之一。

训练回合仍为 **6,600 秒**，不是 600 秒；包含十一个 600 秒需求区间和 1,320 个联合控制器转移。不缩放奖励是一项实验选择，不能据此宣称价值估计不足已解决。

## 命令

在仓库根目录使用 `deeprlsc` 环境。试运行种子 9001 与论文种子 101 分开。

```bash
conda activate deeprlsc
# 短机制检查；仅本试运行缩短监测间隔。
python main.py experiment parent --network monaco --controller ma2c \
  --seed 9001 --pilot --visualization --steps 160 \
  --monitor-every 1 --monitor-rollouts 1

# 全部检查通过后才生成 gate。验证期间不要修改源代码／配置。
CBWCE_VERIFY_DIR="$PWD/runs_eval/revised/verification/endpoint_$(date +%Y%m%d_%H%M%S)"
python main.py experiment verify --workers 4 --output "$CBWCE_VERIFY_DIR" \
  && export CBWCE_GATE="$CBWCE_VERIFY_DIR/gate.json"

# 顺序启动八个全新父模型，不启动 WCE／续训完整矩阵。
unset CBWCE_RESUME CBWCE_STEPS CBWCE_EPISODES
bash scripts/training/08_all_parents.sh --mode publication --seed 101 \
  --gate "$CBWCE_GATE" --no-visualization

# 新原生日志；历史导出器可能仍占用 6006 端口。
tensorboard --logdir runs/revised --host 127.0.0.1 --port 6007
```

论文设置固定为 `--monitor-every 50 --monitor-rollouts 3`，也是默认值。Shell 包装脚本还接受 `CBWCE_MONITOR_EVERY` 和 `CBWCE_MONITOR_ROLLOUTS`；命令行参数优先。监测仅用于观察：它记录配对 rollout、TensorBoard 指标和检查点，但不会再因相对初始 monitor 的性能退化自动终止运行；最终质量判断使用完整配对评估。恢复时显式选择新方案的准确检查点，保持相同监测设置。旧奖励方案检查点不能用于新方案续训；不要按最新文件名选择。

## TensorBoard 应看哪些图

| 标签 | 横轴 | 含义与改善方向 |
|---|---|---|
| `train/reward_by_learning_step` | 实际累计学习步 | 每个转移的实际奖励；**越接近零越好**。IA2C/PPO/IQL 共享奖励只记一次；MA2C 为各智能体邻域奖励均值。 |
| `train/episode/mean_total_queue` | 回合结束时的学习步 | 全部 6,600 秒的平均排队；**越低越好**。 |
| `train/episode/mean_reward`, `reward_sum` | 回合结束时的学习步 | 实际回合奖励聚合；奖励总和受时长影响。 |
| `train/partial_episode/*` | 学习步 | 最终预算截断回合及实际时长；不混入完整回合曲线。 |
| `block/mean_total_queue_recent_600_seconds` | 学习步；冻结阶段为 WCE 宏转移 | 仅用于最近需求区间的诊断。 |
| `learner/*/{mean,min,max}` | 学习步 | 排除补齐样本的实际更新诊断，跨智能体／小批量汇总。 |
| `monitor/mean_total_queue`, `monitor/mean_current_wait_seconds` | 学习步 | 固定 Uniform 监测；越低越好。`_sd` 为重复轨迹的样本标准差，不是置信区间。 |
| `monitor/queue_by_second`, `monitor/current_wait_by_second` | 仿真秒 1–600 | 每轮详细日志在 `monitoring/round_*/tensorboard/`。 |
| `monitor/queue_waiting` | 学习步 | Images 页签：测试内曲线及均值 ± 样本标准差。 |
| `wce/*`, `wce_episode/*` | WCE 宏转移 | WCE 奖励／损失及回合统计；不写控制器学习奖励点。 |

学习器诊断包括策略／价值损失、熵、预测值、回报目标、优势、裁剪前策略／价值梯度范数和裁剪因子；PPO 增加裁剪比例。IQL 报告 TD 损失、Q 预测／目标、梯度范数、epsilon 和回放占用量。原始逐智能体／小批量记录保存在 `learner_metrics.jsonl`。TensorBoard 显示可能降采样，原始记录保留全部实际值。事件采用缓冲写入，定期及回合结束时刷新，不在每一步同步刷盘。

不同控制器类型的奖励数值不能直接比较，因为 MA2C 使用邻域奖励。比较交通性能应使用规范的平均总排队值。监测 SUMO 以无界面方式运行；`--visualization` 控制训练环境。

在对抗续训中，训练需求可能逐渐变难。因此训练奖励下降本身不能证明控制性能下降，应同时观察固定 Uniform 监测曲线。短监测测试不能代替最终泛化评估。

损失与梯度曲线用于诊断学习机制；优化损失下降本身不代表交通控制改善。判断进展时应优先看固定监测的排队下降，并结合完成／待插入车辆数。

## 固定监测测试与保存数据

父模型及五种续训方法均在学习前、每 50 个完整回合及最终预算时监测。每轮顺序执行三次 600 秒 Uniform 测试，从空路网开始，无预热。Grid 为 3,000 辆／小时，Monaco 为 2,383.3333 辆／小时。到达种子 53001–53003、SUMO 种子 63001–63003、策略种子 73001–73003。交通文件按路网／需求哈希缓存于 `runs_eval/revised/monitoring_inputs/`，与最终评估数据分开。

冻结模型实例与独立 SUMO 连接将测试与训练隔离。监测不推进学习计数器、回放、调度器或训练随机流。不得依据监测表现替换规定的最终预算检查点。测试失败时，训练检查点已保存，整个任务停止并记录失败；不得成为有效的零排队观测。

每次运行保存 `tensorboard/`、`episode_metrics.jsonl`、`learner_metrics.jsonl` 和 `monitoring/round_<episode>/`。每轮保存准确检查点引用、三个 `rollout_*/timeseries.csv`、JSONL 原始记录、逐车道 NPZ、`summary.json` 及 PNG/SVG 图。恢复时复用已完成监测轮次；失败轮次在新尝试中重试。

每个 CSV 恰有 600 行。`queue` 为唯一受控进口车道上的停车车辆数（Grid 150 条；Monaco 116 条）。`current_wait_mean_seconds` 是当前位于这些车道的车辆的连续等待时间均值，分母包括正在移动的车辆。监测车道为空时记零，并以 `monitored_vehicles=0` 标识；这不是已完成行程的等待时间。`current_wait_sum_vehicle_seconds` 汇总正在持续的等待片段；`cumulative_stopped_vehicle_seconds` 对排队进行时间积分，保留过去贡献。

`rollout.npz` 包含 `time`、`lanes`、时间×车道的 `queue`；`waiting.npz` 包含匹配的 `time`、`lanes`、`current_wait`。控制 JSONL 记录实际奖励向量、累计 `learning_steps`、阶段仿真步及是否学习。TensorBoard 使用相同记录。监测耗时单独记入 `monitoring`，也计入总流程墙钟时间。

## 兼容性与证据

控制器签名使用 `learner_boundary_scaled_v3`；WCE 签名包含方案版本 6。控制记录和交通指标仍保存原始 `-queue`，仅在 learner 边界缩放奖励。修改前的精确副本与校验索引保存在 `docs/history/endpoint_monitoring_20260916T095506Z/`。历史检查点／结果保留原位置，并且有意禁止进入新的训练工作流。

新验证覆盖奖励算术、TensorBoard／原始数据一致性、有无监测的训练等价性、冻结状态、恢复复用，以及两个路网和四类控制器的 C01–C10。新生成 gate 的实际状态是准入依据，旧历史报告不能代替。本轮完整训练仅包含八个父模型；WCE 与四十次续训留待后续显式启动。

本次完整八组验证已通过。见[新 gate](../runs_eval/revised/verification/endpoint_monitoring_20260916_v1/gate.json)及[实现／训练状态报告](ENDPOINT_MONITORING_REPORT.md)。
