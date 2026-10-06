# CB-WCE 训练进度笔记（2026-09-21）

> 核对时间：2026-09-21 12:24（Europe/Amsterdam）  
> 适用协议：`config/revised/protocol.json`，version 6，论文训练种子 `101`  
> 完成判定：只有运行目录中的 `result.json` 明确记录 `"status": "complete"` 才算完成。仅有 `progress.jsonl`、`manifest.json` 或中间 checkpoint 不算完成。

## 1. 当前结论

| 阶段 | 计划数量 | 已完成 | 当前状态 |
|---|---:|---:|---|
| 验证 gate | 1 | 1 | 已通过：`runs_eval/revised/verification/no_degradation_stop_20260918_134859/gate.json` |
| Parent 控制器 | 8 | 8 | **全部完成**；每个均达到 `1,000,000` 学习步 |
| 离线 WCE | 8 | 8 | **全部完成**；每个均完成 `500` 回合／`660,000` 冻结控制器仿真步 |
| 五种方法续训 | 40 | 3 | 仅 3 个 Grid baseline 完成；总进度 `3/40 = 7.5%` |
| 最终评估 | 9,200 rollouts | 0 | **尚未开始**；必须等 40 个续训模型完成 |

当前没有正在运行的 `main.py experiment` 或训练矩阵进程。最近几个没有 `result.json` 的任务已经停止，不能视为“仍在训练”。

完整研究目标是：

```text
8 parents → 8 offline WCE → 40 continuations → 9,200 evaluation rollouts
```

每个续训任务需要在 parent 的 `1,000,000` 学习步基础上再训练 `1,320,000` 步，最终控制器累计学习步数为 `2,320,000`。续训目录名 `checkpoint_001320000` 表示“本阶段新增步数”，而 `result.json.learning_steps` 应为累计的 `2,320,000`。

## 2. Parent 训练情况及调用方法

以下 8 个是当前选择文件实际引用、并且已经完整完成的 parent checkpoint。

| 路网 | 控制器 | 状态 | 学习步数 | Parent checkpoint |
|---|---|---|---:|---|
| Grid | IA2C | 已完成 | 1,000,000 | `runs/revised/grid/ia2c/seed_101/parent/publication_20260917T163857_9068b691/checkpoint_001000000` |
| Grid | MA2C | 已完成 | 1,000,000 | `runs/revised/grid/ma2c/seed_101/parent/publication_20260917T163857_74df8a9c/checkpoint_001000000` |
| Grid | IQL-LR (`iqll`) | 已完成 | 1,000,000 | `runs/revised/grid/iqll/seed_101/parent/publication_20260917T163857_6987d164/checkpoint_001000000` |
| Grid | PPO | 已完成 | 1,000,000 | `runs/revised/grid/ppo/seed_101/parent/publication_20260917T163857_65b5f55e/checkpoint_001000000` |
| Monaco | IA2C | 已完成 | 1,000,000 | `runs/revised/monaco/ia2c/seed_101/parent/publication_20260918T113143_ec086fed/checkpoint_001000000` |
| Monaco | MA2C | 已完成 | 1,000,000 | `runs/revised/monaco/ma2c/seed_101/parent/publication_20260918T192930_8362f84f/checkpoint_001000000` |
| Monaco | IQL-LR (`iqll`) | 已完成 | 1,000,000 | `runs/revised/monaco/iqll/seed_101/parent/publication_20260918T113221_6fa5eb2e/checkpoint_001000000` |
| Monaco | PPO | 已完成 | 1,000,000 | `runs/revised/monaco/ppo/seed_101/parent/publication_20260918T113241_241592fd/checkpoint_001000000` |

### Parent 怎么使用

单个任务中，将对应 checkpoint 作为 `--parent` 输入。例如 Grid IA2C：

```bash
export CBWCE_PARENT="$PWD/runs/revised/grid/ia2c/seed_101/parent/publication_20260917T163857_9068b691/checkpoint_001000000"

# 用冻结的 parent 训练对应 WCE
bash scripts/training/02_wce.sh --mode publication --network grid \
  --controller ia2c --seed 101 --parent "$CBWCE_PARENT" \
  --gate "$CBWCE_GATE" --no-visualization

# 从同一个 parent 启动一种续训方法
bash scripts/training/03_baseline.sh --mode publication --network grid \
  --controller ia2c --seed 101 --parent "$CBWCE_PARENT" \
  --gate "$CBWCE_GATE" --no-visualization
```

不要按目录时间猜“最新模型”，也不要使用早期 interrupted/degraded parent。矩阵任务应优先读取统一选择文件：

```text
runs_eval/revised/selections/publication_seed101.json
```

## 3. WCE 训练情况、保存位置及使用方法

WCE 是需求对手模型。离线 WCE 训练期间控制器保持冻结；WCE 最大化拥堵成本。下面 8 个 WCE 均已完成 `500` 次更新，checkpoint 后缀均为 `000660000`。

| 路网 | 对应控制器 | 状态 | WCE checkpoint |
|---|---|---|---|
| Grid | IA2C | 已完成 | `output_adversary/revised/grid/ia2c/seed_101/wce/publication_20260918T192222_8d09d236/checkpoint_000660000` |
| Grid | MA2C | 已完成 | `output_adversary/revised/grid/ma2c/seed_101/wce/publication_20260918T192222_8336fd70/checkpoint_000660000` |
| Grid | IQL-LR | 已完成 | `output_adversary/revised/grid/iqll/seed_101/wce/publication_20260918T192222_2645abc9/checkpoint_000660000` |
| Grid | PPO | 已完成 | `output_adversary/revised/grid/ppo/seed_101/wce/publication_20260918T192222_282c40aa/checkpoint_000660000` |
| Monaco | IA2C | 已完成 | `output_adversary_monaco/revised/monaco/ia2c/seed_101/wce/publication_20260919T110336_64312bde/checkpoint_000660000` |
| Monaco | MA2C | 已完成 | `output_adversary_monaco/revised/monaco/ma2c/seed_101/wce/publication_20260919T110336_ff68fed8/checkpoint_000660000` |
| Monaco | IQL-LR | 已完成 | `output_adversary_monaco/revised/monaco/iqll/seed_101/wce/publication_20260919T110336_4819b424/checkpoint_000660000` |
| Monaco | PPO | 已完成 | `output_adversary_monaco/revised/monaco/ppo/seed_101/wce/publication_20260919T110336_7060c9dc/checkpoint_000660000` |

### WCE 怎么使用

- `baseline`、`random_group`、`domain_randomization`：只需要对应 parent，不加载 WCE。
- `fixed_wce`：加载 parent 和对应 WCE；续训时 WCE 参数冻结，只用它选择困难需求。
- `online_wce`：加载同一个 parent 和对应 WCE；控制器续训期间 WCE 每个完整回合后继续更新。
- WCE 与 parent 必须来自同一路网、同一控制器、同一种子，而且 WCE 清单中记录的 parent 哈希必须匹配。

单个 fixed-WCE 任务示例：

```bash
export CBWCE_PARENT='/准确的/checkpoint_001000000'
export CBWCE_WCE='/准确的/checkpoint_000660000'

bash scripts/training/06_fixed_wce.sh --mode publication \
  --network grid --controller ia2c --seed 101 \
  --parent "$CBWCE_PARENT" --wce "$CBWCE_WCE" \
  --gate "$CBWCE_GATE" --no-visualization
```

在线 WCE 将脚本换成 `scripts/training/07_online_wce.sh`。完成的在线续训应有累计 `1,500` 次 WCE 更新：离线预训练 `500` 次，加续训期间 `1,000` 次。

## 4. 五种后续训练方法与模型保存位置

五种方法必须分别从同一个共同 parent 开始，不能把前一种方法的最终模型作为后一种方法的起点。

| 方法 | 含义 | 输入模型 | 启动脚本 | 完成模型保存根目录 |
|---|---|---|---|---|
| `baseline` | 原始的顺序需求配置 | parent | `03_baseline.sh` / `10_all_baseline.sh` | `runs/revised/` |
| `random_group` | 每个 600 秒 block 均匀随机选择一个需求组 | parent | `04_random_group.sh` / `11_all_random_group.sh` | Grid：`output_coevolution/revised/`；Monaco：`output_coevolution_real/revised/` |
| `domain_randomization` | 每个 block 使用 Dirichlet 需求混合 | parent | `05_domain_randomization.sh` / `12_all_domain_randomization.sh` | 同上 |
| `fixed_wce` | 使用预训练 WCE，但冻结 WCE 参数 | parent + WCE | `06_fixed_wce.sh` / `13_all_fixed_wce.sh` | 同上 |
| `online_wce` | 使用预训练 WCE，并在续训时继续更新 WCE | parent + WCE | `07_online_wce.sh` / `14_all_online_wce.sh` | 同上 |

完整运行 40 个续训任务的统一命令是：

```bash
conda activate deeprlsc
cd /home/sdc_joran/Journal/deeprl_signal_control

export CBWCE_GATE="$PWD/runs_eval/revised/verification/no_degradation_stop_20260918_134859/gate.json"
export CBWCE_CHECKPOINTS="$PWD/runs_eval/revised/selections/publication_seed101.json"
unset CBWCE_RESUME CBWCE_STEPS CBWCE_EPISODES

bash scripts/training/15_all_continuations.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" \
  --workers 4 --no-visualization
```

注意：当前 baseline 已有已完成和未完成任务。直接重跑 `15_all_continuations.sh` 会创建重复任务，并不会自动跳过或恢复。应先逐个恢复未完成 baseline，再启动尚未开始的四种方法。`10`–`14` 与 `15` 二选一，不能两套都运行。

## 5. 当前续训的详细进度

### 5.1 Baseline

续训预算为每个任务 `1,320,000` stage steps。表中“最近观察”来自 `progress.jsonl`；“安全 checkpoint”是已完整落盘、可用于 `--resume` 的位置。二者之间的差值在恢复时会丢弃。

| 路网 | 控制器 | 最近观察 stage steps | 观察进度 | 安全 checkpoint | 完成状态 |
|---|---|---:|---:|---|---|
| Grid | IA2C | 1,320,000 | 100% | `runs/revised/grid/ia2c/seed_101/baseline/publication_20260919T110516_bffa1fe4/checkpoint_001320000` | **已完成**，累计学习步 `2,320,000` |
| Grid | MA2C | 1,320,000 | 100% | `runs/revised/grid/ma2c/seed_101/baseline/publication_20260919T110516_de174e9f/checkpoint_001320000` | **已完成**，累计学习步 `2,320,000` |
| Grid | IQL-LR | 1,206,240 | 91.38% | `runs/revised/grid/iqll/seed_101/baseline/publication_20260921T094725_7ac9cbd3/checkpoint_001201200` | **未完成**；最新尝试无 `result.json` |
| Grid | PPO | 1,320,000 | 100% | `runs/revised/grid/ppo/seed_101/baseline/publication_20260921T091601_2d3778ff/checkpoint_001320000` | **已完成**，累计学习步 `2,320,000` |
| Monaco | IA2C | 654,240 | 49.56% | `runs/revised/monaco/ia2c/seed_101/baseline/publication_20260919T195905_d093931b/checkpoint_000646800` | **未完成**；最新尝试无 `result.json` |
| Monaco | MA2C | 至少 739,200 | 至少 56.00% | `runs/revised/monaco/ma2c/seed_101/baseline/publication_20260919T195905_c04f145d/checkpoint_000739200` | **未完成**；有 checkpoint，但无 `result.json` |
| Monaco | IQL-LR | 535,920 | 40.60% | `runs/revised/monaco/iqll/seed_101/baseline/publication_20260921T101956_50fdfb9e/checkpoint_000528000` | **未完成**；最新尝试无 `result.json` |
| Monaco | PPO | 480,480 | 36.40% | `runs/revised/monaco/ppo/seed_101/baseline/publication_20260919T195905_17d304e5/checkpoint_000475200` | **未完成**；最新尝试无 `result.json` |

因此 baseline 当前是 `3/8` 完成。可恢复 checkpoint 的总体已落盘进度为：

```text
(1,320,000 × 3 + 1,201,200 + 646,800 + 739,200 + 528,000 + 475,200)
÷ (1,320,000 × 8) = 71.50%
```

### 5.2 其余四种方法

| 方法 | 已完成／计划 | 状态 |
|---|---:|---|
| `random_group` | 0/8 | 尚未开始正式论文续训 |
| `domain_randomization` | 0/8 | 尚未开始正式论文续训 |
| `fixed_wce` | 0/8 | 尚未开始正式论文续训；WCE 输入已准备好 |
| `online_wce` | 0/8 | 尚未开始正式论文续训；WCE 输入已准备好 |

## 6. 如何恢复未完成训练

恢复时必须同时保持原来的路网、控制器、方法、种子、parent、总预算，以及 `--monitor-every 50 --monitor-rollouts 3`。恢复会创建新的运行目录；不要覆盖旧目录。

以 Grid IQL-LR baseline 为例：

```bash
export CBWCE_GATE="$PWD/runs_eval/revised/verification/no_degradation_stop_20260918_134859/gate.json"
export CBWCE_PARENT="$PWD/runs/revised/grid/iqll/seed_101/parent/publication_20260917T163857_6987d164/checkpoint_001000000"
export CBWCE_RESUME="$PWD/runs/revised/grid/iqll/seed_101/baseline/publication_20260921T094725_7ac9cbd3/checkpoint_001201200"

bash scripts/training/03_baseline.sh --mode publication \
  --network grid --controller iqll --seed 101 \
  --parent "$CBWCE_PARENT" --resume "$CBWCE_RESUME" \
  --gate "$CBWCE_GATE" --monitor-every 50 --monitor-rollouts 3 \
  --no-visualization
```

其余四个未完成 baseline 使用第 5.1 节对应的安全 checkpoint 和 parent 路径，逐个恢复。每次结束后检查新运行目录中的 `result.json`，只有 `status=complete` 且 `learning_steps=2320000` 才能勾选完成。

## 7. 后续模型会保存在哪里、怎么调用

### 保存结构

```text
# baseline 最终控制器
runs/revised/<network>/<controller>/seed_101/baseline/<run_id>/checkpoint_001320000/

# 其余四种方法的最终控制器
output_coevolution/revised/grid/<controller>/seed_101/<method>/<run_id>/checkpoint_001320000/
output_coevolution_real/revised/monaco/<controller>/seed_101/<method>/<run_id>/checkpoint_001320000/

# fixed_wce 的 WCE 仍是离线 checkpoint_000660000
# online_wce 的最终 bundle 同时包含继续更新后的 WCE 状态
```

每个运行目录都应保留：

```text
manifest.json              # 模型身份、输入哈希、参数和来源
progress.jsonl             # 训练中的步数与排队进度
result.json                # 最终完成/失败状态及准确 checkpoint
episode_metrics.jsonl      # 回合指标
learner_metrics.jsonl      # 学习器指标
tensorboard/               # TensorBoard 事件
monitoring/                # 固定 Uniform 监测
checkpoint_*/              # 可恢复模型 bundle
```

### 调用最终控制器做评估

续训完成后，从该运行的 `result.json.checkpoint` 复制准确路径。评估接口仍把最终控制器 checkpoint 传给 `--parent`：

```bash
export CBWCE_FINAL='/准确的/续训运行/checkpoint_001320000'

python main.py experiment check --gate "$CBWCE_GATE"
python main.py experiment evaluate \
  --network grid --controller ia2c --seed 101 \
  --parent "$CBWCE_FINAL" --gate "$CBWCE_GATE" \
  --suite all --rollouts 10 --no-visualization
```

评估输出位于：

```text
runs_eval/revised/<network>/<controller>/seed_101/evaluate/<run_id>/
```

每个完成的续训控制器需要 `23` 个场景 × `10` 次 rollout，共 `230` 条轨迹。40 个模型合计 `9,200` 条。评估阶段冻结控制器和 WCE，不再训练模型。

## 8. 日常检查清单

训练期间可以使用：

```bash
# 查看是否仍有训练进程
ps -eo pid,lstart,etime,cmd | grep -E 'main.py experiment|scripts/training|_matrix.py'

# 查看某次运行的最新进度
tail -n 5 /准确的/run_id/progress.jsonl

# 查看最终状态
cat /准确的/run_id/result.json

# 查看所有训练曲线
tensorboard --logdir "$PWD/runs/revised" \
  --host 127.0.0.1 --port 6007 --reload_interval 15
```

每次更新本笔记时必须同时记录：

- 检查时间以及训练进程是否仍在运行；
- `result.json.status`；
- `learning_steps` 和 `stage_simulation_steps`；
- 最近观察步数与最近可恢复 checkpoint，不能混写；
- parent/WCE 的准确来源路径；
- 是否达到本阶段完整预算；
- 下一步是恢复、开始下一方法，还是开始最终评估。

## 9. 推荐执行顺序

1. 逐个恢复并完成 Grid IQL-LR 与四个 Monaco baseline，直到 baseline 达到 `8/8`。
2. 再运行 `random_group` 的 8 个任务并核对 `8/8`。
3. 运行 `domain_randomization` 的 8 个任务并核对 `8/8`。
4. 使用已完成的对应 WCE 运行 `fixed_wce` 的 8 个任务并核对 `8/8`。
5. 使用同一对应 WCE 运行 `online_wce` 的 8 个任务并核对 `8/8`。
6. 确认 40 个续训 `result.json` 都是 complete、最终累计步数均为 `2,320,000`，然后执行 9,200 条最终评估。
