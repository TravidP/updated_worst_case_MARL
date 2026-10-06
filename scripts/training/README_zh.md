# 分阶段训练启动脚本
PASS all ten corrections: /home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/verification/four_workers_20260917T140900/gate.json
## 现在开始：先筛选八份 revised 配置

方案 6 将奖励缩放移动到 learner 边界，并要求所有学习参数严格来自 INI。
旧方案 5 的 gate 与 checkpoint 已有意失效。在八个配置筛选全部完成且新 gate
通过之前，不要启动 publication parent。

```bash
conda activate deeprlsc
cd /home/sdc_joran/Journal/deeprl_signal_control

python scripts/training/16_tune_all_revised.py \
  --output output_result/revised/tuning_matrix_YYYYMMDD
```

添加 `--dry-run` 可预览全部命令；添加 `--through offline` 可只完成固定
batch 初筛。总控脚本一次只运行一个 SUMO/TensorFlow 任务，为八种组合保留
独立产物，并且只有通过 66k 全部门槛的候选才会写回正式 INI。详见
[配置筛选说明](../../docs/REVISED_TUNING.md)。

合格配置全部写回后，运行 `python -m revision.verify` 生成新 gate；只有新 gate
通过后，才可以使用脚本 `08`–`15` 开始 publication 训练。

新曲线见 [TensorBoard](http://127.0.0.1:6007/)：首先查看 `train/reward_by_learning_step`（负奖励越接近零表示成本越小）与 `monitor/mean_total_queue`（越低越好）。下文第 0–3 节介绍可选试运行，第 4–6 节说明完整训练及后续阶段。

## learner 边界奖励缩放与 TensorBoard（方案 6）

控制器使用五秒动作终点排队的负值，不除以 100；MA2C 保留 0.9 邻域权重。Monaco IA2C/MA2C 批量改为 120。训练回合仍为 6,600 秒。`train/reward_by_learning_step` 每个学习步记录实际奖励；`train/episode/mean_total_queue` 汇总完整回合。学习前、每 50 个完整回合和最终预算时执行三次配对的 600 秒 Uniform 测试，保存逐秒 CSV、NPZ 和 TensorBoard 曲线。

详见[监测设置、命令、指标及输出位置](../../docs/TRAINING_MONITORING_zh.md)。监测默认 `--monitor-every 50 --monitor-rollouts 3`；试运行可缩短间隔。监测只记录固定 rollout、TensorBoard 指标和检查点，不再因相对初始 monitor 的性能退化自动终止训练；最终模型优劣由完整配对评估判断。新奖励方案要求全新父模型和匹配当前源码的新 gate，历史检查点保留。原始文件与校验索引已归档于 `docs/history/endpoint_monitoring_20260916T095506Z/`。

### 训练过程中打开 TensorBoard

在训练电脑上打开**第二个终端**，保持原训练终端继续运行。查看 `runs/revised/` 下全部父模型和基线训练记录：

```bash
conda activate deeprlsc
cd /home/sdc_joran/Journal/deeprl_signal_control

tensorboard --logdir "$PWD/runs/revised" \
  --host 127.0.0.1 --port 6007 --reload_interval 15
```

保持该终端开启，并在浏览器访问 **[http://127.0.0.1:6007/](http://127.0.0.1:6007/)**。一个 TensorBoard 会递归读取四个 worker 写入的所有独立运行，不需要为 IA2C、MA2C、IQLL 和 PPO 分别启动服务。在左侧 **Runs** 中按启动脚本打印的 `publication_<时间>_<ID>` 选择当前批次，并取消旧实验。若此端口已经运行 TensorBoard，直接打开网页，不要重复启动同一端口。若端口被其他服务占用，可改为 `6008`，并访问相应地址。在 TensorBoard 终端按 Ctrl+C 只关闭查看服务，不停止另一个终端中的训练。

如果只想查看一个准确运行的**主训练曲线**，将 `REPLACE_WITH_RUN_ID` 替换为启动脚本打印的运行目录名，再用 6008 端口：

```bash
CBWCE_RUN="$PWD/runs/revised/grid/ia2c/seed_101/parent/REPLACE_WITH_RUN_ID"
tensorboard --logdir "$CBWCE_RUN/tensorboard" \
  --host 127.0.0.1 --port 6008 --reload_interval 15
```

打开 [http://127.0.0.1:6008/](http://127.0.0.1:6008/)。不要把 `REPLACE_WITH_RUN_ID` 原样执行；`CBWCE_RUN` 必须是实际存在的准确目录。若还要显示该运行每轮监测的逐秒曲线，将参数改为 `--logdir "$CBWCE_RUN"`。TensorBoard 递归查找事件文件，不会启动或恢复训练。

| 运行目录／标签 | 查看内容 |
|---|---|
| `<run>/tensorboard` | 持续训练及监测汇总；查看当前学习进展时选择此运行。 |
| `<run>/monitoring/round_000000/tensorboard` | 学习前的首次 600 秒监测，横轴为仿真秒数；它是固定初始参考，不随训练持续更新。 |
| `train/reward_by_learning_step` | 实际学习奖励随累计学习步数变化；负值越接近零，排队成本越小。MA2C 显示各智能体邻域奖励的均值。 |
| `train/episode/mean_total_queue` | 完整 6,600 秒训练回合的平均排队车辆数，越低越好；最终不完整回合单独记入 `train/partial_episode/*`。 |
| `monitor/mean_total_queue` | 三次固定 Uniform 测试的平均排队，横轴为学习步数，越低越好。 |
| `monitor/mean_current_wait_seconds` | 对应监测测试中的平均当前等待时间，越低越好。 |

在 **SCALARS** 中选择主运行，横轴选择 **STEP**。网页没有更新时点击刷新。事件文件采用缓冲写入（约 30 秒），上述命令每 15 秒重新读取，因此不是即时显示。完整回合指标在回合完成后出现；固定监测在学习前、每 50 个完整回合和最终预算时执行。初始监测期间还没有训练奖励点。Smoothing 只影响显示，不改变原始奖励。

`runs/revised/` 包含父模型和基线续训。对比方法续训使用 `output_coevolution/revised/`（grid）或 `output_coevolution_real/revised/`（Monaco）；离线 WCE 使用 `output_adversary/revised/` 或 `output_adversary_monaco/revised/`。把 `--logdir` 指向相应根目录或准确运行的 `tensorboard/` 目录即可。WCE 曲线使用宏步数，不产生控制器学习奖励点。


[English](README.md) · [完整实验方案](../../reviewer_revision_plan_zh.md)

`01`–`07` 脚本通过已有 `python main.py experiment` 入口，为**一个路网／控制器／种子启动一个阶段**，矩阵脚本 `08`–`15` 使用最多四个并行 worker 扩展到所选路网／控制器矩阵，参见第 5–6 节。每个脚本先打印双语阶段标题、方法、真实预算、所选检查点、输出位置及准确命令；Python 使用无缓冲输出，及时显示运行器发出的回合进度。

## 0. 准备环境并选择实验

从仓库根目录执行：

```bash
conda activate deeprlsc
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

export CBWCE_NETWORK=grid
export CBWCE_CONTROLLER=ia2c
export CBWCE_MODE=pilot
export CBWCE_SEED=9001
unset CBWCE_PARENT CBWCE_WCE CBWCE_RESUME CBWCE_GATE
unset CBWCE_STEPS CBWCE_EPISODES CBWCE_VISUALIZATION

python main.py experiment prepare
python main.py experiment check
```

路网为 `grid`、`monaco`；控制器为 `ia2c`、`ma2c`、`iqll`、`ppo`。试运行使用 `9001` 等独立种子，不与论文种子 `101` 重叠。

试运行脚本默认开启 SUMO 可视化，论文脚本默认关闭。试运行需要 `sumo-gui` 和可访问的桌面显示；无显示环境时添加 `--no-visualization`。原始 Python CLI 和自动验证仍保持已有的无窗口默认行为。

## 1. 训练共同父模型

先预览，不启动 SUMO，也不创建输出文件：

```bash
bash scripts/training/01_parent.sh --dry-run
```

启动所选父模型训练：

```bash
bash scripts/training/01_parent.sh
```

使用默认试运行设置时，相当于：

```bash
python -u main.py experiment parent --network grid --controller ia2c \
  --seed 9001 --pilot --visualization --steps 160 --checkpoint-every 1 \
  --monitor-every 50 --monitor-rollouts 3
```

脚本还会显式传入标题中打印的唯一输出目录。运行完成后读取其 `result.json`，确认 `status=complete`，复制准确的 `checkpoint` 值，并设置后续阶段的输入：

```bash
export CBWCE_PARENT='/replace/with/completed/parent/checkpoint_000000160'
```

必须替换为你的真实运行路径，不能猜测或按最新文件名选择。检查点目录旁必须保留原运行的 `manifest.json`。论文父模型的后缀为 `checkpoint_001000000`。

## 2. 针对冻结父模型训练 WCE

```bash
bash scripts/training/02_wce.sh --dry-run
bash scripts/training/02_wce.sh
```

控制器保持冻结。试运行默认两个 WCE 回合，即 2,640 个冻结控制器仿真步；论文阶段为 500 回合，即 660,000 个冻结控制器仿真步，均不增加控制器训练步数。

成功完成后，将 WCE 运行的 `result.json.checkpoint` 复制到：

```bash
export CBWCE_WCE='/replace/with/completed/wce/checkpoint_000002640'
```

论文 WCE 后缀为 `checkpoint_000660000`。该 WCE 必须针对 `CBWCE_PARENT` 指定的准确父模型训练。接下来的五种比较中保持这两个变量不变。

## 3. 五种续训方法

按需逐条执行。每条命令重新加载**同一个共同父模型**，不是上一种续训的最终控制器。

```bash
# III-1: original sequential demand baseline
bash scripts/training/03_baseline.sh

# III-2: one uniformly selected demand group per block
bash scripts/training/04_random_group.sh

# III-3: a Dirichlet demand mixture per block
bash scripts/training/05_domain_randomization.sh

# III-4: pretrained WCE, with model parameters frozen
bash scripts/training/06_fixed_wce.sh

# III-5: the same pretrained WCE, updated after each full episode
bash scripts/training/07_online_wce.sh
```

可为任何命令追加 `--dry-run` 先预览。前三种方法只使用 `CBWCE_PARENT`；后两种还使用同一个 `CBWCE_WCE`。即使已导出 WCE 变量，前三种也不会加载 WCE。续训不会改变父模型检查点或你当前 shell 的变量。

| 阶段／脚本 | 试运行默认预算 | 论文预算 | 输出根目录 |
|---|---:|---:|---|
| I / [01_parent.sh](01_parent.sh) | 160 个训练步 | 1,000,000 个训练步 | `runs/revised/` |
| II / [02_wce.sh](02_wce.sh) | 2 个 WCE 回合 | 500 个 WCE 回合 | Grid：`output_adversary/revised/`；Monaco：`output_adversary_monaco/revised/` |
| III-1 / [03_baseline.sh](03_baseline.sh) | +2,640 个训练步 | +1,320,000 个训练步 | `runs/revised/` |
| III-2 / [04_random_group.sh](04_random_group.sh) | +2,640 个训练步 | +1,320,000 个训练步 | Grid：`output_coevolution/revised/`；Monaco：`output_coevolution_real/revised/` |
| III-3 / [05_domain_randomization.sh](05_domain_randomization.sh) | +2,640 个训练步 | +1,320,000 个训练步 | 同上续训目录 |
| III-4 / [06_fixed_wce.sh](06_fixed_wce.sh) | +2,640 个训练步 | +1,320,000 个训练步 | 同上续训目录 |
| III-5 / [07_online_wce.sh](07_online_wce.sh) | +2,640 个训练步 | +1,320,000 个训练步 | 同上续训目录 |

使用默认试运行父模型时，五种最终控制器均为 **2,800 个训练步**；论文控制器均为 **2,320,000**。WCE 计算单独统计。输出继续采用 `<root>/<network>/<controller>/seed_<seed>/<stage-or-method>/<run_id>/`。

## 4. 切换为论文模式

本方案的新 gate 已通过全部八组验证。优先使用本页开头的实际路径，并用 `check --gate` 做只读检查。旧版 gate 不能用于本方案；只有当前 gate 失效或实现／配置变更时，才需要重新运行完整验证：

```bash
CBWCE_VERIFY_DIR="$PWD/runs_eval/revised/verification/manual_$(date +%Y%m%d_%H%M%S)"
python main.py experiment verify --workers 4 --output "$CBWCE_VERIFY_DIR" \
  && export CBWCE_GATE="$CBWCE_VERIFY_DIR/gate.json"
```

上述命令会实际执行无窗口测试与试运行，不是只读检查。若重新验证成功，后续命令使用刚生成的 gate；不要再切回旧路径。下面是使用本次已通过 gate 的**单个父模型**示例，不要与八父模型命令同时运行。创建全新论文父模型前，清除试运行检查点选择：

```bash
export CBWCE_MODE=publication
export CBWCE_SEED=101
export CBWCE_GATE="$PWD/runs_eval/revised/verification/endpoint_monitoring_20260916_v1/gate.json"
unset CBWCE_PARENT CBWCE_WCE CBWCE_RESUME
unset CBWCE_STEPS CBWCE_EPISODES CBWCE_VISUALIZATION

bash scripts/training/01_parent.sh --dry-run
bash scripts/training/01_parent.sh
```

每个阶段完成后设置真实论文检查点：

```bash
export CBWCE_PARENT='/replace/with/publication/parent/checkpoint_001000000'
bash scripts/training/02_wce.sh

export CBWCE_WCE='/replace/with/publication/wce/checkpoint_000660000'
bash scripts/training/03_baseline.sh
bash scripts/training/04_random_group.sh
bash scripts/training/05_domain_randomization.sh
bash scripts/training/06_fixed_wce.sh
bash scripts/training/07_online_wce.sh
```

这些是由你依次执行的命令，不是自动启动的实验队列。对两个路网、四种控制器及一个论文种子重复流程，得到 8 个父模型、8 次离线 WCE 训练和 40 次续训。论文预算读取 `config/revised/protocol.json`，不传入试运行专用的预算覆盖选项。试运行检查点不能代替论文父模型。

## 5. 使用最多八个 worker 训练全部路网和控制器

[08_all_parents.sh](08_all_parents.sh) 执行**初始父模型训练**。默认 `--workers 4` 时仍先运行 Grid 四任务、再运行 Monaco 四任务；使用 `--workers 8` 时两个路网的八个任务进入同一批。每个任务独占模型、SUMO 进程、输出目录和随机数流。允许范围为 `--workers 1..8`。

| 单个种子的顺序 | 路网 | 控制器 |
|---|---|---|
| 1 | `grid` | `ia2c` |
| 2 | `grid` | `ma2c` |
| 3 | `grid` | `iqll` |
| 4 | `grid` | `ppo` |
| 5 | `monaco` | `ia2c` |
| 6 | `monaco` | `ma2c` |
| 7 | `monaco` | `iqll` |
| 8 | `monaco` | `ppo` |

**预览八次试运行**（不训练、不启动 SUMO、不创建输出文件）：

```bash
conda activate deeprlsc
unset CBWCE_RESUME CBWCE_STEPS CBWCE_EPISODES
bash scripts/training/08_all_parents.sh \
  --mode pilot --seed 9001 --visualization --dry-run
```

删除 `--dry-run` 后执行八次试运行，每个父模型训练 160 个学习步。无桌面环境时使用 `--no-visualization`。这些短父模型试运行不能替代完整验证门槛。

**正常／完整训练：八个父模型**，使用一个论文种子，每个父模型训练 1,000,000 个学习步：

```bash
export CBWCE_GATE="$PWD/runs_eval/revised/verification/endpoint_monitoring_20260916_v1/gate.json"
unset CBWCE_RESUME CBWCE_STEPS CBWCE_EPISODES
bash scripts/training/08_all_parents.sh \
  --mode publication --seed 101 --gate "$CBWCE_GATE" \
  --monitor-every 50 --monitor-rollouts 3 --no-visualization
```

这就是完整父模型矩阵：**8 个父模型，每个路网／控制器组合仅训练一次，种子为 `101`**。添加 `--dry-run` 预览，或使用 `--visualization` 显示 SUMO。论文训练需要当前有效且通过的验证门槛。多种子选项已移除；`--all-seeds` 现在会明确报错。矩阵显式选择覆盖 `CBWCE_NETWORK` 和 `CBWCE_CONTROLLER`。如果此前导出了 `101` 以外的论文种子，请用 `--seed 101` 覆盖。

输出仍分别保存在：

```text
runs/revised/<network>/<controller>/seed_<seed>/parent/<run_id>/
  manifest.json
  progress.jsonl
  result.json
  episode_metrics.jsonl
  learner_metrics.jsonl
  tensorboard/
  monitoring/round_000000/  # 学习前测试；后续 round 按完整回合编号
  checkpoint_001000000/    # 成功完成的论文父模型
```

一个任务失败时，同批次的其他任务会收到 SIGINT，以便记录失败状态并清理 SUMO；之后不再启动新批次。按 Ctrl+C 时采用同样的清理方式。重新执行矩阵会创建新运行，包括已完成的组合；不会自动跳过或恢复。中断后使用 `01_parent.sh --resume` 和该阶段准确的检查点恢复单个组合，再逐个执行剩余组合。不要为整个矩阵导出同一个恢复检查点。

父模型训练后，针对每个路网／控制器／种子按第 2–3 节操作：选择该父模型已完成的 `result.json.checkpoint`，训练对应的冻结控制器 WCE，再让五种续训方法从同一个父模型及匹配的 WCE 开始。`08_all_parents.sh` 不自动执行后续阶段或评估，保留显式检查点选择，不按最新文件名猜测输入。

## 6. 使用四个 worker 执行 WCE 和全部续训方法

这些启动脚本覆盖与 `08_all_parents.sh` 相同的路网／控制器矩阵，调用已有单次训练脚本，保留预算、可视化选项、输出根目录及学习行为。

| 矩阵启动脚本 | 已有单次训练脚本 | 种子 `101` 的运行数 |
|---|---|---:|
| [09_all_wce.sh](09_all_wce.sh) | `02_wce.sh` | 8 |
| [10_all_baseline.sh](10_all_baseline.sh) | `03_baseline.sh` | 8 |
| [11_all_random_group.sh](11_all_random_group.sh) | `04_random_group.sh` | 8 |
| [12_all_domain_randomization.sh](12_all_domain_randomization.sh) | `05_domain_randomization.sh` | 8 |
| [13_all_fixed_wce.sh](13_all_fixed_wce.sh) | `06_fixed_wce.sh` | 8 |
| [14_all_online_wce.sh](14_all_online_wce.sh) | `07_online_wce.sh` | 8 |
| [15_all_continuations.sh](15_all_continuations.sh) | `03`–`07`，按方法顺序 | 40 |

`15_all_continuations.sh` 完成一种方法后才启动下一种：baseline、random_group、domain_randomization、fixed_wce、online_wce。默认使用四个 worker，此时 Grid 和 Monaco 分为两个四任务批次；使用 `--workers 8` 时同一方法的两个路网同时运行。允许范围为 `1..8`。所有方法重新加载各自选定的共同父模型，不从其他方法的结果继续训练。请在 `10`–`14` 和 `15` 之间选择一种执行方式；两种方式都运行会创建新尝试并重复续训。

### 6.1 显式选择已完成的父模型

一个 `CBWCE_PARENT` 无法代表八个不同父模型。先创建选择模板，再填写实际完成的结果路径。一个论文种子的矩阵示例：

```bash
conda activate deeprlsc
unset CBWCE_RESUME CBWCE_STEPS CBWCE_EPISODES
export CBWCE_GATE="$PWD/runs_eval/revised/verification/endpoint_monitoring_20260916_v1/gate.json"
export CBWCE_CHECKPOINTS='runs_eval/revised/selections/publication_seed101.json'

bash scripts/training/09_all_wce.sh --mode publication --seed 101 \
  --write-template "$CBWCE_CHECKPOINTS"
```

模板命令只写入指定 JSON 文件，不训练、不要求验证门槛，也不检查占位路径；已有文件不会被覆盖。用编辑器打开该文件，其八行记录分别采用以下格式：

```json
{
  "network": "grid",
  "controller": "ia2c",
  "seed": 101,
  "parent_result": "/replace/with/exact/parent/run_id/result.json",
  "wce_result": "/replace/with/exact/wce/run_id/result.json"
}
```

顶层对象包含 `"version": 1` 和 `"runs": [...]`。先将每个 `parent_result` 替换为对应成功父模型运行的准确 `result.json` 路径；WCE 训练完成前可保留 `wce_result` 占位值。辅助程序读取 `result.json.checkpoint`，不扫描最新文件名。允许绝对路径或相对仓库根目录的路径，即使从其他目录启动脚本也采用相同规则。

启动任何任务前，共享 [_matrix.py](_matrix.py) 会检查整个所选矩阵：身份是否重复／缺失、运行是否完成、原始清单完整性、路网／控制器／种子／阶段／模式是否一致，以及检查点文件哈希。论文输入必须具有规定的完成预算。固定／在线 WCE 还要求 WCE 记录的父模型哈希匹配所选父模型。实际启动每个任务时，由已有运行器检查运行环境、验证门槛和精确模型签名。预览检查真实选择记录及启动参数，但不启动 SUMO，也不创建运行输出。

### 6.2 为整个矩阵训练 WCE

先预览，再执行八次冻结父模型的 WCE 训练：

```bash
bash scripts/training/09_all_wce.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" \
  --no-visualization --dry-run

bash scripts/training/09_all_wce.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" --no-visualization
```

每次训练 WCE 500 个回合／660,000 个冻结控制器仿真步，控制器保持 1,000,000 个学习步。Grid 输出位于 `output_adversary/revised/`，Monaco 输出位于 `output_adversary_monaco/revised/`，其后沿用路网／控制器／种子／阶段／运行目录层级。

完成后，在每行 `wce_result` 填入对应 WCE 运行的 `result.json` 路径。基线、随机组和域随机化仅需要 `parent_result`，可以在 WCE 尚未完成时运行。固定 WCE、在线 WCE 和组合续训启动脚本需要所有所选身份的两个字段均有效。

### 6.3 执行基线和四种比较方法

对全部八个组合执行所需方法：

```bash
# Baseline: original sequential demand schedule
bash scripts/training/10_all_baseline.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" --no-visualization

# Random demand groups
bash scripts/training/11_all_random_group.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" --no-visualization

# Dirichlet demand mixtures
bash scripts/training/12_all_domain_randomization.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" --no-visualization

# Fixed WCE model parameters
bash scripts/training/13_all_fixed_wce.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" --no-visualization

# Online WCE model updates
bash scripts/training/14_all_online_wce.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" --no-visualization
```

也可以用**一个命令顺序执行全部五种方法**，遇到第一个错误即停止：

```bash
bash scripts/training/15_all_continuations.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" --no-visualization
```

每次续训在共同父模型上增加 1,320,000 个控制器学习步，最终达到 2,320,000。基线输出仍位于 `runs/revised/`；比较方法输出仍位于 Grid 的 `output_coevolution/revised/` 和 Monaco 的 `output_coevolution_real/revised/`。每次运行独立保存清单、进度、结果和检查点包。添加 `--dry-run` 进行预览；使用 `--visualization` 显示 SUMO。

### 6.4 试运行与完整单种子研究

第 5–6.3 节已覆盖完整研究：**8 个父模型 → 8 次 WCE 训练 → 40 次续训 → 9,200 次评估**。整个流程使用同一个八行选择文件，在阶段 II 后补全 WCE 结果路径。所有论文命令使用 `--seed 101`，不再运行五种子矩阵。

试运行使用单独选择文件和已完成的试运行父模型，不能使用论文检查点：

```bash
export CBWCE_CHECKPOINTS='runs_eval/revised/selections/pilot_seed9001.json'
bash scripts/training/09_all_wce.sh --mode pilot --seed 9001 \
  --write-template "$CBWCE_CHECKPOINTS"

# Fill parent_result entries, then preview eight WCE pilots.
bash scripts/training/09_all_wce.sh --mode pilot --seed 9001 \
  --checkpoints "$CBWCE_CHECKPOINTS" --visualization --dry-run

# Remove --dry-run above to train WCE; fill wce_result entries afterwards.
# Preview all 40 continuation pilots using those completed checkpoints.
bash scripts/training/15_all_continuations.sh --mode pilot --seed 9001 \
  --checkpoints "$CBWCE_CHECKPOINTS" --visualization --dry-run
```

试运行默认每次 WCE 训练两个回合，每次续训增加 2,640 个学习步。`--episodes` 仅覆盖试运行 WCE 预算；`--steps` 仅覆盖试运行续训预算。论文预算不能覆盖。切换阶段前清除旧 `CBWCE_STEPS`／`CBWCE_EPISODES`。`--checkpoints` 覆盖 `CBWCE_CHECKPOINTS`；逐行显式选择覆盖已导出的 `CBWCE_PARENT`／`CBWCE_WCE`。矩阵路网／控制器选择覆盖对应的单次运行环境变量。

**恢复：** 所有启动脚本在失败或终端 Ctrl+C 后停止。重新执行会创建新尝试，不自动跳过、恢复或修改检查点映射。使用 `02`–`07` 为失败的单个任务恢复训练，传入原始父模型／WCE 选择及准确的同阶段 `--resume` 检查点。随后逐个启动剩余组合，避免重复。组合启动脚本不启动评估或报告；训练后按完整审稿修订指南执行对应命令。

对于当前 seed 101 的剩余四种续训，使用状态感知的一键入口：

```bash
bash scripts/training/16_remaining_continuations_with_cleanup.sh --dry-run
bash scripts/training/16_remaining_continuations_with_cleanup.sh
```

该入口默认使用 4 workers，以全局队列并行执行最多四个 Grid／Monaco 任务；自动验证或生成当前源码对应的 gate，跳过已完成任务，并从每个任务最高的完整检查点恢复。单任务失败不会取消其他任务，批次结束后最多重试两次。训练期间每 15 分钟清理 60 分钟前的非第十个原始 episode/trip 文件；检查点、结果、指标及监测数据不会删除。默认值为 4；可使用 `--workers 1..8`、`--gate PATH` 或 `--force-new-gate`。重复运行同一命令是安全的。

## 参数、进度与恢复

恢复仅限同一新奖励方案的检查点；不能通过 `--resume` 把历史旧奖励模型转换成新模型。必须保持原有 `--monitor-every` 和 `--monitor-rollouts`；已完成的监测轮次会复用，不会重复执行。

命令行选项优先于环境变量；任何编号脚本都支持 `--help`。即使从其他目录调用脚本，相对检查点／gate 路径仍按仓库根目录解释。

| 环境变量 | 默认值／用途 |
|---|---|
| `CBWCE_MODE` | `pilot`；论文训练须显式选择 `publication` |
| `CBWCE_NETWORK`、`CBWCE_CONTROLLER` | `grid`、`ia2c` |
| `CBWCE_SEED` | 未设置时：试运行 `9001`，论文 `101` |
| `CBWCE_PARENT`、`CBWCE_WCE` | 按阶段明确选择检查点目录 |
| `CBWCE_GATE` | 论文训练必需 |
| `CBWCE_VISUALIZATION` | 未设置时：试运行 `on`，论文 `off` |
| `CBWCE_STEPS` | 可选试运行父模型／续训预算；续训必须为完整 1,320 步回合的倍数 |
| `CBWCE_EPISODES` | 可选试运行 WCE 回合数 |
| `CBWCE_RESUME` | 同阶段完整检查点；开始不同阶段／方法前应清除 |
| `CBWCE_MONITOR_EVERY` | 默认 `50` 个完整回合；论文运行固定为 `50` |
| `CBWCE_MONITOR_ROLLOUTS` | 默认 `3` 次配对监测；论文运行固定为 `3` |
| `CBWCE_WORKERS` | 矩阵脚本 `08`–`15` 每个路网／方法批次的并发任务数；默认 `4`，范围 `1..4` |
| `CBWCE_PYTHON` | 可执行文件路径／名称；默认使用当前环境的 `python` |

不改变已导出选择的示例：

```bash
bash scripts/training/01_parent.sh --mode pilot --network monaco \
  --controller ma2c --seed 9002 --no-visualization --dry-run

bash scripts/training/07_online_wce.sh --resume '/exact/same-stage/checkpoint'
```

恢复时保持原阶段、方法、种子、父模型／WCE 身份和阶段总预算不变。非默认试运行须重复原来的 `--steps`／`--episodes`。恢复创建新输出目录，保存检查点之后的未保存工作被丢弃。论文常规检查点每十个完整回合及阶段完成时保存；这些试运行脚本每回合及完成时保存。

开始前的标题显示训练阶段；执行时已有运行器打印回合完成进度，例如 `grid ia2c parent episode=1 simulation_steps=160 learning_steps=160`。每 120 个控制器转移还写入 `progress.jsonl`，其中排队值对应最近 600 秒。不要把日志行之间的间隔理解为预热期；没有交通预热，160 步等于 800 秒仿真。

Ctrl+C 中断当前 Python 进程及其拥有的 SUMO 会话。脚本使用 `exec` 而非日志管道，保留信号传递与退出状态。控制台输出位于终端，科学记录和检查点保存到打印的运行目录。完成依据应是最终 `result.json`，而不是标题中预告的检查点路径。

`--dry-run` 仅计算临时唯一路径，不创建运行；之后正式调用会选择另一个新路径。它检查选择与预算格式，但不验证检查点内容、gate 或显示可用性，这些由已有运行器在启动时检查。打开脚本或阅读指南不会启动训练。
