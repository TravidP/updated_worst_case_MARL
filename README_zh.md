# CB-WCE：面向变化交通需求的多智能体信号控制

## 新奖励与 TensorBoard 监测（方案 5）

控制器使用五秒动作终点排队的负值，不除以 100；MA2C 保留 0.9 邻域权重。Monaco IA2C/MA2C 批量改为 120。训练回合仍为 6,600 秒。`train/reward_by_learning_step` 每个学习步记录实际奖励；`train/episode/mean_total_queue` 汇总完整回合。学习前、每 50 个完整回合和最终预算时执行三次配对的 600 秒 Uniform 测试，保存逐秒 CSV、NPZ 和 TensorBoard 曲线。

详见[监测设置、命令、指标及输出位置](docs/TRAINING_MONITORING_zh.md)。监测默认 `--monitor-every 50 --monitor-rollouts 3`；试运行可缩短间隔。监测只记录固定 rollout、TensorBoard 指标和检查点，不再因相对初始 monitor 的性能退化自动终止训练；最终模型优劣由完整配对评估判断。新奖励方案要求全新父模型与匹配当前源码的新 gate，历史检查点保留。原始文件与校验索引已归档于 `docs/history/endpoint_monitoring_20260916T095506Z/`。

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


[English](README.md) · [中文交互工作台](docs/site/dist/zh.html) · [私有在线指南](https://cb-wce-training-workspace.loyal-bowl-4834.chatgpt.site/zh.html) · [集成报告](docs/INTEGRATION_REPORT.md) · [完整实验方案](reviewer_revision_plan_zh.md)

本项目研究如何通过有挑战性的交通需求训练，提高交通信号控制器的鲁棒性。控制器学习信号动作；CB-WCE 根据交通状态生成十一种需求组的混合权重。研究目标是提升未见需求下的表现，最终结论必须由配对评估与不确定性分析支持。

支持 **5×5 网格与摩纳哥子网**，以及 **IA2C、MA2C、IQL-LR（`iqll`）和 PPO**。修正实现已集成到 `agents/`、`envs/` 和 `experiments/`。`revision/` 保留兼容入口与历史验证证据。原始数据、检查点和结果保留原路径。

## 开始使用

从仓库根目录运行：

```bash
conda activate deeprlsc
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
python main.py experiment prepare
python main.py experiment check
python main.py experiment dashboard
```

打开 [本地中文工作台](http://127.0.0.1:8765/zh.html)。默认使用试运行模式，一次启动一个任务。页面提供输入检查、命令预览、检查点选择、实时进度、停止和恢复。停止不会删除旧文件；恢复创建新尝试。若尚未保存完整检查点，必须重新开始该阶段。

已验证的训练环境为 Python 3.6.13、TensorFlow 1.12.0、NumPy 1.19.5；使用本机 SUMO。`environment.yml` 保留为历史环境记录；不要在已有环境中盲目重建。新机器先检查平台兼容性。网页本身无需安装前端依赖。

私有 Sites 页面提供指南、命令和已导出结果，实际训练必须在本地工作台执行。直接打开 HTML 文件时同样为只读指南模式。

## SUMO 可视化

**交互试运行：** 下方父模型、冻结 WCE、续训、评估和恢复示例均加入 `--visualization`。在工作台选择运行类型时，**试运行默认开启**可视化，**论文实验默认关闭**。手动试运行也可选择关闭；重新打开页面或切换语言时保留已保存的显示选择。旧表单若保存为关闭，可先选论文实验再切回试运行，或直接选择开启。自动 `verify` 检查保持无窗口运行。CLI 本身在省略两个显示选项时仍默认无窗口。

在 `parent`、`wce`、`continue` 或 `evaluate` 命令中添加 `--visualization`，即可打开独立的本地 SUMO 窗口。使用 `--no-visualization`，或同时省略两个选项，则保持关闭。两个选项互斥；不要使用 `--visualization false`。选项放在 `python main.py experiment <stage>` 后面；`revision.runner` 兼容入口同样支持。

```bash
# SUMO window on
python main.py experiment parent --network grid --controller ia2c \
  --seed 9001 --pilot --steps 160 --visualization

# SUMO window off (also the default when neither flag is supplied)
python main.py experiment parent --network grid --controller ia2c \
  --seed 9001 --pilot --steps 160 --no-visualization
```

本地工作台提供相同的 **SUMO 可视化** 开关。需要安装 `sumo-gui` 并使用图形桌面；Linux 中须有可访问的 `DISPLAY`。计时比较应关闭可视化。更新后请重启空闲的仪表板服务并刷新页面。详见[显示行为、记录与验证](docs/VISUALIZATION.md)。本次源代码修改使旧论文准入记录失效；论文训练前需重新验证。

## 目录用途

| 目录 | 内容 |
|---|---|
| `agents/` | 原有策略，以及统一控制器、WCE 和循环网络实现 |
| `envs/` | 原有路网环境，以及修正的每秒测量与 SUMO 交互 |
| `experiments/` | 实验方案、路径、训练、评估、检查点、统计和本地服务 |
| `config/revised/` | 统一方案与生效配置副本 |
| `data_traffic/revised/` | 网格 train/validation/test 数据 |
| `real_net_subnet/demand_groups/revised/` | 摩纳哥 train/validation/test 数据 |
| `runs/revised/` | 公共父模型和基线续训 |
| `output_adversary*/revised/` | 两个路网的冻结父模型 WCE 训练 |
| `output_coevolution*/revised/` | 两个路网的四种比较方法续训 |
| `runs_eval/revised/` | 评估、验证和任务日志 |
| `output_result/revised/`、`figs/revised/` | 统计表、仪表板数据与科学图表 |
| `docs/history/` | 修改前的精确文件副本与校验索引 |
| `docs/site/dist/` | 中英文交互工作台与研究漫画 |

每次训练使用 `<阶段目录>/<路网>/<控制器>/seed_<种子>/<阶段或方法>/<运行标识>/`。评估再区分数据集、场景、轨迹及尝试。所有历史目录保持原位置。

## 训练流程

可使用[七个编号 Shell 启动脚本](scripts/training/README_zh.md)，分别执行父模型训练、冻结父模型 WCE 训练及五种续训。每个脚本打印当前阶段、设置、准确命令和输出路径；追加 `--dry-run` 可预览。另有[英文说明](scripts/training/README.md)。

| 阶段 | 预算 | 输出 |
|---|---:|---|
| 公共父模型 | 1,000,000 控制器训练步 | 8 个独立父模型 |
| 冻结父模型训练 WCE | 500 回合；660,000 冻结控制器仿真步 | 8 个 WCE |
| 五种方法续训 | 每种增加 1,320,000 训练步 | 40 个最终控制器 |
| 配对评估 | 23 场景 × 10 轨迹 | 完整矩阵 9,200 轨迹 |

五种方法为 `baseline`、`random_group`、`domain_randomization`、`fixed_wce`、`online_wce`。所有方法从同一个对应父模型开始，最终累计 2,320,000 控制器训练步。固定和在线 WCE 使用相同预训练 WCE。固定的是 **模型参数**，需求混合权重仍可随状态变化。

先运行验证：

```bash
python main.py experiment verify --workers 4
```

使用输出中的新 `gate.json` 启动论文训练。移动代码或更改输入会使旧准入记录失效。唯一论文训练种子为 `101`；试运行使用 `9001` 等独立种子。

单个短试运行：

```bash
python main.py experiment parent --network grid --controller ia2c \
  --seed 9001 --pilot --visualization --steps 160
```

程序打印实际输出路径和检查点。续训时明确选择检查点，不使用“最新文件”猜测来源。将下列引号中的路径替换为真实论文检查点和准入文件；短试运行父模型不是论文父模型：

```bash
python main.py experiment wce --network grid --controller ia2c \
  --seed 101 --gate '/replace/with/passing/gate.json' --parent '/replace/with/publication/parent/checkpoint_001000000'

python main.py experiment continue --network grid --controller ia2c \
  --seed 101 --method online_wce --gate '/replace/with/passing/gate.json' \
  --parent '/replace/with/publication/parent/checkpoint_001000000' --wce '/replace/with/publication/wce/checkpoint_000660000'
```

恢复时添加 `--resume <同阶段检查点>`，保留原始阶段、方法、种子、总预算和父检查点。正常检查点默认每十个完整回合保存一次；可使用 `--checkpoint-every 1`。文件名后缀是当前阶段的仿真步，不是累计控制器训练步。完整恢复要求模型、优化器、缓冲区、随机流和计数器；旧权重文件不能代替。

## 测试与报告

交通由各路网归一化后的 OD 配置生成，包含重分配、固定混合、方向切换与高峰负载。每个场景生成十份完整 JSON 车辆计划，固定路径、出发时间和速度因子。详见[验证与测试交通生成说明](reviewer_revision_plan_zh.md#traffic-generation)，其中解释两个路网、种子用途、文件位置及期望需求率与抽样车辆数的区别。

```bash
python main.py experiment prepare --materialize --network grid
python main.py experiment prepare --materialize --network monaco

python main.py experiment evaluate --network grid --controller ia2c \
  --seed 101 --parent '/replace/with/final/checkpoint_001320000' --suite all

python main.py experiment report --input '/replace/with/selected/run'
```

`--suite all` 使用十一种已见需求和十二种新场景；`validation` 使用六个独立验证场景。每次评估只运行所选最终控制器，场景依次执行，不自动启动完整训练矩阵。试运行检查点必须加 `--pilot`；可用 `--rollouts 1` 做小规模验证。

报告输出 CSV、PNG、SVG 和 `dashboard.json`。在工作台“结果与图表”导入 JSON。每个场景报告十次评估轨迹的均值和样本标准差。仅使用一个训练种子，因此不报告训练种子标准差或置信区间；比较结果以该次训练模型为条件。热力图使用统一 `[1200,2400)` 窗口与色标。

主指标为每秒采集的受控进口道平均总排队车辆数，单位为车辆，越低越好。不完整轨迹不能补零后进入结果。不要将试运行曲线当作论文比较结果。

## 历史入口与问题排查

`python main.py train/evaluate` 和原有独立训练脚本仍为历史流程。新的修正实验统一使用 `python main.py experiment ...`。原有 README 已保存在 [历史副本](docs/history/pre_integration/README.md.txt)。历史 `introduction.md` 含过时环境、路径和 Git 操作建议，不作为新实验指南。

- 找不到 TensorFlow：确认已激活 `deeprlsc`，不要使用默认 Python 3.13。
- 无法启动 SUMO：检查 `sumo --version` 和本地 TraCI 连接权限。
- 检查点不兼容：核对路网、控制器、阶段、预算以及父模型身份。
- 输出目录已存在：使用新运行目录，保留原尝试。
- 验证记录失效：更改代码或输入后重新验证。
- 报告发现重复轨迹：只选择明确的一次有效尝试，不合并重复重跑。

## 原项目来源与引用

本项目基于 Tianshu Chu 等人的多智能体交通信号控制实现，保留 [MIT 许可证](LICENSE)。请引用原方法来源，并独立说明本项目的 CB-WCE 扩展：

```bibtex
@article{chu2019multi,
  title={Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control},
  author={Chu, Tianshu and Wang, Jie and Codec{\`a}, Lara and Li, Zhaojian},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2019},
  publisher={IEEE}
}
```

[研究漫画](docs/site/dist/assets/cbwce-comic-zh-v1.png) · [完整中文实验方案](reviewer_revision_plan_zh.md) · [历史审计](reviewer_revision_audit_2026-09-14.md)

## 仓库与便携结果

仓库保留源码、输入、配置和 Grid/Monaco 精简结果。训练输出、checkpoint、生成的冻结需求和完整时序仅保留本地。见[数据管理说明](docs/REPOSITORY_DATA_POLICY.md)和[便携包恢复步骤](docs/evaluation_workbook/grid_results_site/RESTORE.md)。网站使用 Python 3.10+，与旧版训练环境分开。模型和原始数据目前尚无公开 Release。
