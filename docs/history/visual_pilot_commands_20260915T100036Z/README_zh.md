# CB-WCE：面向变化交通需求的多智能体信号控制

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

| 阶段 | 预算 | 输出 |
|---|---:|---|
| 公共父模型 | 1,000,000 控制器训练步 | 40 个独立父模型 |
| 冻结父模型训练 WCE | 500 回合；660,000 冻结控制器仿真步 | 40 个 WCE |
| 五种方法续训 | 每种增加 1,320,000 训练步 | 200 个最终控制器 |
| 配对评估 | 23 场景 × 10 轨迹 | 完整矩阵 46,000 轨迹 |

五种方法为 `baseline`、`random_group`、`domain_randomization`、`fixed_wce`、`online_wce`。所有方法从同一个对应父模型开始，最终累计 2,320,000 控制器训练步。固定和在线 WCE 使用相同预训练 WCE。固定的是 **模型参数**，需求混合权重仍可随状态变化。

先运行验证：

```bash
python main.py experiment verify --workers 4
```

使用输出中的新 `gate.json` 启动论文训练。移动代码或更改输入会使旧准入记录失效。全部论文训练种子为 `101,202,303,404,505`；试运行使用 `9001` 等独立种子。

单个短试运行：

```bash
python main.py experiment parent --network grid --controller ia2c \
  --seed 9001 --pilot --steps 160
```

程序打印实际输出路径和检查点。续训时明确选择检查点，不使用“最新文件”猜测来源。下列尖括号表示必须替换的路径：

```bash
python main.py experiment wce --network grid --controller ia2c \
  --seed 101 --gate <gate.json> --parent <父模型检查点>

python main.py experiment continue --network grid --controller ia2c \
  --seed 101 --method online_wce --gate <gate.json> \
  --parent <父模型检查点> --wce <WCE检查点>
```

恢复时添加 `--resume <同阶段检查点>`，保留原始阶段、方法、种子、总预算和父检查点。正常检查点默认每十个完整回合保存一次；可使用 `--checkpoint-every 1`。文件名后缀是当前阶段的仿真步，不是累计控制器训练步。完整恢复要求模型、优化器、缓冲区、随机流和计数器；旧权重文件不能代替。

## 测试与报告

交通由各路网归一化后的 OD 配置生成，包含重分配、固定混合、方向切换与高峰负载。每个场景生成十份完整 JSON 车辆计划，固定路径、出发时间和速度因子。详见[验证与测试交通生成说明](reviewer_revision_plan_zh.md#traffic-generation)，其中解释两个路网、种子用途、文件位置及期望需求率与抽样车辆数的区别。

```bash
python main.py experiment prepare --materialize --network grid
python main.py experiment prepare --materialize --network monaco

python main.py experiment evaluate --network grid --controller ia2c \
  --seed 101 --parent <最终检查点> --suite all

python main.py experiment report --input <明确的评估或训练目录>
```

`--suite all` 使用十一种已见需求和十二种新场景；`validation` 使用六个独立验证场景。每次评估只运行所选最终控制器，场景依次执行，不自动启动完整训练矩阵。试运行检查点必须加 `--pilot`；可用 `--rollouts 1` 做小规模验证。

报告输出 CSV、PNG、SVG 和 `dashboard.json`。在工作台“结果与图表”导入 JSON。标准差按轨迹和独立训练种子分别计算；五个完整训练种子才生成论文级置信区间。热力图使用统一 `[1200,2400)` 窗口与色标。

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
