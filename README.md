# CB-WCE Standalone：目录说明与运行手册

当前分支：[`updated_worst_case_MAR`](https://github.com/TravidP/updated_worst_case_MARL/tree/updated_worst_case_MAR)。本目录的内容直接作为分支根目录；当前采用本地 CPU 运行。见 [必要文件与目录说明](docs/repository_structure_zh.md)。

[English README](README_en.md) · [中文实验计划](docs/revision_plan_v8_zh.md) · [English plan](docs/revision_plan_v8_en.md) · [实施与验证记录](docs/implementation_status.md)

本项目用于在 **5×5 Grid** 和当前修订入口使用的 **Monaco 子路网**上，比较 MA2C、DQL、PPO 及五种续训方式。源码、必要地图和原始需求从旧项目按白名单提取，本目录具有独立 Git 历史，可以整体迁移，不依赖旧工作区绝对路径。

本 README 说明当前目录实际包含什么、如何运行、结果保存在哪里。实验设计以双语计划为准；实际完成情况以验证记录为准。原文件来源见 [extraction.json](provenance/extraction.json)，许可证见 [LICENSE](LICENSE)。

## 1. 先确认版本和完成状态

| 项目 | 当前情况 |
|---|---|
| 运行代码和 manifest 协议 | v8.1，见 `config/protocol.json` |
| 最新实验计划 | 中文 v8.3（本地多核）；英文仍为 v8.2，待同步 |
| 当前 controller validation | 每场景 3600 秒，常规 3 个固定实现，确认合计 10 个 |
| v8.2 controller validation | 11 个预生成 DR 混合组，暂定每组 600 秒，连续 6600 秒，3 个完整实现；**尚未同步到代码** |
| 目标时长 | 当前中文计划已明确每组 600 秒、总计 6600 秒；英文计划待同步 |
| 离线 WCE validation | 完整 6600 秒，常规 10 个种子组合，确认合计 30 个 |
| 最终测试 | 暂定 11 个 seen + 12 个泛化场景，每场景 10 次、每次 3600 秒 |
| 已有证据 | 11 项单元测试、六组地图／算法短流程、30 个续训分支，以及恢复、审计、迁移检查 |
| 尚未完成 | 正式验证 manifest、SIF 构建及容器内验收、Hábrók 实测、正式训练与正式模型归档 |

**不要把 6600 秒 manifest 直接交给现有 v8.1 校验器，也不要把旧 validation 分数接入新协议的 best/patience。** 新协议需同步代码、版本、缓存身份和测试，清单见计划第 8 节。

历史证据在 [provenance/verification](provenance/verification)。训练流程检查使用 110 秒短回合及小 replay；部分评估检查使用完整 3600 秒。它们不能证明收敛或超算内存足够。过去的临时试验 checkpoint 已清理，复测需重新运行脚本。

## 2. 推荐阅读和执行顺序

1. 阅读第 1 节和双语计划，明确当前实现与目标协议的差异。
2. 查看目录、路径与配置；先在开发环境做检查和短流程。
3. 熟悉 best、恢复点、运行状态和结果导出，再进行阶段分叉。
4. 本地正式实验前，先完成新协议、冻结 manifest、本地环境验收与资源测量。
5. 本地分阶段运行，归档结果并校验独立备份；本阶段不部署 Hábrók。

## 3. 文件夹结构与职责

```text
cbwce_standalone/
├── README.md / README_en.md      # 中英文操作手册
├── main.py                      # 统一命令行入口
├── tf_compat.py                 # TensorFlow compat.v1 兼容入口
├── LICENSE
├── agents/                      # 控制器、WCE 与网络层
├── envs/                        # SUMO 环境、观测、相位和统计接口
│   └── grid_builder.py          # 已归入 envs 的 Grid 辅助依赖
├── experiments/                 # 需求、验证、选模、恢复和归档
├── config/
│   ├── protocol.json            # 当前预算、时长、种子、停止协议
│   └── revised/                 # 两地图的控制器和 WCE INI
├── assets/
│   ├── grid/                    # network.net.xml、detectors.add.xml、demand/
│   └── monaco/                  # 同样结构，使用精简 Monaco 输入
├── tests/                       # 单元与状态一致性检查
├── scripts/                     # 验收、评估、导出与发布
│   └── slurm/                   # 任务矩阵、资源模板、容器启动
├── packaging/                   # 环境锁、容器定义、发布白名单
├── provenance/                  # 来源和已有验证证据
└── docs/                        # 双语计划与实施记录
```

### 3.1 控制器与环境

| 文件 | 职责 |
|---|---|
| [agents/controller.py](agents/controller.py) | MA2C/PPO rollout、有效样本 loss、advantage normalization、训练状态；将 `dql` 转交 DQL 实现 |
| [agents/dql.py](agents/dql.py) | 每路口独立 128/64 MLP、online/target、Huber、epsilon、定长数组 replay 和分块恢复 |
| [agents/recurrent.py](agents/recurrent.py) | MA2C/PPO 紧凑 recurrent 图及 fingerprint 编码 |
| [agents/wce.py](agents/wce.py) | Grid CNN-A2C／Monaco GCN-A2C，输出需求混合权重并训练或冻结推理 |
| [agents/policies.py](agents/policies.py)、[agents/utils.py](agents/utils.py) | 复用的策略、网络层、初始化及辅助实现 |
| [envs/experiment_env.py](envs/experiment_env.py) | 新实验环境入口：TraCI、交通注入、每秒测量、流式指标和审计 |
| [envs/env.py](envs/env.py) | 基础交通仿真、相位、观测和节点定义 |
| [envs/large_grid_env.py](envs/large_grid_env.py)、[envs/real_net_env.py](envs/real_net_env.py) | 地图拓扑与观测接口 |

CLI 仅开放 `ma2c`、`dql`、`ppo`；辅助文件保留旧类不代表支持 `ia2c`／`iqll` 实验。MA2C/PPO 含 LSTM 64，actor、critic 分开；DQL 无 LSTM。网络结构和参数公平性说明见计划第 3 节。

### 3.2 实验编排

| 文件 | 职责 |
|---|---|
| [experiments/runner.py](experiments/runner.py) | 三阶段主循环、预检、验证、分段和恢复 |
| [experiments/protocol.py](experiments/protocol.py) | 协议读取、路径变量、配置定位和可迁移身份 |
| [experiments/configuration.py](experiments/configuration.py) | INI 必填项、类型及合法值检查 |
| [experiments/demand.py](experiments/demand.py) | 原需求读取／归一化、混合、车辆生成和来源哈希 |
| [experiments/scenarios.py](experiments/scenarios.py) | seen／泛化测试定义与需求变换；正式 validation 另由 manifest 提供 |
| [experiments/validation.py](experiments/validation.py) | Manifest 校验、交通缓存、固定随机流、独立推理进程 |
| [experiments/selection.py](experiments/selection.py) | Best、实质改善锚点、patience、平台和共同预算 |
| [experiments/checkpoint.py](experiments/checkpoint.py) | 完整／轻量状态、哈希、原子发布、引用保护与清理 |
| [experiments/metrics.py](experiments/metrics.py) | 在线队列统计、分钟曲线、窗口及学习诊断汇总 |
| [experiments/storage.py](experiments/storage.py)、[experiments/core.py](experiments/core.py) | JSON/JSONL、随机流、哈希与队列奖励等工具 |
| [experiments/runtime.py](experiments/runtime.py) | 软件身份与正式运行环境锁检查 |
| [experiments/tuning.py](experiments/tuning.py) | 生成 24 个开发参数候选，不自动训练或选胜者 |
| [experiments/release.py](experiments/release.py) | 文件清单、发布包、复制校验与归档 |

### 3.3 脚本、测试与证据

| 文件 | 用途 |
|---|---|
| [scripts/verify.py](scripts/verify.py) | 六案例 SUMO 短流程：父模型、WCE、五分支和独立进程恢复 |
| [scripts/verify_audit.py](scripts/verify_audit.py) | 对已有验证目录重跑审计父模型，比较训练张量及队列 |
| [scripts/verify_crash.py](scripts/verify_crash.py) | Grid DQL 试验进程组 SIGKILL、重复提交拒绝和精确恢复 |
| [scripts/measure_metrics.py](scripts/measure_metrics.py) | 合成 6600 秒统计缓冲测量，不代表总作业 RSS |
| [scripts/evaluate_suite.py](scripts/evaluate_suite.py) | 选定 controller endpoint 的测试、缓存及规定审计 |
| [scripts/report.py](scripts/report.py) | 正式评估配对检查、三训练种子统计与 CSV |
| [scripts/export_history.py](scripts/export_history.py) | 排除恢复后被丢弃分支，导出已持久化完整回合 |
| [scripts/package_code.py](scripts/package_code.py) | 白名单代码发布，排除 Git、镜像和缓存 |
| [scripts/slurm/prepare_jobs.py](scripts/slurm/prepare_jobs.py) | 检查前置条件并生成待提交任务，不自动提交 |
| [scripts/slurm/train.sbatch](scripts/slurm/train.sbatch) | CPU 作业数组资源模板与信号转发 |
| [scripts/slurm/launch.py](scripts/slurm/launch.py) | 校验包、镜像与 gate，解压到 `$TMPDIR` 并启动容器 |
| [tests/test_learning.py](tests/test_learning.py) | Replay、target、恢复后的下一次更新、有效样本及推理图 |
| [tests/test_protocol.py](tests/test_protocol.py) | 平台、指标、奖励、manifest、发布和归档 |
| [tests/test_storage.py](tests/test_storage.py) | 引用清理与失败写入保护 |
| [provenance/extraction.json](provenance/extraction.json) | 原提交、提取时工作区状态、源文件与 SHA-256 |
| [provenance/verification](provenance/verification) | 本地 gate、单元日志、资源及审计／恢复／迁移摘要 |

## 4. 数据存放与路径设置

```text
workspace/
├── cbwce_standalone/   # Git：源码、必要静态 assets、配置和文档
├── cbwce_data/         # 冻结 manifest 与共享生成交通
├── cbwce_runs/         # 模型、摘要和验证历史
├── cbwce_cache/        # 可重新生成的缓存
└── cbwce_archive/      # 经校验的发布包及保留成果
```

上述数据目录按需生成，并非都已有内容。

| 变量 | 默认位置 | 用途 |
|---|---|---|
| `CBWCE_ASSETS_ROOT` | 项目 `assets/` | 网络、检测器和原需求 |
| `CBWCE_DATA_ROOT` | 项目同级 `cbwce_data/` | 按地图共享的不可变交通 |
| `CBWCE_RUN_ROOT` | 项目同级 `cbwce_runs/` | 默认运行根目录 |
| `CBWCE_CACHE_ROOT` | 项目同级 `cbwce_cache/` | 最终评估等可重建缓存 |
| `TMPDIR` | 系统临时目录 | SUMO 和验证 worker 临时文件；超算使用作业临时盘 |

显式 `--output` 优先；恢复必须指定原目录。科学身份由内容、配置、协议决定，不依赖旧机器路径。输入不得依赖项目外符号链接。在项目根目录可执行：

```bash
export CBWCE_PROJECT_ROOT="$PWD"
export CBWCE_ASSETS_ROOT="$CBWCE_PROJECT_ROOT/assets"
export CBWCE_DATA_ROOT="$(dirname "$CBWCE_PROJECT_ROOT")/cbwce_data"
export CBWCE_RUN_ROOT="$(dirname "$CBWCE_PROJECT_ROOT")/cbwce_runs"
export CBWCE_CACHE_ROOT="$(dirname "$CBWCE_PROJECT_ROOT")/cbwce_cache"
mkdir -p "$CBWCE_DATA_ROOT" "$CBWCE_RUN_ROOT" "$CBWCE_CACHE_ROOT"
```

## 5. 输入、参数与训练阶段

每地图包含 `network.net.xml`、`detectors.add.xml` 和十一份需求 CSV：四个直向、四个对角方向、`Periphery_to_Center`、`Center_to_Periphery`、`Uniform`。`demand.py` 保留源哈希、原总量和归一化尺度。混合权重必须对应 profile 名称和顺序，不能随意重排。

| 配置 | 对应内容 |
|---|---|
| `config/revised/config_{ma2c,dql,ppo}_large.ini` | Grid 三控制器 |
| `config/revised/config_{ma2c,dql,ppo}_real.ini` | Monaco 三控制器 |
| `config/revised/config_wce_large.ini` / `config_wce_real.ini` | 两地图 WCE |
| `config/protocol.json` | 时长、种子、预算、验证次数和停止阈值 |

正式训练默认 6600 秒／1320 个联合控制步，控制／需求间隔为 5／600 秒。MA2C/PPO rollout=120，PPO 四轮优化；DQL replay batch=64、warm-up=5000 步、每四步更新、容量 50,000 联合转移。不同算法的这些数值含义不同，无须强行一致。

```text
parent：初始训练 → best 父控制器
  ├── baseline
  ├── random_group
  ├── domain_randomization
  └── wce：冻结父控制器 → best 离线 WCE
        ├── fixed_wce：控制器续训，WCE 冻结
        └── online_wce：控制器和 WCE 同时续训
```

五种续训共享同一父模型，后两种共享同一 WCE。Baseline 使用原需求顺序，random group 抽取单一组，DR 使用 Dirichlet 混合。正式分叉要求上游 `plateau_confirmed`，并核对地图、算法、训练种子及 WCE 的父模型归属。

## 6. 环境准备与基础检查

主要目标为 Linux CPU，基线是 Python 3.10、TensorFlow CPU 2.15.1（compat.v1）、NumPy 1.26.4、SUMO／TraCI／sumolib 1.26.0。直接依赖见 [requirements.in](packaging/requirements.in)，完整版本与下载哈希见 [requirements.lock](packaging/requirements.lock)。

开发机可建立隔离环境；以下会安装依赖：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r packaging/requirements.lock
python -m pip check
sumo --version

export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
export MPLBACKEND=Agg

python main.py --help
python main.py check --network grid
python main.py check --network monaco
python -m unittest discover -s tests -v
```

SUMO 系统库安装见 [Apptainer.def](packaging/Apptainer.def)。仅在使用系统 SUMO tools 而没有对应 Python 包时额外设置 `PYTHONPATH`，不要加入旧项目。TraCI 需要本机端口通信。`check` 只检查配置与需求，不等于 SUMO 集成检查；本地依赖安装成功也不等于容器验收。

## 7. 当前代码的可执行短流程

以下为 **v8.1 功能试验**，不是新 v8.2 validation 或正式训练。请在同一 Bash 会话、项目根目录执行；新输出目录不能预先存在。

```bash
CBWCE_DEMO_ROOT="$(mktemp -d /tmp/cbwce-demo.XXXXXX)"
python main.py pilot-manifest --network grid \
  --output "$CBWCE_DEMO_ROOT/validation.json"
pilot_args=(--network grid --controller dql --seed 9001 --pilot
  --validation "$CBWCE_DEMO_ROOT/validation.json"
  --max-episodes 1 --validate-every 1 --validation-repeats 1
  --pilot-block-seconds 10)

python main.py parent "${pilot_args[@]}" --output "$CBWCE_DEMO_ROOT/parent"
CBWCE_PARENT=$(python - "$CBWCE_DEMO_ROOT/parent" <<'PY'
import sys
from experiments.checkpoint import Checkpoints
print(Checkpoints(sys.argv[1]).role('best'))
PY
)
python main.py wce "${pilot_args[@]}" --parent "$CBWCE_PARENT" \
  --output "$CBWCE_DEMO_ROOT/wce"
CBWCE_WCE=$(python - "$CBWCE_DEMO_ROOT/wce" <<'PY'
import sys
from experiments.checkpoint import Checkpoints
print(Checkpoints(sys.argv[1]).role('best'))
PY
)
for method in baseline random_group domain_randomization fixed_wce online_wce; do
  wce_args=()
  if [[ "$method" == fixed_wce || "$method" == online_wce ]]; then
    wce_args=(--wce "$CBWCE_WCE")
  fi
  python main.py continue "${pilot_args[@]}" --method "$method" \
    --parent "$CBWCE_PARENT" "${wce_args[@]}" \
    --output "$CBWCE_DEMO_ROOT/$method"
done
```

这里 11×10=110 秒、22 个控制步，低于默认 DQL 5000 步 warm-up，主要检查管线。完整 verifier 会临时缩小 replay/warm-up，实际覆盖梯度更新和恢复。短预算出现 `no_learning` 或 `max_budget_unconfirmed` 可以是正常结果。

以下会启动 SUMO 并生成模型，明显比单元测试耗时和占空间：

```bash
python scripts/verify.py --output "$CBWCE_RUN_ROOT/verification-v81"
# 单案例只产生 partial gate，不能替代完整六案例 gate。
python scripts/verify.py --case grid_dql \
  --output "$CBWCE_RUN_ROOT/verification-grid-dql"
python scripts/verify_audit.py \
  --verification "$CBWCE_RUN_ROOT/verification-v81" --case grid_dql
python scripts/verify_crash.py --verification "$CBWCE_RUN_ROOT/verification-v81"
python scripts/measure_metrics.py --output "$CBWCE_RUN_ROOT/metrics-memory.json"
```

审计／强制中断脚本依赖 verifier 创建的模型目录仍存在，不能只传入历史证据摘要。

## 8. Manifest、正式训练和恢复

### 8.1 当前 manifest

这是现有 v8.1 支持的未冻结开发样例，**不是新 DR 验证集**：

```json
{
  "version": "8.1", "network": "grid", "split": "validation", "frozen": false,
  "scenarios": [{
    "id": "development_example", "horizon": 3600, "generation_seed": 9001,
    "blocks": [{"start": 0, "duration": 3600, "profile": "Uniform"}]
  }]
}
```

Blocks 支持 profile 或十一项归一化 `weights`，以及 `multiplier`／`sigma`。当前必须连续覆盖 3600 秒、ID 唯一、`frozen` 为 JSON 布尔值。正式训练要求冻结的 `split=validation`；最终测试要求 `split=test`。不要仅将 Uniform 样例改为 `true` 就当作正式设计。

v8.2 需冻结十一组 DR 权重、顺序与三个随机实现，组间保留交通和循环状态。每组时长确认、程序同步后再生成新协议 manifest。离线 WCE 继续验证其自身需求策略，不能用固定 DR 序列替代它的动作。

### 8.2 参数、门禁与恢复

| 参数 | 含义 |
|---|---|
| `--network` / `--controller` | `grid`、`monaco`；`ma2c`、`dql`、`ppo` |
| `--seed` | 开发默认 9001；正式显式使用 101、202、303 |
| `--validation` | 必须提供的 manifest |
| `--output` | 新运行目录；省略时在运行根目录生成唯一目录 |
| `--parent` / `--wce` | 完整 best checkpoint 目录，不是单个 `.index` 文件 |
| `--method` | 仅续训使用，五种方法之一 |
| `--gate` | 正式运行的验证 gate |
| `--resume` | 在原目录、原科学身份下恢复 |
| `--segment-seconds` | 默认 82,800 秒／23 小时，为 Slurm 24 小时留余量 |
| `--audit` | 详细审计；常规训练建议关闭 |
| `--pilot` | 开发模式，允许覆盖预算等参数，不能用正式种子 |

`--config`、`--max-episodes`、`--validate-every`、`--validation-repeats`、`--pilot-block-seconds` 仅用于开发覆盖，正式模式拒绝。正式入口还校验软件锁和源码 fingerprint；进程退出成功不代表达到平台。

```bash
python main.py parent --help
python main.py wce --help
python main.py continue --help
python main.py check --network grid --publication \
  --validation "$CBWCE_DATA_ROOT/manifests/grid-validation.json"
```

新协议、manifest、镜像和 gate 准备完成后，正式命令结构为：

```bash
python main.py parent --network grid --controller ma2c --seed 101 \
  --validation "$CBWCE_DATA_ROOT/manifests/grid-validation.json" \
  --gate "$CBWCE_DATA_ROOT/verification/gate.json" \
  --output "$CBWCE_RUN_ROOT/grid/ma2c/seed_101/parent"
```

恢复时重复原完整命令并加 `--resume`；保留原 seed、manifest、配置和父 checkpoint。预算、RNG、patience、optimizer、replay 均恢复；DQL 续训 epsilon 只在初次分叉重置。`interrupted` 应继续同阶段，不直接启动下游。

```bash
python main.py tuning-configs --output "$CBWCE_DATA_ROOT/tuning-candidates"
```

这只生成 24 份 INI 和 `candidates.json`。胜出配置需要纳入对应版本的 `config/revised/` 并重新验收，因为正式入口不接受 `--config`。

## 9. 运行目录与 checkpoint

```text
one_run/
├── manifest.json          # 协议、源码、输入、祖先与运行身份
├── result.json            # 应用状态和资源摘要
├── references.json        # best/latest/final/common/window/pins
├── attempts.jsonl         # 重启与恢复起点
├── episodes.jsonl.gz      # 完整回合摘要及 attempt 身份
├── blocks.jsonl.gz        # 权重、交通哈希、区间诊断
├── validation.jsonl.gz    # 常规和确认验证
├── checkpoints/e.../
│   ├── manifest.json      # 类型、签名、父引用、文件哈希
│   ├── runner.pkl         # 运行与选模状态
│   ├── controller/        # 参数；完整状态另含 state.pkl、DQL replay/
│   └── wce/               # 需要持久化 WCE 时才出现
├── audit/                 # 可选 NPZ
└── scheduler.json         # Slurm launcher 写出的作业回执
```

不同阶段不一定有所有文件或两个模型目录；冻结父模型／固定 WCE 通过引用共享。

| 引用或状态 | 含义 |
|---|---|
| `best` | 常规验证最佳的完整状态，用于阶段分叉 |
| `best_inference` | 对应轻量推理快照，不能恢复训练 |
| `latest` | 最近有效完整恢复点，由 `--resume` 读取 |
| `final` | 阶段终止时状态，不保证比 best 好 |
| `best_at_common_budget` | 续训 1000 回合时冻结的最佳推理终点 |
| `plateau_confirmed` | 经验平台已确认，不是全局最优证明 |
| `max_budget_unconfirmed` | 达到最大预算，未确认平台 |
| `no_learning` | 父模型／离线 WCE 到上限仍无规定的实质改善 |
| `interrupted` | 未完成，需恢复 |
| `failed` | 发生错误，需先查看原因 |

每十个完整回合保存恢复点，保留最新两个；保留完整 best、最近五验证轻量候选、共同预算终点、最终状态和下游 pin。旧恢复点有依赖时会保留额外对象，不能按修改时间直接删除。

完整状态含 optimizer、RNG、replay／pending、计数、选模及同时刻在线 WCE。快照仅用于推理。仅加载可信 pickle 状态，不手工编辑 checkpoint、哈希或引用索引。

## 10. 评估、报告和训练历史

测试 manifest 生成 23 个场景，但不生成交通或启动仿真：

```bash
python main.py test-manifest --network grid \
  --output "$CBWCE_DATA_ROOT/manifests/grid-test.json"
python main.py test-manifest --network monaco \
  --output "$CBWCE_DATA_ROOT/manifests/monaco-test.json"
```

对已经终止的 controller run 评估；不要将离线 WCE run 当作 controller run。正式评估要求正式模型、冻结 test manifest、匹配环境和每场景十次。开发模型需加 `--pilot`，不能进入正式报告。

```bash
python scripts/evaluate_suite.py \
  --run "$CBWCE_RUN_ROOT/grid/ma2c/seed_101/online_wce" \
  --manifest "$CBWCE_DATA_ROOT/manifests/grid-test.json" \
  --selection best --output "$CBWCE_RUN_ROOT/evaluation/grid-ma2c-101-best"
# 若对应引用已生成，可将 selection 改为 best_at_common_budget，并使用新输出目录。

python scripts/export_history.py \
  --run "$CBWCE_RUN_ROOT/grid/ma2c/seed_101/online_wce" \
  --output "$CBWCE_RUN_ROOT/exports/grid-ma2c-101-episodes.csv"

# 三个输入目录必须已经分别完成评估。
python scripts/report.py \
  --input "$CBWCE_RUN_ROOT/evaluation/grid-ma2c-101-best" \
  --input "$CBWCE_RUN_ROOT/evaluation/grid-ma2c-202-best" \
  --input "$CBWCE_RUN_ROOT/evaluation/grid-ma2c-303-best" \
  --output "$CBWCE_RUN_ROOT/reports/grid-ma2c-online-best"
```

评估目录含 manifest、result、`rollouts.jsonl.gz` 与规定 audit。缓存按 checkpoint、测试、源码、runtime 身份复用。Uniform／peak_1.5 第一条实现额外审计，常规保留标量、分钟曲线和车道窗口。

Report 校验每份 23×10 rollout、哈希、交通/SUMO 配对及每单元的三训练种子，输出 `seeds.csv`、`comparison.csv`；具备 online 和 comparator 时另写 `paired_differences.csv`。标准差跨训练种子计算。训练曲线使用 `export_history.py`，避免恢复前后的重复记录被重复计数。

## 11. 控制内存与文件增长

| 数据 | 默认处理 |
|---|---|
| 每秒队列 | 逐秒测量并在线累加，不保存整回合 Python 列表 |
| 当前状态 | 当前车道向量、最多 600 秒全网队列环形缓冲 |
| 学习诊断 | 按需求区间汇总 |
| 训练交通 | 按区间生成，保留权重和哈希，不导出全部车辆列表 |
| 固定评估交通 | 内容寻址、压缩共享、引用 route table |
| DQL replay | 定长数组、联合转移容量、分块保存 |
| 验证模型 | 独立推理图，不加载 replay／optimizer |
| 详细审计 | 显式启用，不为所有正式回合无条件开启 |
| 图表与 CSV | 从规范结果离线派生 |

统计缓冲测量不等于总内存。历史 trainer `ru_maxrss` 也不包含并行 SUMO／验证进程总量；应在目标镜像、replay 填满时测量总作业峰值，正式配额留至少 1.3 倍余量。

## 12. 可选历史配套：Apptainer 与 Hábrók（当前不部署）

### 12.1 构建与容器验收

当前没有已构建 SIF 或生成的 `packaging/runtime.lock.json`。在具备 Apptainer 的开发机，用实际 Python 3.10 镜像及真实 digest 替换占位值：

```bash
export CBWCE_BASE_IMAGE='python:3.10-slim-bookworm@sha256:REPLACE_WITH_REAL_DIGEST'
bash packaging/build.sh
```

该占位命令不能直接完成构建。脚本使用已有哈希依赖锁，生成 `cbwce-v8.1.sif`、`runtime.actual.json` 和 `runtime.lock.json`（后两者在 packaging/）。只有主动刷新依赖才设 `CBWCE_UPDATE_LOCK=1` 并提供 pip-tools。新协议实施时需同步这些 v8.1 命名和标识。

```bash
apptainer exec --cleanenv \
  --bind "$PWD,$CBWCE_DATA_ROOT,$CBWCE_RUN_ROOT,$CBWCE_CACHE_ROOT" \
  cbwce-v8.1.sif python scripts/verify.py \
  --output "$CBWCE_RUN_ROOT/container-verification-v81"
```

Verifier 将共享输入放入其输出目录。`--cleanenv` 下自定义环境变量需显式传入，不能假设继承宿主 `CBWCE_*`。正式集群任务需要同镜像完整六案例 gate；历史本地 gate 不能代替容器验收。

### 12.2 存储与资源

| 位置 | 用途 |
|---|---|
| 本地 Git | 代码、配置、文档及版本历史 |
| `/scratch/<account>/cbwce/` | 包、SIF、manifest、活跃运行和持久 checkpoint |
| `$TMPDIR` | 解压代码与 SUMO 临时文件 |
| `/projects/<account>/cbwce/` | 登录端校验归档 |
| RDMS／独立备份 | 长期保存重要成果 |

Scratch 无备份，projects 不直接供计算节点使用；从登录端归档，用 `hbquota` 检查实际配额。官方资料：[存储](https://wiki.hpc.rug.nl/habrok/migration/migration_habrok)、[配额](https://wiki.hpc.rug.nl/habrok/data_management/quota)、[Apptainer](https://wiki.hpc.rug.nl/habrok/examples/apptainer)、[Slurm](https://wiki.hpc.rug.nl/habrok/job_management/scheduling_system)。

模板起点为 regular、1 task、2 CPU、8 GiB、24 小时、数组并发 4，结束前 600 秒发信号。8 GiB 不是已证实足够的正式资源配置。

### 12.3 生成与提交矩阵

先准备新协议源码、SIF、匹配 gate、冻结 manifest 和代码包。以下仅示例命令结构；替换 ACCOUNT、版本和实际文件路径。Gate／validation 放在绑定的 data root 内。

```bash
CBWCE_HPC_ROOT=/scratch/ACCOUNT/cbwce
python scripts/slurm/prepare_jobs.py --stage parent \
  --image "$CBWCE_HPC_ROOT/cbwce-v8.1.sif" \
  --code-release "$CBWCE_HPC_ROOT/releases/code-release.tar.gz" \
  --data-root "$CBWCE_HPC_ROOT/data" --run-root "$CBWCE_HPC_ROOT/runs" \
  --cache-root "$CBWCE_HPC_ROOT/cache" \
  --validation-grid "$CBWCE_HPC_ROOT/data/manifests/grid-validation.json" \
  --validation-monaco "$CBWCE_HPC_ROOT/data/manifests/monaco-validation.json" \
  --gate "$CBWCE_HPC_ROOT/data/verification/gate.json" \
  --output "$CBWCE_HPC_ROOT/parents.json"
# 仅当生成器确认 18 项任务时使用该范围。
sbatch --array=0-17%4 scripts/slurm/train.sbatch "$CBWCE_HPC_ROOT/parents.json"
```

在项目根目录提交，模板相对路径需要 `scripts/slurm/`。代码包与 `.tar.gz.manifest.json` 一起上传。矩阵生成环境需要项目依赖，下游检查还需要 `sacct`。

生成器不自动提交或自动排后续阶段。完整矩阵为 18 parent、18 WCE、90 continuation；实际任务数可能因已有完成项减少。父模型通过平台且 Slurm `COMPLETED/0:0` 后生成新的 `--stage wce` JSON，WCE 同样通过再生成 `--stage continue`。Interrupted 项会带 `--resume`；failed／活跃目录需先检查。

Launcher 校验包、镜像与 gate，解压到作业临时盘，转发信号；训练进程周期性直接写入持久 run root，不能只靠退出时复制。

## 13. 打包、归档与独立保存

代码使用 [release_allowlist.json](packaging/release_allowlist.json)。包输出必须在源目录之外且使用新文件名。

```bash
CBWCE_RELEASE_ROOT="$(dirname "$PWD")/cbwce_archive/releases"
mkdir -p "$CBWCE_RELEASE_ROOT"
python scripts/package_code.py \
  --output "$CBWCE_RELEASE_ROOT/cbwce-code-docs-update.tar.gz"
python main.py package --kind inputs --source assets \
  --output "$CBWCE_RELEASE_ROOT/cbwce-assets-new.tar.gz"
python main.py package --kind run \
  --source "$CBWCE_RUN_ROOT/grid/ma2c/seed_101/online_wce" \
  --output "$CBWCE_RELEASE_ROOT/grid-ma2c-101-online-run.tar.gz"
```

`--source assets` 只含地图和原需求；需要 manifest／共享交通时另建 input staging。每包附逐文件及整体 SHA-256 清单。代码包排除 Git、SIF、缓存，不能代替 Git 仓库备份；SIF 单独保存。运行包只保存指定目录现状，祖先模型属于外部引用，须一并保留相应对象。

登录端可归档已经停止写入的目录：

```bash
python main.py archive \
  --source /scratch/ACCOUNT/cbwce/runs/completed-run \
  --destination /projects/ACCOUNT/cbwce/completed-run
```

工具先复制到目标临时目录、校验、发布，生成 `archive_inventory.json`，不删除源。归档与独立备份验证后才清理不活跃 scratch；不能删仍被引用的 parent/WCE，不能自动删旧项目历史数据。

同级 `cbwce_archive/releases/*v8.1*` 是初始提交 `a367a66` 的历史快照，**不含后续 v8.2 计划和本次 README**。要获取最新说明，重新打包新文件名，不覆盖历史哈希记录。

## 14. 常见问题与交接

| 情况 | 排查方向 |
|---|---|
| 6600 秒 manifest 被拒绝 | 当前校验仍为 v8.1／3600 秒；完成协议迁移，不伪造旧字段 |
| 未冻结 manifest | 检查设计、split 与 frozen；不要以 Uniform pilot 代替正式验证 |
| Missing/stale gate | 科学源码／配置／输入变化后，在匹配镜像中重新验收 |
| 缺少 runtime.lock.json | 尚未完成实际容器构建与环境记录 |
| Signature/identity mismatch | 核对配置、输入、协议、seed、祖先；不能混用旧状态 |
| Snapshot 不能恢复 | 分叉使用完整 best，恢复让 `--resume` 读取 latest |
| Another process owns this run | 检查已有进程／作业，不并发写同目录 |
| 输出目录存在 | 新任务换新目录，原任务恢复使用 `--resume` |
| 短试验 no_learning | 检查预算和 warm-up；不等同于程序崩溃 |
| 下游不能生成 | 核对 plateau、scheduler 回执与 sacct 成功状态 |
| 无逐秒日志或图 | 默认精简输出；离线导出或有目的地启用 audit |
| 迁移后缺输入 | 检查路径变量、bind 和 manifest；不要重新依赖旧绝对路径 |

修改算法需核对配置和测试；修改验证设计需同步两份计划、协议、manifest、缓存、gate 和 README。科学身份变化不能直接恢复旧 run。只修改 Markdown 不会自动改变运行行为。

交接至少保留：独立 Git、源码包、输入清单／生成器版本、镜像及软件锁、冻结 manifest、选定模型及祖先引用、规范结果与资源记录。代码、研究计划、验收证据、实际训练成果应分别标明版本保存。

## 精简训练分支与存储交接

本地训练工作分支为 `updated_worst_case_MAR`。旧数据云端迁移进度、恢复方法和未完成项见 [存储交接记录](docs/storage_handover_20261116.md)。创建新分支本身不会释放旧工作区磁盘空间。
