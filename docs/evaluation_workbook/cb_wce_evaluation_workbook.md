# CB-WCE 测试与数据验证工作手册

**CB-WCE Evaluation and Data Validation Workbook**

Chinese Part I / English Part II / Shared generated appendices

| Item | Value |
| --- | --- |
| Workbook release | 1.0 |
| Operational protocol | v7 implemented; versioned artifacts pending |
| Training seed | 101 |
| Publication evaluation status | **0 / 9,200 formal rollouts** |
| Snapshot date | 2026-09-29 |

This workbook is independent from `paper/main.tex`. Pilot and historical results are not publication evidence.

## 中文测试与数据验证手册

### 目标、现状与冻结评估协议

#### 本手册的作用

本手册是正式评估、数据验证、统计分析、网站导出和论文呈现的唯一操作说明。它不替代论文，也不把尚未运行的结果写成事实。当前 40 个最终续训组合均存在完整 checkpoint，但正式 publication evaluation 仍为 **0/9,200**。`runs_eval/revised/peak_validation` 仅为 pilot，旧论文中的 baseline/retrained 数字也属于另一套协议，二者均不得进入新版主结果。

本轮核心问题是：在两个路网、四个控制器和五种等预算续训方法上，online WCE 是否能在已见需求和冻结的未见需求套件中降低网络总排队，同时保持速度、通行完成率、可靠性和计算成本处于可解释范围。

#### 模型矩阵与公平性边界

五种方法为 `baseline`、`random_group`、`domain_randomization`、`fixed_wce` 和 `online_wce`。四种控制器为 IA2C、MA2C、IQLL 和 PPO；路网为 Grid 与 Monaco。所有最终控制器的累计学习步数必须为 2,320,000，其中 parent 为 1,000,000，continuation 为 1,320,000。

用户决定保留当前 baseline。审计发现 baseline continuation 使用 Python 3.10.12 / TensorFlow 2.15.1，其他多数 continuation 使用 Python 3.6.13 / TensorFlow 1.12.0，且 `envs/experiment_env.py` 的源码哈希不同。因此：

- 性能结论必须写成“对冻结 checkpoint 与 seed 101 的条件性结果”；
- wall-clock 只能作为描述性数据，不用于严格公平的训练速度排序；
- 不声称 CB-WCE accelerates training，不给出无同环境证据的 time-to-threshold 结论；
- 最终 release 必须包含 runtime/source compatibility matrix。

#### 实验规模

| Item / 项目 | Frozen value / 冻结值 |
| --- | --- |
| Networks | Grid, Monaco |
| Controllers | IA2C, MA2C, IQLL, PPO |
| Methods | baseline, random group, domain randomization, fixed WCE, online WCE |
| Training seed | 101 |
| Final controller steps | 2,320,000 (1,000,000 parent + 1,320,000 continuation) |
| Evaluation horizon | 3,600 s; empty start; no warm-up; no drain |
| SUMO / control step | 1 s / 5 s (2 s yellow + 3 s green) |
| Scenarios | 11 seen + 12 test = 23 |
| Paired seeds | arrival 51001--51010 / SUMO 61001--61010 |
| Formal rollouts | 9,200 (current accepted: 0) |
| Implemented / required protocol | v7 / v7 |

正式矩阵为

$$
2\;\text{networks}\times4\;\text{controllers}\times5\;\text{methods}\times23\;\text{scenarios}\times10\;\text{pairs}=9{,}200.
$$

每个最终模型运行 230 条 rollout，每个网络 4,600 条。总仿真时间为 9,200 小时，即 33.12 million simulated seconds；每条 rollout 720 次控制决策，总计 6.624 million decisions。

#### 通用仿真默认值

每次评估从空路网开始，时间区间为 $[0,3600)$ 秒，无 warm-up，也不在 3600 秒后排空。SUMO 每秒采样一次；控制周期 5 秒，由 2 秒 yellow 和 3 秒 green 组成。Grid 基准需求为 3,000 veh/h，Monaco 为 2,383.3333 veh/h。训练 episode 为 6,600 秒，WCE 每 600 秒接收一次成本，共 11 个等长窗口。

Arrival seeds 固定为 51001--51010，SUMO seeds 固定为 61001--61010，并按 rollout index 一一配对。相同 network/scenario/index 的 demand artifact、SUMO seed 和场景定义必须跨所有 controller/method 复用；policy seed 使用协议规定的确定性摘要生成。

#### 11 个 seen 场景与 12 个 test 场景

Seen split 是 11 个归一化训练 profile，每个 profile 在完整 3,600 秒内保持不变。Test split 包含四个场景族，每族三个场景：

| 场景族 | 场景 | 精确定义 |
| --- | --- | --- |
| OD redistribution | $\sigma=0.25,0.50,0.75$ | 以 Uniform 的正 OD 为 support，乘独立 LogNormal$(0,\sigma)$ 因子后重新归一化到基准总率。零 OD 保持为零。 |
| Demand mixture | mixture 1, 2, 3 | 由固定 generation seed 产生的 Dirichlet$(1,\ldots,1)$ 权重，对 11 个归一化 profile 作凸组合；整小时固定，总率不变。 |
| Temporal switching | 300, 900, 1200 s | 每个区块从完整的 11 个 seen profiles 中按冻结 generation seed 随机选择下一个 profile，并禁止相邻区块重复。300 秒场景用于快速切换压力测试；900 和 1200 秒场景用于检验更长驻留时间。 |
| Peak demand | $1.10,1.25,1.50$ | $[0,1200)$ 基准，$[1200,2400)$ 乘 peak multiplier，$[2400,3600)$ 恢复基准。 |

当前代码已实现 protocol v7：三个 temporal 场景均从完整的 11 个 seen profiles 中按冻结 generation seed 随机选择，并禁止相邻区块重复。v7 artifacts 尚待物化；正式运行前必须在独立的版本化目录中生成并审计它们，旧 v6 artifacts 不得进入 v7 release。

Mixture 是 held-out composition/interpolation，不称为严格 OOD；peak 是强度外推；redistribution 是固定 OD support 内重分配；switch 是 11 个已见空间 profile 的冻结随机时间重排。Validation 六场景只用于调参，test 冻结后只评估一次。

Peak 的期望一小时车辆量为：Grid 3,100 / 3,250 / 3,500；Monaco 约 2,462.78 / 2,581.94 / 2,780.56。最终实际 mean/min/max 必须从 v7 冻结 artifact 自动计算，不能从旧表手工抄写。

#### 需求物化规则

每个 block 对每个 OD 采样 $N\sim\mathrm{Poisson}(r\,d/3600)$；departure time 在 block 内均匀采样并加 $\mathcal{N}(0,2\,\mathrm{s})$ jitter，再裁剪到 block 边界内并保留两位小数。Speed factor 采样自 $\mathcal{N}(1,0.1)$，非正值重新采样。路线由当前 SUMO network 解析并冻结在 artifact 中。Artifact 必须保存 network、profiles、scenario 和自身 hash。

### 指标、奖励一致性与统计分析

#### 统一的底层交通量

在全局去重的受控进口车道集合 $L_{\mathrm{controlled}}$ 上定义

$$
Q(t)=\sum_{l\in L_{\mathrm{controlled}}}q_l(t),\qquad
J_Q=\frac{1}{3600}\sum_{t=1}^{3600}Q(t).
$$

Grid 有 150 条、Monaco 有 116 条受控进口车道。$J_Q$ 的单位为 vehicles，是 **network-total mean queue**，*lower is better*；不得写成 per-intersection queue。

Controller 在每个 5 秒控制端点接收 raw negative queue reward。除 MA2C 使用本地加 $0.9$ 邻居加权外，IA2C、IQLL、PPO 的 raw reward 是 $-Q$ 的网络复制。Learner 实际使用

$$
r_{\mathrm{ctrl}}=\operatorname{clip}(r_{\mathrm{raw}}/d_c,-2,2),
$$

其中 $d_c$ 必须从每个最终 manifest 的 `reward_norm` 读取。

对任意区间定义

$$
C_{[a,b)}=\frac{1}{b-a}\sum_{t=a+1}^{b}Q(t).
$$

CB-WCE 每 600 秒最大化 $C_k=C_{[600k,600(k+1))}$，learner 使用

$$
r_{\mathrm{WCE}}=\operatorname{clip}(C_k/d_w,-2,2),
\quad d_w=3000\;(\mathrm{Grid}),\;1000\;(\mathrm{Monaco}).
$$

主评价始终使用未缩放、未裁剪的 $J_Q$。正比例缩放不改变方向，但 clipping 会破坏严格单调等价。必须从选定训练日志计算 controller/WCE clipping fraction；只有它为零时才能写“严格一致”，否则报告实际比例并限制论文表述。

#### Rollout 级指标字典

| 字段 | 单位 | 方向 | 定义与分母 |
| --- | --- | --- | --- |
| mean_total_queue | vehicles | lower | $3600^{-1}\sum_t Q(t)$，主指标。 |
| integrated_queue | vehicle-seconds | lower | $\sum_tQ(t)$。 |
| peak / p95 queue | vehicles | lower | 每秒 network-total queue 的最大值和 95th percentile。 |
| window queue/AUC | vehicles, vehicle-seconds | lower | 三个 1,200 秒窗口分别汇总；用于 peak response。 |
| last_600_s_slope | vehicles/second | lower | 对最后 600 个 $Q(t)$ 样本作预注册线性斜率，仅作稳定性描述。 |
| mean_speed | m/s | higher | $\sum_t\mathrm{speed_sum}_t/\sum_t\mathrm{active}_t$；分母单独保存。 |
| scheduled/inserted/completed | vehicles | context | Artifact 车辆数、实际出发数、到达数。 |
| pending/remaining | vehicles | lower | 3600 秒尚未出发、仍在路网中的车辆。 |
| completion_rate | ratio | higher | completed/inserted；inserted=0 时为 null。 |
| completed trip means | seconds | lower | travel/waiting/time-loss/departure-delay；仅对 completed vehicles，必须显示分母和 unfinished counts。 |
| teleports/collisions | count | lower | 可靠性指标。两个路网 teleport threshold 不同，不跨网直接排名。 |
| wall_seconds | seconds | lower | 单条评估的单调时钟耗时。 |

#### 预注册配对比较

主要比较固定为 online WCE 分别减 baseline、random group、domain randomization、fixed WCE。对场景 $s$ 和 rollout pair $r$：

$$
d_{s,r}=J_{\mathrm{online},s,r}-J_{\mathrm{comparator},s,r}.
$$

Queue 的负差值表示 online WCE 更好。每场景报告 $n=10$ 的 mean、sample SD（`ddof=1`）和确定性 10,000 次 paired percentile bootstrap 95% CI。Bootstrap seed 由固定标签、network、controller、comparator、scope 和 scenario 的摘要生成。

Family 或完整 test suite 先对相同 rollout index 跨预注册场景等权平均：

$$
D_r=|S|^{-1}\sum_{s\in S}d_{s,r},
$$

再对十个 $D_r$ 做 bootstrap。不得把逐秒数据、车道数据或 120 条跨场景 rollout 当作独立 IID 样本。CI 只反映固定 checkpoint、固定测试套件下的仿真/策略波动，不是训练 seed 不确定性；单训练 seed 不能支持算法总体显著性声明。若增加多场景假设检验，使用 Holm correction。

Queue 百分比改善可补充报告为 $(J_{\mathrm{comp}}-J_{\mathrm{online}})/J_{\mathrm{comp}}\times100%$，但绝对差必须优先；接近零分母时不报告百分比。速度使用相反方向，且不以夸张的 $>100%$ 百分比代替绝对 m/s 差。

#### 稳健性与最差场景

Test split 同时报告 12 场景等权平均、各场景均值的最大值、worst-3 average/CVaR 和场景间 SD。不同方法可能有不同 worst scenario，因此不得把一个方法的 worst 场景曲线冒充所有方法的共同 worst。Queue-worst 与 speed-worst 必须分别选择。

#### 训练和评估耗时

从最终 manifest 的 `resume` 递归追踪所有 attempt，汇总 parent、offline WCE、continuation、evaluation 和 abandoned/duplicate work。成功 attempt 使用 `result.json.wall_seconds`；无 result 的前序 attempt 使用最后一条 `progress.jsonl.wall_seconds` 作为 lower bound，并标记 `exact` 或 `lower_bound`。

Standalone cost 对 fixed/online 分别计入一次 offline WCE；deduplicated campaign cost 对共享 parent/WCE 只计一次。Component seconds 不是完备分解，不能替代总 wall time。所有耗时表同时给出 OS、CPU/RAM（可得时）、Python、TensorFlow、线程数、并发数和源码哈希；baseline 环境差异必须作为表注。

### 数据保存协议与发布验证

#### 三层数据结构

1. **Authoritative raw**: `runs_eval/revised/publication_seed101_v1/<network>/<controller>/<method>/`，保存不可变 manifest、environment、summary、result、压缩逐秒记录和 NPZ。
1. **Validated release**: `output_result/revised/publication_seed101_v1/`，保存 release manifest、validation report、CSV、图表、paired bootstrap 和兼容 dashboard JSON。
1. **Sanitized site bundle**: `docs/site/dist/data/publication/`，只保存字段白名单允许的聚合与脱敏 JSON。

JSON 是权威格式，CSV 仅供阅读。JSON 禁止 NaN/Infinity；不适用或缺失值使用 `null`，并在 metric registry 说明原因、单位、方向和分母。Raw 数据不得复制到托管站点。

#### Selection registry 与主键

`runs_eval/revised/selections/final_evaluation_seed101.json` 冻结 8 个 parent、8 个 offline WCE、40 个 final continuation 和 40 个 planned evaluation suite。每项保存 repo-relative path、绝对本机路径、checkpoint step、目录聚合 SHA-256、manifest hash、manifest 文件 SHA-256、runtime/source compatibility 状态和计划输出目录。模型选择只允许使用该 allowlist，不允许按 mtime、“latest”或宽目录扫描推断。

Rollout 唯一键为

```text
network/controller/method/training_seed/split/scenario/
generation_seed/arrival_seed/sumo_seed/policy_seed/attempt
```

公开 rollout ID 使用上述稳定字段的摘要；绝对路径不进入公开 ID。

#### Publication-complete 硬门槛

只有全部规则同时通过，`publication_complete` 才能为 true：

- accepted 恰好 9,200，rejected、missing、duplicate、extra 均为 0；
- 40 个 checkpoint 全部来自 selection allowlist，controller learning steps 为 2,320,000；
- 所有记录 `pilot=false`、`status=complete`、`horizon=sample_count=3600`；
- arrival 51001--51010 与 SUMO 61001--61010 按 index 精确对应；
- 同一 network/scenario/index 的 demand hash 与 SUMO seed 跨 20 个 controller-method 组合一致；
- protocol、scenario、profiles、network、artifact、checkpoint 和 manifest hashes 与冻结输入一致；
- 时间戳严格为 1--3600，queue 非负且 finite；
- $\mathrm{integrated\ queue}\approx3600\times\mathrm{mean\ queue}$，peak 不小于 mean；
- scheduled = inserted + pending，completed $\le$ inserted，completed-trip denominator = completed；
- NPZ 每车道 queue 之和与 JSONL network total 对齐；
- public bundle 不含绝对路径、token、checkpoint payload、raw log 或 pilot/history 数据。

#### 磁盘与压缩门槛

2026-09-29 实测根文件系统约有 69 GiB 可用，已超过 60 GiB 启动阈值，但每次正式启动前都必须重新检查。按现有 pilot 估算，9,200 条 raw rollout 可能超过 33 GB。启动条件是“可用空间至少 60 GiB，且预计完成后保留至少 20 GiB”。若不满足，应先实施 suite-level shared provenance、精简 per-rollout manifest、gzip JSONL 和压缩 NPZ，并用 pilot 重新测量；不得通过删除未知用户数据解决。

#### Reward 与环境验证

发布验证还要计算 selected training runs 的 controller/WCE clipping fraction，生成 source/config/runtime compatibility matrix，并将 baseline 不同运行环境标为已知限制。环境差异不阻断用户选择的 release，但必须令 `strict_runtime_comparability=false`，并禁止训练速度或严格公平性的强结论。

### 本地/静态网站契约与论文可视化

#### 混合读取架构

现有 `docs/site/.openai/hosting.json` 是静态 Sites 项目，应保留其架构与 project ID。本轮不改站点代码，也不发布。托管浏览器不能扫描实验机目录；“自动读取”定义为：本地 dashboard/API 从 selection allowlist 发现和验证 release，随后导出脱敏静态快照；托管站点只 `fetch` 已发布的相对 JSON。

下一阶段接口规格：

```text
GET /api/releases
GET /api/releases/<id>/manifest
GET /api/releases/<id>/overview
GET /api/releases/<id>/rollouts?network=&controller=&method=&split=&scenario=
```

建议公开分片为 `manifest.json`、`overview.json`、8 个 network-controller rollout shards（每个 1,150 条）、24 个 network-controller-peak heatmap shards、可选 curve shards 和 `checksums.json`。旧 `results.js` 是 pilot schema v1 数据，不得作为正式发布源。

首屏必须立即显示 9,200/9,200、rejected=0、protocol hash、release 时间、主要 metric 和 network/controller/method/scenario filters。页面还应提供 paired-effect interval、method-scenario heatmap、time series、flow/reliability、cost、下载 JSON/CSV 和中英文切换。

#### 论文主图与补充材料

1. **Paired-effect forest plot**: online WCE 对四个 comparator 的绝对 queue 差与条件 95% CI，零线和“negative is better”明确标注。
1. **Method $\times$ scenario heatmap**: 12 test 场景，支持 absolute $J_Q$ 与 relative-to-baseline；seen/test 分开。
1. **Robustness frontier**: suite average 对 worst-3 CVaR，避免只展示平均值。
1. **Peak response**: 主文预注册 peak 1.25 与 1.50 的 $Q(t)$ mean 与区间，阴影标注 $[1200,2400)$；1.10 放补充材料。
1. **Network heatmaps**: 五方法 absolute maps 与 online-minus-comparator maps；同 network/metric 使用固定公共绝对色标，差值图用以 0 对称的公共色标。
1. **Training cost**: parent/offline WCE/continuation 堆叠条，同时展示 standalone 和 deduplicated campaign cost，并标 exact/lower-bound。
1. **Flow/reliability**: completed、remaining、pending、teleports、collisions；completed-trip 指标必须带分母。
1. **Mechanism**: WCE mixture weights 与 entropy 放补充材料。

若要满足“network-wide heatmap”，正式评估前必须新增所有 non-internal lanes 在峰前、峰中、峰后三窗口的在线聚合。当前数据仅覆盖受控进口车道；若未扩展，图名必须是“network-spanning controlled-approach heatmap”，不能写全路网全部车道。

#### 论文措辞检查表

- “per intersection” 改为 “network-total mean queue”。
- “unbounded growth” 改为 “sustained growth within the 3,600-s horizon”。
- “algorithm-agnostic confirmed” 改为 “consistent trend across four architectures for training seed 101”。
- 无统一环境和 time-to-threshold 证据时删除 “accelerates training”。
- 绝对差作为主结果，百分比仅补充；queue-worst 与 speed-worst 分开选择。

### 执行工作簿与签字流程

#### Phase 0：正式运行前必须完成

1. 实现 protocol v7：三个 temporal 场景按冻结 seed 从 11 个 seen profiles 中随机切换、相邻区块不得重复，并重新生成/哈希全部 60 个对应 artifacts。
1. 实现或明确降级 network-wide heatmap 采集；若不实现，锁定 controlled-approach 表述。
1. 重新运行环境 gate，验证 deeprlsc、SUMO、TraCI、TensorFlow、两个网络和 11 profiles。
1. 重新检查磁盘阈值与所有 40 个 checkpoint hash；确认输出目录不存在。
1. 独立复核 selection registry 和命令目录的一一对应关系。

#### 现有可用命令

```text
cd /home/sdc_joran/Journal/deeprl_signal_control
export CBWCE_GATE=/absolute/path/to/protocol_v7/gate.json

./docs/evaluation_workbook/generated/publication_workflow.sh --status
./docs/evaluation_workbook/generated/publication_workflow.sh --preflight
./docs/evaluation_workbook/generated/publication_workflow.sh --list

CONFIRM_PUBLICATION=RUN_9200 CBWCE_EVALUATION_WORKERS=4 \
  ./docs/evaluation_workbook/generated/publication_workflow.sh --execute
```

`publication_workflow.sh` 默认只显示计划，不启动仿真。正式执行需要显式确认变量、protocol v7、至少 60 GiB 可用空间、与四 worker 匹配的有效 gate、已物化的 demand artifacts 和不存在的输出根目录。通过后，调度器默认并行运行四个 suites；每个 suite 内的 230 个 rollouts 仍顺序执行。任一 suite 失败后停止派发新任务，不自动重试或覆盖输出。

#### 下一阶段待实现命令

以下接口是规格，不是当前可用功能：

```text
python main.py experiment report --selection <selection.json> --output <release>
python main.py experiment validate-report --report <release> \
  --require-publication-complete
python main.py experiment export-site --report <release> \
  --output docs/site/dist/data/publication
```

#### 故障、恢复与防止重复样本

失败后不得把新的 attempt 当成额外随机重复。恢复必须沿 manifest 的 `resume` 链进行，最终报告仅接受 selection 指定的成功 attempt；失败、interrupted 和重复 attempt 单独进入成本审计。需求 artifact、seeds、checkpoint 或 scenario hash 任一变化都必须创建新 protocol/release ID，禁止覆盖历史目录。

#### 审稿意见映射

| 编号 | 要求 | 本手册中的证据 |
| --- | --- | --- |
| 1 | 记录各 setup wall-clock 与 CB-WCE 增量成本 | Resume-chain 审计、standalone/campaign 双口径、runtime matrix。 |
| 2 | 生成与训练分离的未见需求组 | 12 个冻结 test、6 个 validation 与 11 seen 的分离；解释 interpolation/OOD 边界。 |
| 3 | Reward 与主指标使用同一交通量、聚合和方向 | 统一 $Q(t)$、$C_{[a,b)}$、$J_Q$，披露 normalization/clipping 与 adversary 的最大化方向。 |
| 4 | 每 rollout 指标及重复实验不确定性 | 完整 metric registry、10 对 seeds、sample SD 和 paired bootstrap CI。 |
| 5 | Peak-demand network heatmaps | 所有 non-internal lanes 的窗口聚合前置要求、absolute/difference 公共色标方案。 |

#### 最终验收

PDF 必须通过字体嵌入、文本提取和逐页渲染检查；LaTeX 不得含未解析引用或占位标记。Selection 必须有 8 parent、8 WCE、40 continuation、40 planned suites；Schema 示例必须验证；40 条命令不得重复且必须与 selection checkpoint hash 对应。正式结果验收则另要求 accepted=9,200、rejected=0，并完成公开 bundle 的隐私审查。

## English Evaluation and Data Validation Workbook

### Objective, Current State, and Frozen Evaluation Protocol

#### Purpose

This workbook is the operational source of truth for publication evaluation, validation, statistical analysis, site export, and paper presentation. It does not turn planned experiments into results. Complete checkpoints exist for all 40 continuation combinations, but the formal publication campaign remains at **0/9,200**. Data under `runs_eval/revised/peak_validation` are pilots only. Numbers in the current paper belong to an older baseline-versus-retrained protocol and must not be mixed with the revised five-method study.

The primary question is whether online WCE reduces network-total queue under seen and frozen held-out demand, across two networks, four controller families, and five equal-budget continuation methods, without unacceptable changes in speed, completion, reliability, or computational cost.

#### Model matrix and comparability boundary

The methods are `baseline`, `random_group`, `domain_randomization`, `fixed_wce`, and `online_wce`. Controllers are IA2C, MA2C, IQLL, and PPO; networks are Grid and Monaco. Every final controller must have 2,320,000 cumulative learning steps: a 1,000,000-step parent plus a 1,320,000-step continuation.

The current baselines are retained by decision. Audit evidence shows that baseline continuations used Python 3.10.12 / TensorFlow 2.15.1, while most other continuations used Python 3.6.13 / TensorFlow 1.12.0, with a different `envs/experiment_env.py` source hash. Consequently:

- performance claims are conditional on the frozen checkpoints and training seed 101;
- wall-clock results are descriptive rather than strict fair-runtime rankings;
- the paper must not claim that CB-WCE accelerates training or report a strict time-to-threshold advantage;
- the release must contain a runtime/source compatibility matrix.

#### Campaign size

| Item / 项目 | Frozen value / 冻结值 |
| --- | --- |
| Networks | Grid, Monaco |
| Controllers | IA2C, MA2C, IQLL, PPO |
| Methods | baseline, random group, domain randomization, fixed WCE, online WCE |
| Training seed | 101 |
| Final controller steps | 2,320,000 (1,000,000 parent + 1,320,000 continuation) |
| Evaluation horizon | 3,600 s; empty start; no warm-up; no drain |
| SUMO / control step | 1 s / 5 s (2 s yellow + 3 s green) |
| Scenarios | 11 seen + 12 test = 23 |
| Paired seeds | arrival 51001--51010 / SUMO 61001--61010 |
| Formal rollouts | 9,200 (current accepted: 0) |
| Implemented / required protocol | v7 / v7 |

The formal matrix is

$$
2\;\text{networks}\times4\;\text{controllers}\times5\;\text{methods}\times23\;\text{scenarios}\times10\;\text{pairs}=9{,}200.
$$

Each final model receives 230 rollouts and each network 4,600. The campaign represents 9,200 simulated hours, 33.12 million simulated seconds, and 6.624 million control decisions.

#### Simulation defaults

Every evaluation starts from an empty network and covers $[0,3600)$ seconds, with no warm-up and no post-horizon drain. SUMO is sampled every second. The controller acts every 5 seconds using 2 seconds of yellow followed by 3 seconds of green. Base total demand is 3,000 veh/h for Grid and 2,383.3333 veh/h for Monaco. Training episodes last 6,600 seconds; WCE receives eleven 600-second costs.

Arrival seeds are 51001--51010 and SUMO seeds are 61001--61010, paired by rollout index. For a fixed network/scenario/index, the frozen demand artifact, scenario definition, and SUMO seed are reused across every controller and method. The policy seed is produced by the protocol's deterministic digest.

#### Eleven seen and twelve test scenarios

The seen split contains the eleven normalized training profiles, each held constant for the full hour. The test split contains three scenarios in each of four families:

| Family | Scenarios | Exact definition |
| --- | --- | --- |
| OD redistribution | $\sigma=0.25,0.50,0.75$ | Multiply positive Uniform-profile ODs by independent LogNormal$(0,\sigma)$ factors and renormalize to the base total. Zero-support ODs remain zero. |
| Demand mixture | mixture 1, 2, 3 | Fixed Dirichlet$(1,\ldots,1)$ weights from the registered generation seeds form convex combinations of the eleven normalized profiles; the mixture is constant for one hour. |
| Temporal switching | 300, 900, 1200 s | At each block boundary, select the next profile from all eleven seen profiles using the frozen generation seed, with no adjacent repeat. The 300-second case is the fast-switching stress test; the 900- and 1200-second cases test longer dwell times. |
| Peak demand | $1.10,1.25,1.50$ | Base demand in $[0,1200)$, multiplied demand in $[1200,2400)$, and base demand again in $[2400,3600)$. |

The code now implements protocol v7: every temporal scenario selects from all eleven seen profiles using its frozen generation seed, with no adjacent repeat. Versioned v7 artifacts remain to be materialized and audited before formal evaluation; protocol-v6 artifacts must not enter the v7 release.

Mixtures are held-out compositions/interpolations, not strict OOD samples. Peaks are demand-intensity extrapolations; redistributions alter mass within the Uniform OD support; switches are frozen random temporal rearrangements of all eleven seen spatial profiles. The six validation scenarios are tuning-only. The frozen test suite is evaluated once after decisions are final.

Expected one-hour vehicle counts for peaks are 3,100 / 3,250 / 3,500 on Grid and approximately 2,462.78 / 2,581.94 / 2,780.56 on Monaco. Final mean/min/max counts must be regenerated from the v7 artifacts, rather than copied from a pre-amendment table.

#### Demand materialization

For every OD in each block, $N\sim\mathrm{Poisson}(r\,d/3600)$. Departure time is sampled uniformly inside the block, perturbed by $\mathcal{N}(0,2\,\mathrm{s})$, clipped inside the block, and rounded to two decimals. Speed factor is sampled from $\mathcal{N}(1,0.1)$ with nonpositive values resampled. Routes are resolved on the current SUMO network and frozen in the artifact. Every artifact stores network, profile, scenario, and content hashes.

### Metrics, Reward Alignment, and Statistical Analysis

#### Common traffic quantity

On the globally deduplicated controlled incoming-lane set $L_{\mathrm{controlled}}$, define

$$
Q(t)=\sum_{l\in L_{\mathrm{controlled}}}q_l(t),\qquad
J_Q=\frac{1}{3600}\sum_{t=1}^{3600}Q(t).
$$

Grid contains 150 and Monaco 116 controlled incoming lanes. $J_Q$ is measured in vehicles and is a **network-total mean queue**; *lower is better*. It is not a per-intersection metric.

At each 5-second control endpoint, controllers receive raw negative queue reward. MA2C uses local plus $0.9$ neighbor-weighted queue, whereas IA2C, IQLL, and PPO receive replicated network-total negative queue. The learner uses

$$
r_{\mathrm{ctrl}}=\operatorname{clip}(r_{\mathrm{raw}}/d_c,-2,2),
$$

with $d_c$ read from the selected run manifest.

For any interval define

$$
C_{[a,b)}=\frac{1}{b-a}\sum_{t=a+1}^{b}Q(t).
$$

CB-WCE maximizes each 600-second cost $C_k=C_{[600k,600(k+1))}$ and learns from

$$
r_{\mathrm{WCE}}=\operatorname{clip}(C_k/d_w,-2,2),
\quad d_w=3000\;(\mathrm{Grid}),\;1000\;(\mathrm{Monaco}).
$$

Evaluation always reports raw, unscaled, unclipped $J_Q$. Positive scaling preserves direction, but clipping can break strict monotonic equivalence. Controller and WCE clipping fractions must be computed for every selected training run. Strict alignment may be claimed only when the fraction is zero; otherwise the observed fraction and limitation must be reported.

#### Rollout metric registry

| Field | Unit | Direction | Definition and denominator |
| --- | --- | --- | --- |
| mean_total_queue | vehicles | lower | $3600^{-1}\sum_tQ(t)$; primary outcome. |
| integrated_queue | vehicle-seconds | lower | $\sum_tQ(t)$. |
| peak / p95 queue | vehicles | lower | Maximum and 95th percentile of per-second network-total queue. |
| window queue/AUC | vehicles, vehicle-seconds | lower | Separate summaries for three 1,200-second windows. |
| last_600_s_slope | vehicles/second | lower | Preregistered linear slope over the last 600 queue samples; descriptive stability indicator. |
| mean_speed | m/s | higher | $\sum_t\mathrm{speed_sum}_t/\sum_t\mathrm{active}_t$ with denominator stored. |
| scheduled/inserted/completed | vehicles | context | Artifact vehicles, departed vehicles, and arrived vehicles. |
| pending/remaining | vehicles | lower | Not departed and still active at 3,600 seconds. |
| completion_rate | ratio | higher | completed/inserted; null when inserted is zero. |
| completed trip means | seconds | lower | Travel, waiting, time loss, and departure delay for completed vehicles only; denominator and unfinished counts are mandatory. |
| teleports/collisions | count | lower | Reliability indicators. Different network teleport thresholds prohibit direct cross-network ranking. |
| wall_seconds | seconds | lower | Monotonic-clock evaluation duration. |

#### Preregistered paired comparisons

The four primary comparisons subtract baseline, random group, domain randomization, and fixed WCE from online WCE. For scenario $s$ and rollout pair $r$:

$$
d_{s,r}=J_{\mathrm{online},s,r}-J_{\mathrm{comparator},s,r}.
$$

A negative queue difference favors online WCE. For each scenario report the mean, sample SD (`ddof=1`), and deterministic 10,000-resample paired percentile-bootstrap 95% CI for $n=10$ pairs. The bootstrap seed is a digest of a fixed analysis label, network, controller, comparator, scope, and scenario.

For a family or the full fixed test suite, first average the registered scenario differences within each rollout index,

$$
D_r=|S|^{-1}\sum_{s\in S}d_{s,r},
$$

then bootstrap the ten $D_r$ blocks. Per-second values, lane observations, and 120 cross-scenario rollouts must not be treated as independent IID observations. Intervals describe simulation/policy variation conditional on the fixed checkpoints and fixed suite; they are not training-seed uncertainty. A single training seed cannot establish algorithm-wide statistical superiority. Apply Holm correction if multiplicity-adjusted hypothesis tests are added.

Queue percent improvement, $(J_{\mathrm{comp}}-J_{\mathrm{online}})/J_{\mathrm{comp}}\times100%$, is supplemental to the absolute effect and is omitted for near-zero denominators. Speed uses the reverse direction, with absolute m/s differences primary.

#### Robustness and worst cases

Report the equal-weight 12-scenario mean, maximum scenario mean, worst-three average/CVaR, and across-scenario SD. Different methods may have different worst scenarios; one method's selected worst-case curve must not be presented as a common paired worst case. Queue-worst and speed-worst selections are separate.

#### Training and evaluation wall time

Trace the `resume` chain from each selected final manifest and sum parent, offline-WCE, continuation, evaluation, and abandoned/duplicate attempt time. Use `result.json.wall_seconds` for completed attempts. If an earlier attempt lacks a result, use the final `progress.jsonl.wall_seconds` as a lower bound and label the total `exact` or `lower_bound`.

Standalone cost charges offline WCE to fixed and online WCE; deduplicated campaign cost counts shared parents and WCE once. Component times are not a complete decomposition and cannot replace total wall time. Cost tables include OS, CPU/RAM when available, Python, TensorFlow, thread and concurrency settings, and source hashes. The baseline runtime mismatch is a mandatory table note.

### Data Storage Contract and Release Validation

#### Three data layers

1. **Authoritative raw**: `runs_eval/revised/publication_seed101_v1/<network>/<controller>/<method>/`, containing immutable manifests, environments, summaries, results, compressed per-second records, and NPZ arrays.
1. **Validated release**: `output_result/revised/publication_seed101_v1/`, containing release/validation manifests, CSV, figures, paired-bootstrap outputs, and a compatible dashboard JSON.
1. **Sanitized site bundle**: `docs/site/dist/data/publication/`, containing only allowlisted aggregates and de-identified rollout fields.

JSON is authoritative and CSV is a human-readable export. JSON forbids NaN and Infinity. Missing or inapplicable values use `null`, with reason, unit, direction, and denominator documented in the metric registry. Raw data are never copied into the hosted site.

#### Selection registry and primary key

`runs_eval/revised/selections/final_evaluation_seed101.json` freezes eight parents, eight offline WCE models, forty final continuations, and forty planned evaluation suites. Each item stores repo-relative and absolute execution paths, checkpoint step, aggregate directory SHA-256, manifest hash, manifest-file SHA-256, runtime/source compatibility, and planned output directory. Selection by mtime, latest directory, or broad scan is prohibited.

The rollout primary key is

```text
network/controller/method/training_seed/split/scenario/
generation_seed/arrival_seed/sumo_seed/policy_seed/attempt
```

The public rollout ID is a digest of stable fields and never contains an absolute path.

#### Publication-complete gate

`publication_complete=true` requires every condition below:

- exactly 9,200 accepted and zero rejected, missing, duplicate, or extra records;
- all forty checkpoints are allowlisted and contain 2,320,000 controller learning steps;
- every record has `pilot=false`, `status=complete`, and `horizon=sample_count=3600`;
- arrival seeds 51001--51010 map exactly to SUMO seeds 61001--61010;
- demand hash and SUMO seed match across all twenty controller-method combinations for a fixed network/scenario/index;
- protocol, scenario, profile, network, artifact, checkpoint, and manifest hashes match frozen inputs;
- timestamps are exactly 1--3600 and queues are nonnegative and finite;
- integrated queue agrees with $3600\times$ mean queue and peak is not below mean;
- scheduled = inserted + pending, completed $\le$ inserted, and completed-trip denominator = completed;
- the NPZ lane sum agrees with the JSONL network total;
- the public bundle contains no absolute paths, credentials, checkpoint payloads, raw logs, pilots, or historical records.

#### Disk and compression gate

The root filesystem had approximately 69 GiB free on 2026-09-29, above the 60-GiB launch threshold, but space must be checked immediately before every campaign. The current pilot layout projects to more than 33 GB for 9,200 rollouts. Launch requires at least 60 GiB free and at least 20 GiB projected reserve after completion. Otherwise implement suite-level shared provenance, smaller per-rollout manifests, gzip JSONL, and compressed NPZ, then remeasure with a pilot. Unknown user data must never be deleted to create space.

#### Reward and environment validation

Release validation computes controller/WCE clipping fractions and emits a source/config/runtime compatibility matrix. The retained baseline mismatch does not block the user-selected release, but it sets `strict_runtime_comparability=false` and prohibits strong claims about training speed or strict runtime fairness.

### Local/Static Site Contract and Paper Visualizations

#### Hybrid data architecture

The existing `docs/site/.openai/hosting.json` describes a static Sites project; its architecture and project ID are preserved. This delivery does not edit or publish the site. A hosted browser cannot scan the experiment machine. “Automatic reading” therefore means that a local dashboard/API discovers allowlisted releases, validates them, and produces a sanitized static snapshot; the hosted site fetches only relative published JSON.

The next-stage local API specification is:

```text
GET /api/releases
GET /api/releases/<id>/manifest
GET /api/releases/<id>/overview
GET /api/releases/<id>/rollouts?network=&controller=&method=&split=&scenario=
```

The public bundle contains `manifest.json`, `overview.json`, eight network-controller rollout shards (1,150 records each), twenty-four network-controller-peak heatmap shards, optional curve shards, and `checksums.json`. The old `results.js` contains pilot schema-v1 data and is not a publication source.

The first viewport exposes 9,200/9,200 completeness, rejected=0, protocol hash, release time, the primary metric, and network/controller/method/scenario filters. Further views provide paired intervals, method-scenario heatmaps, time series, flow/reliability, computational cost, JSON/CSV downloads, and Chinese/English labels.

#### Paper figures and supplements

1. **Paired-effect forest plot**: absolute online-WCE-minus-comparator queue effects with conditional 95% CIs, a zero reference, and “negative is better.”
1. **Method $\times$ scenario heatmap**: twelve test scenarios with absolute $J_Q$ and relative-to-baseline modes; seen and test remain separate.
1. **Robustness frontier**: suite average versus worst-three CVaR.
1. **Peak response**: preregistered peak-1.25 and peak-1.50 $Q(t)$ means and intervals in the main paper with $[1200,2400)$ shaded; peak 1.10 in the supplement.
1. **Network heatmaps**: five absolute method maps and online-minus-comparator difference maps; common absolute scales within network/metric and zero-centered symmetric difference scales.
1. **Training cost**: stacked parent/offline-WCE/continuation bars showing standalone and deduplicated campaign cost, with exact/lower-bound markers.
1. **Flow/reliability**: completed, remaining, pending, teleports, and collisions; completed-trip metrics always show their denominator.
1. **Mechanism**: WCE mixture weights and entropy in supplementary material.

A true network-wide heatmap requires online aggregation for all non-internal lanes in pre-peak, peak, and post-peak windows before formal evaluation. Current data cover controlled incoming lanes only. Without that extension the figure must be named “network-spanning controlled-approach heatmap,” not an all-lane network heatmap.

#### Paper wording checklist

- Replace “per intersection” with “network-total mean queue.”
- Replace “unbounded growth” with “sustained growth within the 3,600-s horizon.”
- Replace “algorithm-agnostic confirmed” with “consistent trend across four architectures for training seed 101.”
- Remove “accelerates training” without common-runtime time-to-threshold evidence.
- Lead with absolute effects and keep percentages supplemental; select queue-worst and speed-worst scenarios separately.

### Execution Workbook and Sign-off

#### Phase 0 prerequisites

1. Implement protocol v7 so all three temporal scenarios switch among the eleven seen profiles using frozen seeds with no adjacent repeat, then regenerate and hash all sixty affected artifacts.
1. Implement all-non-internal-lane heatmap aggregation or formally lock the controlled-approach wording.
1. Rerun the environment gate for deeprlsc, SUMO, TraCI, TensorFlow, both networks, and all eleven profiles.
1. Recheck disk thresholds and all forty checkpoint hashes; require every output directory to be absent.
1. Independently review the one-to-one selection-registry and command-catalogue mapping.

#### Commands available now

```text
cd /home/sdc_joran/Journal/deeprl_signal_control
export CBWCE_GATE=/absolute/path/to/protocol_v7/gate.json

./docs/evaluation_workbook/generated/publication_workflow.sh --status
./docs/evaluation_workbook/generated/publication_workflow.sh --preflight
./docs/evaluation_workbook/generated/publication_workflow.sh --list

CONFIRM_PUBLICATION=RUN_9200 CBWCE_EVALUATION_WORKERS=4 \
  ./docs/evaluation_workbook/generated/publication_workflow.sh --execute
```

`publication_workflow.sh` prints the plan by default and does not start simulation. Formal execution requires an explicit confirmation token, protocol v7, at least 60 GiB free, a valid gate recorded with four workers, pre-materialized demand artifacts, and an absent output root. The scheduler then runs four suites in parallel by default; the 230 rollouts inside each suite remain sequential. A suite failure stops new dispatch, with no automatic retry or overwrite.

#### Next-stage commands to implement

These are interface specifications, not current capabilities:

```text
python main.py experiment report --selection <selection.json> --output <release>
python main.py experiment validate-report --report <release> \
  --require-publication-complete
python main.py experiment export-site --report <release> \
  --output docs/site/dist/data/publication
```

#### Failure, resume, and duplicate prevention

A retried attempt is not an additional random replicate. Resume follows the manifest's `resume` chain. The final report accepts only the selected successful attempt; failed, interrupted, and duplicate attempts enter a separate cost audit. Any change to demand artifacts, seeds, checkpoints, or scenario hashes requires a new protocol/release ID and must not overwrite history.

#### Reviewer-response matrix

| No. | Request | Evidence produced by this workbook |
| --- | --- | --- |
| 1 | Record setup wall-clock and CB-WCE overhead | Resume-chain audit, standalone/campaign accounting, and runtime matrix. |
| 2 | Generate unseen groups separate from training | Frozen 12-test/6-validation/11-seen separation with interpolation/OOD classification. |
| 3 | Align reward and primary traffic metric | Common $Q(t)$, $C_{[a,b)}$, and $J_Q$ definitions with normalization, clipping, and adversarial direction disclosed. |
| 4 | Record rollout metrics and uncertainty | Complete registry, ten paired seeds, sample SD, and paired bootstrap CI. |
| 5 | Produce peak-demand network heatmaps | Prerequisite all-non-internal-lane window aggregation and common-scale absolute/difference map specification. |

#### Acceptance

The PDF must pass embedded-font, text-extraction, and page-render review; LaTeX must contain no unresolved references or placeholder markers. The selection must contain eight parents, eight WCE models, forty continuations, and forty planned suites. Schema examples must validate. The forty commands must be unique and agree with selected checkpoint hashes. Result acceptance later additionally requires 9,200 accepted, zero rejected, and a privacy review of the public bundle.

## Shared Generated Appendices / 共享自动生成附录

### Frozen Protocol Snapshot / 冻结协议快照

| Item / 项目 | Frozen value / 冻结值 |
| --- | --- |
| Networks | Grid, Monaco |
| Controllers | IA2C, MA2C, IQLL, PPO |
| Methods | baseline, random group, domain randomization, fixed WCE, online WCE |
| Training seed | 101 |
| Final controller steps | 2,320,000 (1,000,000 parent + 1,320,000 continuation) |
| Evaluation horizon | 3,600 s; empty start; no warm-up; no drain |
| SUMO / control step | 1 s / 5 s (2 s yellow + 3 s green) |
| Scenarios | 11 seen + 12 test = 23 |
| Paired seeds | arrival 51001--51010 / SUMO 61001--61010 |
| Formal rollouts | 9,200 (current accepted: 0) |
| Implemented / required protocol | v7 / v7 |

#### Named mixture weights / 按名称列出的 mixture 权重

The same numerical vector is interpreted against each network's explicit profile order. Values are generated from NumPy RandomState with seeds 41004--41006.

##### Grid

| Profile | Mixture 1 | Mixture 2 | Mixture 3 |
| --- | --- | --- | --- |
| Center_to_Periphery | 0.003033506 | 0.092474867 | 0.107161264 |
| E_to_W | 0.146417727 | 0.119750935 | 0.075787215 |
| NE_to_SW | 0.052147496 | 0.025368038 | 0.003491398 |
| NW_to_SE | 0.084043500 | 0.018755111 | 0.033028230 |
| N_to_S | 0.299739949 | 0.039273965 | 0.126210811 |
| Periphery_to_Center | 0.061484199 | 0.130447278 | 0.030165314 |
| SE_to_NW | 0.005641641 | 0.045555643 | 0.393794939 |
| SW_to_NE | 0.012238742 | 0.171576624 | 0.160653095 |
| S_to_N | 0.085509345 | 0.124758300 | 0.054454339 |
| Uniform | 0.027042405 | 0.192396667 | 0.009800486 |
| W_to_E | 0.222701489 | 0.039642572 | 0.005452909 |

##### Monaco

| Profile | Mixture 1 | Mixture 2 | Mixture 3 |
| --- | --- | --- | --- |
| N_to_S | 0.003033506 | 0.092474867 | 0.107161264 |
| S_to_N | 0.146417727 | 0.119750935 | 0.075787215 |
| W_to_E | 0.052147496 | 0.025368038 | 0.003491398 |
| E_to_W | 0.084043500 | 0.018755111 | 0.033028230 |
| NW_to_SE | 0.299739949 | 0.039273965 | 0.126210811 |
| SE_to_NW | 0.061484199 | 0.130447278 | 0.030165314 |
| SW_to_NE | 0.005641641 | 0.045555643 | 0.393794939 |
| NE_to_SW | 0.012238742 | 0.171576624 | 0.160653095 |
| Periphery_to_Center | 0.085509345 | 0.124758300 | 0.054454339 |
| Center_to_Periphery | 0.027042405 | 0.192396667 | 0.009800486 |
| Uniform | 0.222701489 | 0.039642572 | 0.005452909 |

#### Materialized test vehicle counts / 已物化测试车辆数

Protocol-v7 artifact counts remain pending until the versioned demand suite is materialized; no protocol-v6 artifact is included here.

| Network | Scenario | Mean | Min | Max |
| --- | --- | --- | --- | --- |
| Grid | redistribution_0.25 | pending | -- | -- |
| Grid | redistribution_0.5 | pending | -- | -- |
| Grid | redistribution_0.75 | pending | -- | -- |
| Grid | mixture_1 | pending | -- | -- |
| Grid | mixture_2 | pending | -- | -- |
| Grid | mixture_3 | pending | -- | -- |
| Grid | switch_300 | pending | -- | -- |
| Grid | switch_900 | pending | -- | -- |
| Grid | switch_1200 | pending | -- | -- |
| Grid | peak_1.1 | pending | -- | -- |
| Grid | peak_1.25 | pending | -- | -- |
| Grid | peak_1.5 | pending | -- | -- |
| Monaco | redistribution_0.25 | pending | -- | -- |
| Monaco | redistribution_0.5 | pending | -- | -- |
| Monaco | redistribution_0.75 | pending | -- | -- |
| Monaco | mixture_1 | pending | -- | -- |
| Monaco | mixture_2 | pending | -- | -- |
| Monaco | mixture_3 | pending | -- | -- |
| Monaco | switch_300 | pending | -- | -- |
| Monaco | switch_900 | pending | -- | -- |
| Monaco | switch_1200 | pending | -- | -- |
| Monaco | peak_1.1 | pending | -- | -- |
| Monaco | peak_1.25 | pending | -- | -- |
| Monaco | peak_1.5 | pending | -- | -- |

### Selected Models / 最终模型清单

| No. | Network | Controller | Method | Runtime | Exact checkpoint (repo-relative) |
| --- | --- | --- | --- | --- | --- |
| 1 | grid | ia2c | baseline | Py 3.10.12/TF 2.15.1 | `runs/revised/grid/ia2c/seed_101/baseline/publication_20260919T110516_bffa1fe4/checkpoint_001320000` |
| 2 | grid | ia2c | random_group | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/ia2c/seed_101/random_group/publication_20260922T114628_37d306e3/checkpoint_001320000` |
| 3 | grid | ia2c | domain_randomization | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/ia2c/seed_101/domain_randomization/publication_20260923T145705_7a6b9270/checkpoint_001320000` |
| 4 | grid | ia2c | fixed_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/ia2c/seed_101/fixed_wce/publication_20260924T054007_773e599b/checkpoint_001320000` |
| 5 | grid | ia2c | online_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/ia2c/seed_101/online_wce/publication_20260925T092820_79e065be/checkpoint_001320000` |
| 6 | grid | ma2c | baseline | Py 3.10.12/TF 2.15.1 | `runs/revised/grid/ma2c/seed_101/baseline/publication_20260919T110516_de174e9f/checkpoint_001320000` |
| 7 | grid | ma2c | random_group | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/ma2c/seed_101/random_group/publication_20260922T114628_4242218e/checkpoint_001320000` |
| 8 | grid | ma2c | domain_randomization | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/ma2c/seed_101/domain_randomization/publication_20260923T145705_080f7af1/checkpoint_001320000` |
| 9 | grid | ma2c | fixed_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/ma2c/seed_101/fixed_wce/publication_20260924T082634_aa85666b/checkpoint_001320000` |
| 10 | grid | ma2c | online_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/ma2c/seed_101/online_wce/publication_20260925T102236_52e76186/checkpoint_001320000` |
| 11 | grid | iqll | baseline | Py 3.6.13/TF 1.12.0 | `runs/revised/grid/iqll/seed_101/baseline/publication_20260921T101956_966174e3/checkpoint_001320000` |
| 12 | grid | iqll | random_group | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/iqll/seed_101/random_group/publication_20260922T114628_05ccf7be/checkpoint_001320000` |
| 13 | grid | iqll | domain_randomization | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/iqll/seed_101/domain_randomization/publication_20260923T145705_6c1d3fb4/checkpoint_001320000` |
| 14 | grid | iqll | fixed_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/iqll/seed_101/fixed_wce/publication_20260924T102114_f27b5c6f/checkpoint_001320000` |
| 15 | grid | iqll | online_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/iqll/seed_101/online_wce/publication_20260925T103928_e2318819/checkpoint_001320000` |
| 16 | grid | ppo | baseline | Py 3.10.12/TF 2.15.1 | `runs/revised/grid/ppo/seed_101/baseline/publication_20260921T091601_2d3778ff/checkpoint_001320000` |
| 17 | grid | ppo | random_group | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/ppo/seed_101/random_group/publication_20260922T114628_8937c084/checkpoint_001320000` |
| 18 | grid | ppo | domain_randomization | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/ppo/seed_101/domain_randomization/publication_20260923T145705_6a9f9cf7/checkpoint_001320000` |
| 19 | grid | ppo | fixed_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/ppo/seed_101/fixed_wce/publication_20260924T112233_d23fa5d8/checkpoint_001320000` |
| 20 | grid | ppo | online_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution/revised/grid/ppo/seed_101/online_wce/publication_20260925T111448_26956f37/checkpoint_001320000` |
| 21 | monaco | ia2c | baseline | Py 3.6.13/TF 1.12.0 | `runs/revised/monaco/ia2c/seed_101/baseline/publication_20260921T101956_d364f6a8/checkpoint_001320000` |
| 22 | monaco | ia2c | random_group | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/ia2c/seed_101/random_group/publication_20260922T114628_0aad1a36/checkpoint_001320000` |
| 23 | monaco | ia2c | domain_randomization | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/ia2c/seed_101/domain_randomization/publication_20260923T145705_25ac16f3/checkpoint_001320000` |
| 24 | monaco | ia2c | fixed_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/ia2c/seed_101/fixed_wce/publication_20260924T132346_8a25e1dd/checkpoint_001320000` |
| 25 | monaco | ia2c | online_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/ia2c/seed_101/online_wce/publication_20260925T133729_5eb7e27b/checkpoint_001320000` |
| 26 | monaco | ma2c | baseline | Py 3.6.13/TF 1.12.0 | `runs/revised/monaco/ma2c/seed_101/baseline/publication_20260921T232411_e7c37342/checkpoint_001320000` |
| 27 | monaco | ma2c | random_group | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/ma2c/seed_101/random_group/publication_20260922T114628_a9f21707/checkpoint_001320000` |
| 28 | monaco | ma2c | domain_randomization | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/ma2c/seed_101/domain_randomization/publication_20260923T145705_0e8b380e/checkpoint_001320000` |
| 29 | monaco | ma2c | fixed_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/ma2c/seed_101/fixed_wce/publication_20260924T152812_121189a7/checkpoint_001320000` |
| 30 | monaco | ma2c | online_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/ma2c/seed_101/online_wce/publication_20260925T182406_2c5ff93d/checkpoint_001320000` |
| 31 | monaco | iqll | baseline | Py 3.6.13/TF 1.12.0 | `runs/revised/monaco/iqll/seed_101/baseline/publication_20260921T101956_50fdfb9e/checkpoint_001320000` |
| 32 | monaco | iqll | random_group | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/iqll/seed_101/random_group/publication_20260922T114628_9ead06e3/checkpoint_001320000` |
| 33 | monaco | iqll | domain_randomization | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/iqll/seed_101/domain_randomization/publication_20260923T145705_a1c98737/checkpoint_001320000` |
| 34 | monaco | iqll | fixed_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/iqll/seed_101/fixed_wce/publication_20260925T010612_973d5910/checkpoint_001320000` |
| 35 | monaco | iqll | online_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/iqll/seed_101/online_wce/publication_20260926T015715_cac25e47/checkpoint_001320000` |
| 36 | monaco | ppo | baseline | Py 3.6.13/TF 1.12.0 | `runs/revised/monaco/ppo/seed_101/baseline/publication_20260921T101956_21fa0919/checkpoint_001320000` |
| 37 | monaco | ppo | random_group | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/ppo/seed_101/random_group/publication_20260922T114628_0881e4fb/checkpoint_001320000` |
| 38 | monaco | ppo | domain_randomization | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/ppo/seed_101/domain_randomization/publication_20260923T145705_294f5ccd/checkpoint_001320000` |
| 39 | monaco | ppo | fixed_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/ppo/seed_101/fixed_wce/publication_20260925T052439_8e7a81ea/checkpoint_001320000` |
| 40 | monaco | ppo | online_wce | Py 3.6.13/TF 1.12.0 | `output_coevolution_real/revised/monaco/ppo/seed_101/online_wce/publication_20260926T094349_1a6ecd92/checkpoint_001320000` |

Full absolute paths, aggregate checkpoint hashes, manifest hashes, source hashes, and planned suite outputs are authoritative in `runs_eval/revised/selections/final_evaluation_seed101.json`.

### Command Catalogue / 命令目录

#### Command status / 命令状态

The environment check, evaluation, current report, and dashboard commands exist now. The proposed selection-driven report, validate-report, and export-site interfaces are specifications only and are not represented as current capabilities.

The exact command catalogue is `docs/evaluation_workbook/generated/evaluate_publication_seed101.sh`; it retains forty explicit invocations and a sequential fallback. The primary entry point is `docs/evaluation_workbook/generated/publication_workflow.sh`. It is safe by default and uses `docs/evaluation_workbook/run_publication_parallel.py` to run four suites concurrently only after protocol, gate, storage, selection, artifact, and output-root checks pass. Rollouts remain sequential within each suite, and failures stop new dispatch without automatic retry.

```text
./docs/evaluation_workbook/generated/evaluate_publication_seed101.sh --preflight
./docs/evaluation_workbook/generated/evaluate_publication_seed101.sh --list
./docs/evaluation_workbook/generated/evaluate_publication_seed101.sh --execute
```

#### Evaluation campaign script

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/sdc_joran/Journal/deeprl_signal_control
GATE=${CBWCE_GATE:-}
MODE=${1:-}

usage() {
  echo 'Usage: evaluate_publication_seed101.sh --preflight | --list | --execute'
  echo 'This script will not execute formal evaluation unless protocol v7 and storage gates pass.'
}

preflight() {
  cd "$ROOT"
  version=$(python3 -c "import json; print(json.load(open('config/revised/protocol.json'))['version'])")
  if [ "$version" != 7 ]; then
    echo "BLOCKED: protocol v7 is required; current version is $version." >&2
    return 2
  fi
  if [ -z "$GATE" ]; then echo "BLOCKED: export CBWCE_GATE to a current protocol-v7 gate.json." >&2; return 4; fi
  available_kb=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
  if [ "$available_kb" -lt 62914560 ]; then
    echo "BLOCKED: at least 60 GiB free is required." >&2
    return 3
  fi
  conda run -n deeprlsc python -c "from experiments.runner import require_gate; require_gate(r'''$GATE'''); print('PASS current verification gate')"
}

run_all() {
  cd "$ROOT"
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ia2c/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ia2c/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/grid/ia2c/seed_101/baseline/publication_20260919T110516_bffa1fe4/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ia2c/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ia2c/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ia2c/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ia2c/seed_101/random_group/publication_20260922T114628_37d306e3/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ia2c/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ia2c/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ia2c/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ia2c/seed_101/domain_randomization/publication_20260923T145705_7a6b9270/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ia2c/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ia2c/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ia2c/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ia2c/seed_101/fixed_wce/publication_20260924T054007_773e599b/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ia2c/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ia2c/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ia2c/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ia2c/seed_101/online_wce/publication_20260925T092820_79e065be/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ia2c/online_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ma2c/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ma2c/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/grid/ma2c/seed_101/baseline/publication_20260919T110516_de174e9f/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ma2c/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ma2c/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ma2c/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ma2c/seed_101/random_group/publication_20260922T114628_4242218e/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ma2c/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ma2c/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ma2c/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ma2c/seed_101/domain_randomization/publication_20260923T145705_080f7af1/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ma2c/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ma2c/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ma2c/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ma2c/seed_101/fixed_wce/publication_20260924T082634_aa85666b/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ma2c/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ma2c/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ma2c/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ma2c/seed_101/online_wce/publication_20260925T102236_52e76186/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ma2c/online_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/iqll/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/iqll/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/grid/iqll/seed_101/baseline/publication_20260921T101956_966174e3/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/iqll/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/iqll/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/iqll/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/iqll/seed_101/random_group/publication_20260922T114628_05ccf7be/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/iqll/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/iqll/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/iqll/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/iqll/seed_101/domain_randomization/publication_20260923T145705_6c1d3fb4/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/iqll/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/iqll/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/iqll/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/iqll/seed_101/fixed_wce/publication_20260924T102114_f27b5c6f/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/iqll/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/iqll/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/iqll/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/iqll/seed_101/online_wce/publication_20260925T103928_e2318819/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/iqll/online_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ppo/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ppo/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/grid/ppo/seed_101/baseline/publication_20260921T091601_2d3778ff/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ppo/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ppo/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ppo/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ppo/seed_101/random_group/publication_20260922T114628_8937c084/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ppo/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ppo/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ppo/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ppo/seed_101/domain_randomization/publication_20260923T145705_6a9f9cf7/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ppo/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ppo/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ppo/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ppo/seed_101/fixed_wce/publication_20260924T112233_d23fa5d8/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ppo/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ppo/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ppo/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ppo/seed_101/online_wce/publication_20260925T111448_26956f37/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ppo/online_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ia2c/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ia2c/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/monaco/ia2c/seed_101/baseline/publication_20260921T101956_d364f6a8/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ia2c/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ia2c/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ia2c/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ia2c/seed_101/random_group/publication_20260922T114628_0aad1a36/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ia2c/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ia2c/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ia2c/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ia2c/seed_101/domain_randomization/publication_20260923T145705_25ac16f3/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ia2c/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ia2c/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ia2c/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ia2c/seed_101/fixed_wce/publication_20260924T132346_8a25e1dd/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ia2c/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ia2c/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ia2c/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ia2c/seed_101/online_wce/publication_20260925T133729_5eb7e27b/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ia2c/online_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ma2c/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ma2c/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/monaco/ma2c/seed_101/baseline/publication_20260921T232411_e7c37342/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ma2c/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ma2c/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ma2c/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ma2c/seed_101/random_group/publication_20260922T114628_a9f21707/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ma2c/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ma2c/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ma2c/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ma2c/seed_101/domain_randomization/publication_20260923T145705_0e8b380e/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ma2c/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ma2c/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ma2c/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ma2c/seed_101/fixed_wce/publication_20260924T152812_121189a7/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ma2c/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ma2c/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ma2c/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ma2c/seed_101/online_wce/publication_20260925T182406_2c5ff93d/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ma2c/online_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/iqll/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/iqll/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/monaco/iqll/seed_101/baseline/publication_20260921T101956_50fdfb9e/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/iqll/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/iqll/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/iqll/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/iqll/seed_101/random_group/publication_20260922T114628_9ead06e3/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/iqll/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/iqll/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/iqll/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/iqll/seed_101/domain_randomization/publication_20260923T145705_a1c98737/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/iqll/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/iqll/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/iqll/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/iqll/seed_101/fixed_wce/publication_20260925T010612_973d5910/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/iqll/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/iqll/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/iqll/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/iqll/seed_101/online_wce/publication_20260926T015715_cac25e47/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/iqll/online_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ppo/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ppo/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/monaco/ppo/seed_101/baseline/publication_20260921T101956_21fa0919/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ppo/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ppo/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ppo/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ppo/seed_101/random_group/publication_20260922T114628_0881e4fb/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ppo/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ppo/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ppo/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ppo/seed_101/domain_randomization/publication_20260923T145705_294f5ccd/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ppo/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ppo/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ppo/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ppo/seed_101/fixed_wce/publication_20260925T052439_8e7a81ea/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ppo/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ppo/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ppo/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ppo/seed_101/online_wce/publication_20260926T094349_1a6ecd92/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ppo/online_wce'
}

case "$MODE" in
  --preflight) preflight ;;
  --list) sed -n '/^run_all()/,/^}/p' "$0" ;;
  --execute) preflight; run_all ;;
  *) usage; exit 1 ;;
esac
```

### Machine-readable Contracts / 机器可读协议

The authoritative JSON Schemas are stored in `docs/evaluation_workbook/schemas/`. The selection registry is `runs_eval/revised/selections/final_evaluation_seed101.json`. JSON is authoritative; CSV is a human-readable export only.

### Release sign-off / 发布签字页

| Check | Sign-off |
| --- | --- |
| Protocol v7 implemented and hashed | ____________________ |
| Storage gate rechecked immediately before launch | ____________________ |
| 40 checkpoints and commands independently reviewed | ____________________ |
| 9,200 accepted, zero rejected | ____________________ |
| Public export privacy review | ____________________ |
| Paper figures and wording review | ____________________ |
| Reviewer / date | ____________________ |
