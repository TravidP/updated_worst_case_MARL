# Revision Plan v8.1：独立项目、验证驱动训练与 Hábrók 部署

日期：2026-10-27。对应英文版：[revision_plan_v8_en.md](revision_plan_v8_en.md)。

本文件规定实验协议和验收标准，不把代码存在等同于已经通过验收。实际检查结果单独记录在 `implementation_status.md`；不宣称已完成正式训练、容器构建或 Hábrók 部署。

## 1. 目标与实验矩阵

建立 `cbwce_standalone` 独立精简项目，在 Groningen 的 Hábrók 上运行。保留 MA2C、DQL、PPO，使用验证选模、平台停止、精简输出及可恢复 Slurm 作业。

| 项目 | 协议 |
|---|---|
| 地图 | 5×5 Grid、当前修订入口使用的 Monaco 子路网；不混入修复版地图 |
| 正式种子 | 101、202、303；开发种子 9001 |
| 训练回合 | 6600 秒，1320 个全路网联合控制步 |
| 控制／需求间隔 | 5 秒／600 秒 |
| 初始训练 | 十一种原需求配置的固定顺序 |
| 续训方法 | `baseline`、`random_group`、`domain_randomization`、`fixed_wce`、`online_wce` |
| 选模 | 控制器最小化整体平均排队；WCE 最大化冻结父模型的平均排队 |
| 比较终点 | 共同预算内 best，以及各自停止时 best |

**验证场景数、组成和生成参数尚由用户确定。** 必须提供冻结 manifest，不能用程序自带 Uniform 开发样例替代正式验证集。开发验证、正式验证和最终测试分别保存身份和随机流。

正式规模：18 个父模型、18 个离线 WCE、90 个续训分支。最终测试暂保留十一种 seen 场景和十二种泛化场景，每场景十次：每类终点 20,700 次评估；相同模型哈希的两个终点可以复用评估。

## 2. 独立保存和项目组织

原工作区约 164 GB（检查时），其中评估约 64 GB、共演化约 42 GB、评估图目录约 24 GB、Git 历史约 18 GB。不得直接复制整个工作区。

新项目采用新 Git 历史，按白名单提取实际运行代码、两套地图、各十一份需求、配置、测试、文档、Slurm 模板、许可证和原作者信息。记录原提交、工作区差异和每个提取文件的哈希。不能只导出 Git 提交而遗漏未提交修改。

Monaco 只复制运行依赖的网络和检测器输入，不复制整套历史仿真目录。原项目及旧结果保持原样。

```text
cbwce_standalone/
  agents/ envs/ experiments/ config/ assets/ tests/
  scripts/slurm/ docs/ packaging/ provenance/ main.py
cbwce_data/       # 共享、版本化输入，项目外
cbwce_runs/       # 运行结果，项目外
cbwce_archive/    # 经校验的长期成果，项目外
```

支持 `CBWCE_ASSETS_ROOT`、`CBWCE_DATA_ROOT`、`CBWCE_RUN_ROOT`、`CBWCE_CACHE_ROOT`。不能依赖旧绝对路径、外部符号链接或旧目录导入。内容哈希、协议版本和运行身份决定兼容性；机器路径仅作定位。

分别制作代码发布包、输入发布包和运行归档包，均附文件清单及 SHA-256。相同地图、交通输入和固定模型通过引用共享。

## 3. 控制器与参数

IA2C 从新矩阵移除；MA2C 保留邻居观测、fingerprint、邻域奖励和 LSTM。开启有效样本 advantage normalization，排除 padding；邻居奖励权重从配置读取，默认 0.9。检查 entropy 梯度、策略梯度、value loss 的有效系数、奖励裁剪和梯度裁剪。奖励只在 learner 边界缩放一次。保留 n-step return，不同时引入 GAE；MA2C 每 rollout 更新一次。

DQL 替换线性 `iqll`，每路口独立 `128 ReLU → 64 ReLU → Q`；独立 online／target 网络，均匀 replay、Huber loss、epsilon-greedy，验证使用 greedy。不加入 Double DQN、Dueling 或 prioritized replay。

| 参数 | MA2C | PPO | DQL |
|---|---:|---:|---:|
| 学习率 | 5e-4 | 3e-4 | 1e-4 |
| gamma | 0.99 | 0.99 | 0.99 |
| 优化器 | RMSProp | Adam | Adam |
| LSTM | 64 | 64 | 无 |
| Advantage normalization | 开 | 开 | 不适用 |
| Rollout／batch | 120 | 120 | 64 |
| 更新 | 每 rollout 一次 | 每 rollout 四轮 | 每 4 个学习步一次 |
| Entropy | 0.01 | 0.01 | 不适用 |
| PPO clip | — | 0.2 | — |
| Replay | — | — | 50,000 联合转移 |
| Warm-up | — | — | 5000 步 |
| Target 同步 | — | — | 每 1000 学习步 |
| 梯度范数上限 | 40 | 40 | 10 |

保留现有 MA2C／PPO 编码宽度。DQL 初始 epsilon 在 500,000 步内从 1.0 线性降至 0.01。

| 奖励除数 | MA2C | PPO | DQL |
|---|---:|---:|---:|
| Grid | 2000 | 3000 | 3000 |
| Monaco | 200 | 1000 | 1000 |

训练裁剪阈值为 2；评估指标不缩放、不裁剪。MA2C 是邻域奖励，其他控制器是全局奖励，不强制使用相同除数。

每地图／算法四个开发候选，各 1000 回合：MA2C 学习率 `{5e-4,2.5e-4}` × entropy `{0.01,0.001}`；PPO `{3e-4,1.5e-4}` × 同样 entropy；DQL `{1e-4,3e-4}` × replay `{50000,20000}`。按开发验证 best 选择，同分优先表中起点。正式种子和最终测试不调参；五种续训共用参数。

## 4. 验证、best 和平台停止

控制器第 0 回合、每 100 个完整回合及停止时验证。每场景三个固定实现，每次 3600 秒，空网络开始。MA2C／PPO 采用固定随机策略种子，DQL greedy；交通 artifact 和 SUMO seeds 在方法间配对。验证使用独立模型与随机流，不影响训练参数、优化器、replay 或计数。

选模分数为场景等权的平均全网排队：

\[
J=\frac1G\sum_g\frac1R\sum_r\frac1{3600}\sum_{t=1}^{3600}\sum_{\ell\in L}q_\ell(t),\quad R=3.
\]

任何严格下降都保存 best；同分保留更早模型。吞吐、pending、teleports、碰撞和尾部指标只诊断，不参与排名。不完整 rollout 无有效分数。

| 阶段 | 最少回合 | 最大回合 |
|---|---:|---:|
| Initial | 1000 | 5000 |
| Offline WCE | 500 | 3000 |
| Continuation | 1000 | 5000 |

候选平台必须同时满足：达到最少回合；五个验证间隔（500 回合）没有实质改善；最近五分数极差 ≤ `max(mean,1)×2%`；拟合斜率绝对值 ≤ `max(mean,1)×0.5%`／验证间隔；当前距历史 best ≤ `max(best,1)×2%`。初始训练须至少发生一次相对 episode 0 的实质改善。

实质改善阈值 `max(anchor×0.01,0.1)`；累计超过阈值才更新 anchor 和 patience，但更小改善仍更新 best。

触发后对最近五候选及 best 补充七个独立固定实现，合计每场景十次，复核波动、趋势和当前与 best 差距；不改变常规 best 排名。

停止结果区分 `plateau_confirmed`、`max_budget_unconfirmed`、`no_learning`、`failed`。分段未完成为 `interrupted`，不是收敛。经验平台不等于理论最优；预算耗尽不能冒称收敛。

### WCE

冻结 best 父模型。Grid 保留 CNN-A2C，Monaco 保留 GCN-A2C；每 600 秒输出十一维需求权重，每完整回合更新一次。每 100 回合用十个固定种子组合验证完整 6600 秒平均排队，越高越好；平台确认增加二十个，共三十次。

固定的是外生随机流与需求约束，不是 WCE 自身生成的完整交通轨迹。随机流按区间、OD 和随机量类型索引。保留原学习率、奖励尺度、entropy；WCE advantage normalization 明确关闭。父模型通过哈希引用，不重复复制。正式 fixed／online 只使用已确认平台的同一 best WCE。

### 续训与公平性

五种方法同父模型分叉；random group 均匀抽独热组，DR 使用 `Dirichlet(1,…,1)`，fixed 冻结 WCE，online 每回合更新 WCE。DQL 保留 replay／target／优化器，但分叉时 epsilon 从 0.1 在 132,000 步内降至 0.01；恢复不能重置。

父模型作为续训 episode 0 的 best 候选；允许续训未改善。共同预算为 1000 回合／1,320,000 新学习步，冻结 `best_at_common_budget`，后续不可改写。Online 停止仅看外生验证，不看变化对手的训练 reward。完整恢复点保存同一时刻 controller 与 WCE，不能拼接不同时刻的各自 best。

## 5. 内存和数据减量

每秒仍测量队列。使用 float64 在线累加器计算和、平方和、峰值、车辆计数与时间完整性；保留当前车道向量和最长 600 秒全网队列环形缓冲。WCE 直接累积区间均值。episode 一条摘要，学习诊断按需求区间聚合。`batch_lengths` 改为直方图；只在测试保留完整序列。

DQL replay 用定长 NumPy 数组，容量按联合转移计。观测／reward float32，action 整数，terminal bool；不存不用的 policy/value 数据。分块序列化，避免完整缓冲深拷贝；禁止 float16 或有损压缩。

验证采用 inference-only 图，不创建优化器和 replay，DQL 只加载 online 网络。训练暂停期间在独立进程顺序验证；冻结父控制器同样使用推理模式。

| 输出模式 | 数据 |
|---|---|
| 正式训练 | episode 摘要、区间诊断、需求权重与哈希、验证历史 |
| 常规验证 | rollout 标量、场景分数、候选身份 |
| 最终评估 | 标量、每分钟曲线、预定义窗口的车道统计 |
| 审计 | 逐秒车道、控制记录和必要 SUMO 数据 |

训练默认关闭每回合 NPZ、逐秒 JSONL、每步奖励、每 agent 每更新梯度日志、自动图表和不必要 tripinfo。完整审计限集成检查及最终模型 Uniform／peak_1.5 第一条实现。需要 tripinfo 时在本地临时盘流式解析成摘要再删除 XML；失败保留有限日志尾部。

训练交通每 600 秒即时生成，只保存生成状态、权重、版本及哈希。固定评估交通按需生成一次并压缩共享，车辆引用 route table，避免重复边序列。种子必须配套生成器、地图、需求和交通哈希。评估写压缩分片；CSV、图和 TensorBoard 离线生成，不重复保存等价原始数据。不同软件版本不保证逐位一致。

## 6. Checkpoint 与归档

完整 checkpoint 含参数、优化器、replay／pending、RNG、调度、计数、验证状态和在线 WCE；轻量 snapshot 只有推理参数及身份，不能恢复训练。

每十回合保存恢复点，保留最新两个；当前 best 保留完整状态；最近五验证候选保留轻量快照；共同预算 best 保留推理快照；最终保留完整状态；下游引用的父模型与 WCE 固定保留。角色相同复用文件；仅清理无引用且有有效替代的中间产物。持续保存 best 不代表永久保存所有曾经的 best replay。

写临时目录→状态与参数→哈希→完成 manifest→目标端校验→更新引用→清理。跨文件系统先复制目标临时目录，再在目标文件系统原子发布。失败不得删除上一有效恢复点。清理范围只包含新项目托管数据。

## 7. Hábrók

本地 Git 保存代码；`/scratch/<account>/cbwce` 保存活跃输入、checkpoint 和结果；`$TMPDIR` 保存解压代码、SUMO 临时文件和缓存；`/projects/<account>/cbwce` 保存已校验归档；RDMS／独立备份作长期保存。

`/scratch` 无备份，`/projects` 不供计算节点直接使用，计算作业先持久化到 scratch，再由登录端归档。账户真实容量和 inode 配额使用 `hbquota`，不假设文档示例即账户配额。

Apptainer 环境：Python 3.10、TensorFlow CPU 2.15.1（compat.v1）、NumPy 1.26.4、SUMO／TraCI／sumolib 1.26.0。构建时锁定 Python 补丁、基础镜像 digest、所有依赖和 SIF 哈希。正式运行必须同镜像，重新验证，不能宣称与旧 TF1.12 运行等价。任务内不联网安装依赖。

CPU 试验起点：regular、1 task、2 CPU、8 GiB、数值库单线程、24 小时分段、数组并发 4。8 GiB 未经测量不能视为保证；测峰值 RSS、回合时间、写出耗时后，正式内存至少实测峰值×1.3 并上取整。关闭 oneDNN 优化固定数值运行设置。

Slurm 信号必须转发给 worker；提前停止启动新回合，剩余时间不足则丢弃未完成回合。恢复不清零预算、patience 或探索。周期性写回 scratch，不能仅依赖退出拷贝。下游同时检查作业结果和 `plateau_confirmed`；正常分段结束但未完成只续当前阶段。归档校验成功后才可清理非活跃 scratch 副本。

官方资料：[存储](https://wiki.hpc.rug.nl/habrok/migration/migration_habrok)、[配额](https://wiki.hpc.rug.nl/habrok/data_management/quota)、[Apptainer](https://wiki.hpc.rug.nl/habrok/examples/apptainer)、[Slurm](https://wiki.hpc.rug.nl/habrok/job_management/scheduling_system)。

## 8. 验收和交付

验收包含：脱离原目录运行两图；两图×三算法集成；在线汇总与审计重算一致；精简日志不影响学习；replay 和监测有界；验证不改变训练；完整恢复与轻量拒绝恢复；强制中断／复制损坏／重复提交保护；引用保护清理；记录 RSS、字节数、文件数和写入时间；双语一致；未冻结 manifest 拒绝正式运行。

交付独立源码、双语方案、环境锁与容器构建定义、输入发布包、Slurm 模板、验证证据、资源报告和归档工具。实际模型与结果归档由随后被授权且通过门禁的运行产生，不能预先伪造。
