# v8.1 实施与验证记录 / Implementation and verification

更新日期 / Updated: 2026-10-27. 本文件描述实际完成情况；双语 revision plan 描述完整实验合同。
This file records completed engineering and its limits; the bilingual plans define the experimental contract.

## 已实现 / Implemented

- 独立项目、独立 Git、两地图必要输入和逐文件来源哈希；不需要旧工作区。
  Standalone source, independent Git history, minimal network inputs and extraction provenance.
- MA2C/PPO recurrent 控制器、有效样本 advantage normalization、可配置邻域奖励；DQL 为独立 128/64 MLP、target、Huber、固定数组 replay。
  MA2C/PPO recurrent controllers with valid-sample advantage normalization and configured neighborhood rewards; DQL uses independent 128/64 MLPs, targets, Huber loss and fixed-array replay.
- 父模型、离线 WCE、五种续训共用验证选模；best、累计改善锚点、平台确认、共同预算和停止状态明确分离。
  Validation selection across parent/WCE/five continuation branches, with separate best, meaningful-improvement, confirmation and common-budget state.
- 独立进程只加载推理状态；常规记录为压缩摘要；审计包含逐秒排队、动作和原始奖励。
  Separate inference-only evaluation workers, compressed routine summaries, and optional queue/action/raw-reward audits.
- 完整恢复、原子发布、引用保护清理、下游父模型固定引用、作业排他锁和恢复分支日志。
  Full recovery, atomic checkpoint publication, reference-aware retention, pinned ancestry, exclusive run ownership and attempt journals.
- Hábrók Slurm 数组模板、信号转发、镜像哈希校验、Slurm/application 双重下游检查、登录端校验归档。
  Hábrók job-array templates, signal forwarding, image verification, scheduler/application prerequisites and verified login-side archival.
- 哈希锁定 Python 依赖、Apptainer 构建定义、白名单代码包、输入包与归档工具。
  Hash-locked Python dependencies, Apptainer definition, whitelisted source releases, input releases and archive tools.

## 实测证据 / Measured evidence

规范记录在 [`provenance/verification`](../provenance/verification)。
Canonical evidence is retained in that directory, including exact runtime and source fingerprint.

| 检查 / Check | 结果 / Result |
|---|---|
| Unit tests | 11 passed: plateau, replay, target updates, exact next-update recovery, masking, metrics, integrity and retention |
| Grid/Monaco × MA2C/DQL/PPO | All six real-SUMO workflow pilots passed; 6 parents, 6 offline WCEs and 30 continuation branches |
| Fresh-process continuation resume | All six cases match uninterrupted controller and online-WCE tensors within recorded tolerances |
| Audit/compact training | All six cases produce exactly equal model/optimizer tensors and queue summaries |
| Forced interruption | Grid DQL killed with SIGKILL mid-episode; previous checkpoint preserved; recovery tensors exactly match uninterrupted run |
| Duplicate submission | Exclusive run ownership correctly rejects the second process |
| Relocation | Both networks run from `/tmp`; Python opens under the original workspace denied; no modules loaded from it |
| Complete evaluation horizon | Both maps: Uniform and peak_1.5, 3600 seconds each, one realization plus audit rerun; equal results |
| Publication preflight | Both maps reject missing and unfrozen validation manifests |
| Static checks | Python compilation and Bash syntax checks passed |

训练流程试验采用每需求块 10 秒、每回合 110 秒、每阶段 1 回合、每场景 1 次验证。DQL 试验 replay=128、warm-up=8、batch=8、target interval=20，用于实际触发更新和恢复验证。它们不是调参、性能比较或稳定收敛证据。只有评估专项检查使用完整 3600 秒。

Workflow pilots use 10-second demand blocks, 110-second episodes, one episode per stage and one validation realization. DQL pilot replay=128, warm-up=8, batch=8 and target interval=20 deliberately exercise updates and restoration. These are correctness tests, not tuning, performance comparisons or convergence evidence. The separate evaluation checks use the complete 3600-second horizon.

| 网络/算法 / Network/controller | Pilot peak RSS, GiB | Maximum checkpoint time, s |
|---|---:|---:|
| Grid MA2C | 1.487 | 0.518 |
| Grid DQL | 0.698 | 0.167 |
| Grid PPO | 1.392 | 0.414 |
| Monaco MA2C | 1.503 | 0.509 |
| Monaco DQL | 0.711 | 0.164 |
| Monaco PPO | 1.380 | 0.429 |

RSS 来自训练进程 `ru_maxrss`，不是同时运行的训练、验证和 SUMO 进程的总和；这些数字不能直接用于 Slurm 总内存定额。完整资源记录还包含文件数、字节数和耗时。正式资源申请须在目标镜像中测量总作业 RSS，包括填满的 replay，再乘至少 1.3。

RSS is the trainer's `ru_maxrss`, not aggregate concurrent trainer/validator/SUMO memory. Do not use it directly as a Slurm job limit. Resource records also contain file counts, bytes and durations. Measure aggregate job memory inside the target image with filled replay before applying the ≥1.3 resource margin.

固定容量 50,000 联合转移的 DQL 数组理论大小：Grid 323,250,000 字节；Monaco 198,600,000 字节，不包含 TensorFlow 和 SUMO。6600 秒合成队列缓冲测量从旧列表约 11–14 MB 降至约 45–59 KB（约 99.6%），不等于整个训练内存下降 99.6%。可用 `scripts/measure_metrics.py` 重现。

At 50,000 joint transitions, replay arrays require 323,250,000 bytes for Grid and 198,600,000 for Monaco, excluding TensorFlow/SUMO. A synthetic 6600-second queue-buffer comparison reduces retained storage from approximately 11–14 MB to 45–59 KB (~99.6%); this is not a 99.6% total-training-memory reduction. Reproduce with `scripts/measure_metrics.py`.

## 正式运行前仍需完成 / Outstanding publication gates

1. 用户确定并冻结两地图的验证场景 manifest。本次没有替用户选择组数或正式场景。
   The user must supply and freeze both validation designs; no publication scenario count/composition has been invented.
2. 在具备 Apptainer 的构建机选定 digest-pinned Python 3.10 基础镜像，构建 SIF，生成 `packaging/runtime.lock.json`，在该镜像中重新运行六案例 verifier。本机没有可用 Apptainer，Docker daemon 也不可访问，因此没有生成或验证 SIF。
   Build a SIF from a digest-pinned Python 3.10 base on an Apptainer-capable host, generate the runtime lock and rerun the six-case verifier inside it. Apptainer is unavailable here and the Docker daemon is inaccessible; no SIF has been built or verified.
3. 本地测试使用 Python 3.10、TF 2.15.1、NumPy 1.26.4 和已安装的 SUMO 开发构建；SciPy 是主机旧版本并产生兼容性警告。它不是计划要求的完整容器基线。依赖锁已生成，环境完全匹配检查仍必须在容器执行。
   Local tests use Python 3.10, TF 2.15.1, NumPy 1.26.4 and the installed SUMO development build. The host's older SciPy emits a compatibility warning. This is not the fully locked publication runtime.
4. 目标 Hábrók 上完成完整 6600 秒训练、填满 replay 的总作业资源试验及实际 Slurm 信号/存储验证。现有测量不能证明集群上的 8 GiB 足够。
   Complete full-6600-second training/resource pilots with filled replay, aggregate job memory, real Slurm signals and cluster storage on Hábrók. Current measurements do not certify the 8 GiB request.
5. 开发调参结束后，将六组胜出配置纳入版本、重新验证，再运行正式 18 父模型、18 WCE、90 续训与最终测试。本次没有启动正式训练，也没有声称已有收敛模型或正式模型归档。
   Version the six chosen tuning configurations, revalidate, then run the publication matrix and final evaluation. No publication training, converged policy or publication-model archive is claimed.

Hábrók 模板依据官方 [存储说明](https://wiki.hpc.rug.nl/habrok/migration/migration_habrok)、[配额](https://wiki.hpc.rug.nl/habrok/data_management/quota)、[Apptainer](https://wiki.hpc.rug.nl/habrok/examples/apptainer) 和 [Slurm 说明](https://wiki.hpc.rug.nl/habrok/job_management/scheduling_system) 核对。账户、配额和资源实测需在目标集群确认。
Templates were checked against those official sources; account-specific quotas and resources still require target-cluster verification.
