# CB-WCE 正式测试执行方案

本方案把 evaluation workbook 转换为分阶段命令。所有危险步骤都需要明确确认变量；当前不会自动执行任何测试。

## 测试规模

正式评估包括 2 个路网、4 个控制器、5 种方法、23 个场景和每场景 10 个配对 rollout，总计 40 个 suite、9,200 个 rollout。每个 suite 包含 230 个 rollout。

主指标是 network-total mean queue，单位为 vehicles，越低越好。

## v7 实现状态与剩余步骤

1. 已完成：protocol 已升级为 v7。
2. 已完成：三个 temporal 场景从 11 个 seen profiles 中按冻结 seed 随机选择，相邻区块不得重复。
3. 已完成：场景元数据包含 temporal_policy: seeded_random_seen_profiles 和 protocol_version: 7。
4. 已完成：switch_300 保持 12 个 300 秒区块，作为快速切换压力测试。
5. 待执行：针对当前 v7 源码重新生成 4-worker verification gate；此前 gate 已因 protocol 源码变化而失效。
6. 待执行：在独立 protocol_v7 目录物化完整 v7 demand suite，其中包括两个路网、三个 temporal 场景、十个 rollout 的 60 个 temporal artifacts；不得覆盖或混用 v6 输入。
7. 正式运行前保证至少 60 GiB 空间，并预计完成后仍保留至少 20 GiB。

## 安全入口

以下命令不会启动正式评估：

~~~bash
./docs/evaluation_workbook/generated/publication_workflow.sh --plan
./docs/evaluation_workbook/generated/publication_workflow.sh --status
./docs/evaluation_workbook/generated/publication_workflow.sh --list
~~~

## Phase A：v7 verification gate

这一步会运行 deterministic tests 和 8 个 SUMO pilot cases，但不会运行 9,200 条正式 rollout。

~~~bash
CONFIRM_VERIFICATION=RUN_8_PILOTS \
VERIFY_WORKERS=4 \
./docs/evaluation_workbook/generated/publication_workflow.sh --verify
~~~

也可以指定输出目录：

~~~bash
CONFIRM_VERIFICATION=RUN_8_PILOTS \
./docs/evaluation_workbook/generated/publication_workflow.sh --verify \
  runs_eval/revised/verification/protocol_v7_YYYYMMDDTHHMMSSZ
~~~

成功后指定新 gate：

~~~bash
export CBWCE_GATE=/absolute/path/to/protocol_v7_YYYYMMDDTHHMMSSZ/gate.json
~~~

## Phase B：物化 v7 demand artifacts

~~~bash
CONFIRM_MATERIALIZE=BUILD_V7_ARTIFACTS \
CBWCE_GATE="$CBWCE_GATE" \
./docs/evaluation_workbook/generated/publication_workflow.sh --materialize
~~~

人工检查：

- 三个 temporal scenario 均包含 temporal_policy；
- block profile 全部来自对应路网的 11 个 seen profiles；
- 相邻 block 不重复；
- switch_300 有 12 个区块；
- 相同 network/scenario/rollout 的 artifact 跨方法和控制器复用；
- 60 个 temporal artifacts 使用新的 scenario/artifact hash；
- v7 必须使用新的版本化目录；现有 immutable preparation 若检测到同路径内容变化，应保持报错而不是覆盖；
- v6 artifact 不得进入 v7 release。

## Phase C：正式运行前检查

~~~bash
CBWCE_GATE="$CBWCE_GATE" \
./docs/evaluation_workbook/generated/publication_workflow.sh --preflight
~~~

检查内容包括 protocol v7、temporal contract、40 个 checkpoint、输出目录、磁盘、gate 和运行环境。

保存 40 个命令供第二位 reviewer 复核：

~~~bash
./docs/evaluation_workbook/generated/publication_workflow.sh --list \
  > /tmp/cb_wce_publication_commands.txt
~~~

Reviewer 应逐项核对 network、controller、method、checkpoint、hash、gate 和 output path。

## Phase D：validation-only canary

先用 validation split 验证完整链路，不要提前使用冻结 test split：

~~~bash
CONFIRM_CANARY=RUN_VALIDATION_ONLY \
CBWCE_GATE="$CBWCE_GATE" \
./docs/evaluation_workbook/generated/publication_workflow.sh \
  --canary grid ia2c online_wce
~~~

通过条件：

- 6 个 validation scenarios × 10 rollouts = 60；
- 每条 rollout 有 3,600 个样本；
- 无 NaN、负 queue 或 seed/hash 不一致；
- integrated_queue ≈ 3600 × mean_queue；
- scheduled = inserted + pending；
- completed ≤ inserted；
- completed-trip denominator = completed。

Canary 仅验证工程链路，不作为论文结果。

## Phase E：正式运行

只有前述阶段全部通过并完成独立复核后才执行：

~~~bash
CONFIRM_PUBLICATION=RUN_9200 \
CBWCE_GATE="$CBWCE_GATE" \
CBWCE_EVALUATION_WORKERS=4 \
./docs/evaluation_workbook/generated/publication_workflow.sh --execute
~~~

launcher 默认并行运行 4 个 suite，每个 suite 内部仍顺序执行 230 个 rollout，并拒绝覆盖已有目录。每个 worker 固定使用单个 OpenBLAS/OMP 线程。

当前并行 scheduler 不支持安全的 suite 内自动续跑，也不会自动重试失败 suite。失败后停止派发新 suite，但已启动的其他 suite 会完成。中途失败时不要删除或覆盖目录，也不要把 retry 视为额外随机重复。应先保存现场并设计显式 attempt/resume 方案。

## 监控命令

~~~bash
watch -n 60 'df -h /home/sdc_joran/Journal/deeprl_signal_control'
watch -n 60 "find runs_eval/revised/publication_seed101_v1 -name rollout_summary.json | wc -l"
find runs_eval/revised/publication_seed101_v1 -name suite_result.json -print
./docs/evaluation_workbook/generated/publication_workflow.sh --dashboard
~~~

## Phase F：报告与审计

当前已有 experiment report，但 workbook 设计的 validate-report 和 export-site 尚未实现。

~~~bash
CONFIRM_REPORT=BUILD_REPORT \
./docs/evaluation_workbook/generated/publication_workflow.sh --report
~~~

只读审计：

~~~bash
./docs/evaluation_workbook/generated/publication_workflow.sh --audit
~~~

或者直接调用：

~~~bash
conda run -n deeprlsc python \
  docs/evaluation_workbook/validate_publication_release.py \
  --raw-root runs_eval/revised/publication_seed101_v1 \
  --report output_result/revised/publication_seed101_v1
~~~

发布完整性要求：

- accepted = 9,200；
- rejected、missing、duplicate、extra 均为 0；
- 每个 network/controller/method 有 230 条；
- 每场景有十个 arrival seeds；
- arrival 51001–51010 与 SUMO 61001–61010 精确配对；
- pilot=false、status=complete；
- horizon=sample_count=3,600；
- queue 与车辆记账恒等式通过；
- report 中 publication_complete=true。

辅助审计脚本没有替代尚未实现的 NPZ/JSONL 对齐、完整 schema、隐私扫描和 site-export 验证。

## 失败与恢复原则

- 不覆盖已有 output。
- 不删除 failed 或 interrupted attempt。
- 不把 retry 计为新的随机重复。
- artifact、seed、checkpoint 或 scenario hash 改变时创建新 protocol/release ID。
- 报告只接受 selection 指定的成功 attempt。
- test split 冻结后只运行一次正式评估。

## 推荐顺序

~~~text
实现 v7
→ --verify
→ export CBWCE_GATE=...
→ --materialize
→ --preflight
→ --list + 独立复核
→ --canary（validation only）
→ --execute
→ 监控 9,200 rollouts
→ --report
→ --audit
→ 完整 release/privacy/site validation
~~~
