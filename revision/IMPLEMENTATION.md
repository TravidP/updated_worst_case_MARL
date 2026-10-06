# Implementation and verification report

> Historical implementation report: the divided-by-100 controller reward below is superseded by protocol 5. Current behavior and monitoring are documented in [the monitoring guide](../docs/TRAINING_MONITORING.md). The old evidence remains historical.

> Active protocol update (15 September 2026): one training seed (`101`), 8 parents, 8 WCE runs, 40 continuations and 9,200 evaluations. Evidence below describes historical verification, including checks of the earlier statistical design; it does not certify the updated code/protocol.

> Historical pre-integration report. The source subsequently moved into `agents/`, `envs/`, and `experiments/`; the old gate below no longer certifies current code. See the root project guide for the integrated workflow. This report and its original evidence are preserved for provenance.

**Date:** 14 September 2026  
**Scope:** C01–C10 in the separate `revision/` experiment path  
**Status:** Deterministic checks and all eight integration pilot cases passed. Publication training has not started.

Use the [runner instructions](README.md) for commands and the [English](../reviewer_revision_plan.md) or [Chinese](../reviewer_revision_plan_zh.md) guide for the publication protocol. Original source files, configurations, datasets, checkpoints, and historical results remain unchanged. These corrections apply to the new runner; invoking a legacy entrypoint does not activate them.

## Implemented corrections and evidence

| ID | Implemented behavior | Verification |
|---|---|---|
| C01 | Evaluation mode is set before reset; actual SUMO startup commands and seeds are recorded. | Requested seeds match startup seeds; identical seeded evaluations reproduce lane queues. |
| C02 | Independent recoverable random streams; complete demand artifacts include routes and speed factors. | The scheduled-demand hash stays identical across methods, controller families, and policy-seed changes within each network. Actual insertion is recorded separately. |
| C03 | One family-specific controller reward path, divided by 100 once. | Synthetic shared/MA2C arithmetic and every pilot's recorded controller rewards agree with lane measurements. |
| C04 | Shared inference → fingerprint update → environment step ordering; episode resets clear inference state. | Fingerprint/reset checks pass; frozen-controller parameters remain identical during WCE training and evaluation. |
| C05 | IQL learns every twenty new learning transitions, with ten minibatch updates per agent; frozen simulation does not populate replay. | A 200-transition fixture yields ten backward calls and 100 updates per agent. All five methods have identical counters in both networks. |
| C06 | The shared IQL adapter uses IQL policy/replay interfaces and greedy frozen/evaluation inference. | Both Monaco IQL frozen-WCE and continuation paths, restoration, and full evaluations execute successfully. |
| C07 | Uncapped lane halting counts are collected every second, including yellow time, with a unique monitored lane set. | Constant queues 12 and 3 give total 15, shared reward −0.15, MA2C rewards −0.147/−0.138, and WCE reward +0.15. Time-varying and pilot measurements also reconcile. |
| C08 | Savers include optimizer variables; bundles include Python buffers, schedules, counters, random state, and parent identities. Partial batches use masks and correct bootstrapping. | Fresh-process next-update recovery passes for all controller families; resumed online episodes reproduce tensors. Missing/incompatible checkpoints fail. Seven successive controller/WCE bundles remain intact and the oldest restores. |
| C09 | Full timestamp/horizon validation; no padding into valid results. | Eight deliberately interrupted evaluations are excluded. All accepted evaluations contain 3,600 seconds. |
| C10 | Exclusive run directories, immutable manifests, hashed inputs/checkpoints, and separate attempt records. | Duplicate outputs fail; accepted records resolve to their actual inputs. Canceled and preliminary attempts are excluded from the accepted gate. |

The [deterministic test log](verification/final_safeguards/unit.log) reports **10 tests passed**. The original fixed-length recurrent computation and its compact loop implementation agree in outputs and gradients; batched inference also agrees with per-agent inference.

## Integration matrix

Runtime: `deeprlsc`, Python 3.6.13, TensorFlow 1.12.0, NumPy 1.19.5, and SUMO `1_26_0+0455-77b9dbc222e`. The host has 20 logical CPUs and approximately 62 GiB RAM. TensorFlow and BLAS threads were limited; up to eight pilot workers ran concurrently.

Every case used pilot seed `9001`, a 160-learning-step parent, two frozen-controller WCE episodes, and two continuation episodes per method. Each method therefore finished at **2,800 learning steps**, comprising 160 + 2,640. These are short correctness budgets, not the publication budgets.

| Network | Controller | Backward calls, including parent | Minibatch updates per agent, including parent | Result |
|---|---|---:|---:|---|
| Grid | IA2C | 24 | 24 | Passed |
| Grid | MA2C | 24 | 24 | Passed |
| Grid | IQL-LR | 140 | 1,400 | Passed |
| Grid | PPO | 24 | 96 | Passed |
| Monaco | IA2C | 70 | 70 | Passed |
| Monaco | MA2C | 70 | 70 | Passed |
| Monaco | IQL-LR | 140 | 1,400 | Passed |
| Monaco | PPO | 24 | 96 | Passed |

Within each row, all five methods have exactly the same counts. Differences between rows follow the prescribed controller/network batch sizes and PPO epochs.

The accepted primary matrix contains:

- Eight parent pilots and eight offline WCE pilots.
- Forty continuation runs and eight resumed-episode checks.
- Sixty-four complete evaluations: five methods plus exact-repeat, changed-SUMO-seed, and changed-policy-seed checks per case.
- Eight deliberate incomplete-evaluation attempts and eight direct speed-subscription checks.

Fixed WCE model parameters stayed unchanged in every case. Online WCE model parameters changed, with two offline and two online updates. The frozen parent stayed unchanged. Resumed controller and WCE tensors matched the uninterrupted runs. Mixture weights may vary even when WCE model parameters are frozen.

Evidence is recorded in the Grid case directories under [the Grid verification run](verification/corrections_20260914_v2/) and Monaco case directories under [the Monaco verification run](verification/monaco_parallel_20260914/). Each accepted case has `checks.json`, run manifests, checkpoints, measurements, and summaries. The combined gate identifies the exact eight accepted cases.

## Final safeguards and source provenance

The full pilot matrix used one consistent source snapshot. After it completed, a seven-save reproduction exposed TensorFlow's default retention deleting files from older checkpoint bundles. Both savers now disable automatic deletion. Two small runner changes also include preflight work in stage timing and correctly label evaluation manifests with the 3,600-second horizon.

These final changes were verified with the complete ten-test suite and one additional full Grid IQL evaluation. Its lane queues exactly match the corresponding pilot evaluation; its manifest horizon matches the result. There are therefore **65 accepted full evaluations including this final repeat**.

The [post-pilot validation record](verification/final_safeguards/post_pilot_validation.json) contains the exact before/after source hashes and diffs. The full eight-case matrix was not rerun after those narrowly scoped changes; learning equations and budgets did not change. The final unit suite covers checkpoint retention, and the additional evaluation checks the changed runner path.

The [accepted verification gate](verification/corrections_20260914_gate.json) records final source hashes, the pilot source snapshot, input hashes, evidence hashes, and the final safeguard validation. `require_gate` accepts it against the current files. Later changes to covered source or experiment inputs invalidate this gate and require renewed verification.

Monaco pilots were run in a parallel shard while Grid pilots finished. Redundant Monaco jobs subsequently queued by the first coordinator were canceled and recorded separately. Earlier debugging and canceled attempts remain on disk but are not accepted cases. Because concurrency changed during verification, these pilot timings should not be used as the paper's method-cost comparison.

## Remaining publication work

The five-method matrix and publication budgets remain unchanged: 8 parents, 8 offline WCE runs, 40 continuations, and 9,200 final evaluation rollouts. None of that publication matrix has been launched.

The corrected runner supports materialized Uniform demand and explicit mixture/switching schedules. The complete twelve-scenario manuscript generator, final statistical tables, and peak-demand heatmap campaign remain tasks for the publication workflow. Passing these checks establishes the tested experimental mechanics; it does not establish comparative performance or guarantee behavior for every future input.

Verification artifacts are retained locally and ignored by Git. This report and the implementation source are available for version control; no commit or push was performed by this task.
