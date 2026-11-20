# Revision Plan v8.4: Single Training Seed, Standalone Project and Local Multicore Execution

Updated: 2026-11-18. Synchronized with the Chinese plan and eight task documents. Machine protocol version remains 8.1; separate schemas/source/input hashes determine compatibility.

This document specifies the protocol and acceptance criteria. Code availability is not acceptance evidence. Actual verification is recorded separately in `implementation_status.md`; publication training, container construction and deployment are not claimed complete.

Validation update, 2026-11-18: routine controller validation is **one fixed realization of eleven frozen DR blocks, 600 seconds each, 6600 seconds total**. Code now enforces the separate `dr_sequence_v1` validation schema; final tests remain 3600 seconds. Historical acceptance results do not cover this new schema. The bilingual plan and task documents now also specify local execution, single-seed campaigns and final evaluation.

## 1. Objectives and experimental matrix

Use the compact `cbwce_standalone` project on a local Linux CPU workstation, retaining MA2C, DQL and PPO with validation selection, patience stopping, bounded outputs and resumable independent processes. Cluster deployment is deferred.

| Item | Protocol |
|---|---|
| Networks | 5×5 Grid and the Monaco subnet used by the current revised runner; do not substitute the repaired network |
| Publication seeds | Default 101 only; development 9001. Add seeds through a new explicit campaign |
| Training episode | 6600 seconds, 1320 joint network decisions |
| Control/demand interval | 5/600 seconds |
| Initial training | Existing fixed sequence of eleven demand profiles |
| Continuations | `baseline`, `random_group`, `domain_randomization`, `fixed_wce`, `online_wce` |
| Selection | Minimize mean queue for controllers; maximize frozen-parent mean queue for WCE |
| Evaluation checkpoint | One per continuation, the lowest validation mean queue through stopping (including episode 0) |

**Controller validation uses 11 pregenerated, frozen DR demand groups per network, in a fixed sequence lasting 6600 seconds, with one fixed realization.** Each group lasts the confirmed 600 seconds. Exact weights, design seeds and random streams still require an explicit frozen manifest; the development manifest is not a substitute. Separate development, selection and final-test identities and random streams.

Publication totals: 6 parents, 6 offline WCE runs and 30 continuations. Final evaluation retains eleven seen and twelve generalization scenarios, ten realizations each, each lasting 3600 seconds: 690 scenario tasks and 6,900 main rollouts in total. The 6600-second change applies to controller validation, not automatically to final testing.

## 2. Standalone preservation and layout

The original workspace occupied approximately 164 GB when inspected: 64 GB evaluation, 42 GB coevolution, 24 GB evaluation figures and 18 GB Git history. Do not copy the entire workspace.

Create new Git history and extract a whitelist of runtime code, both networks, eleven demand inputs per network, configurations, tests, documentation, Slurm templates, licenses and authorship. Record the original commit, working-tree changes and every extracted file hash. A committed-tree export alone would omit relevant local modifications.

Copy only necessary Monaco network/detector dependencies, not historical simulation directories. Preserve the original project and results.

```text
cbwce_standalone/
  agents/ envs/ experiments/ config/ assets/ tests/
  scripts/slurm/ docs/ packaging/ provenance/ main.py
cbwce_data/       # External shared versioned inputs
cbwce_runs/       # External outputs
cbwce_archive/    # External verified retained artifacts
```

Provide `CBWCE_ASSETS_ROOT`, `CBWCE_DATA_ROOT`, `CBWCE_RUN_ROOT` and `CBWCE_CACHE_ROOT`. No original absolute paths, external symlinks or imports from the old workspace. Content hashes, protocol versions and run identities determine compatibility; paths only locate artifacts.

Produce separate code, input and run releases with inventories and SHA-256 checksums. Reference shared networks, traffic and frozen models instead of duplicating them.

## 3. Controllers and parameters

Remove IA2C from the new matrix. Preserve MA2C neighbor observations, fingerprints, neighborhood rewards and LSTM. Normalize advantages over valid samples, excluding padding. Read the neighborhood coefficient from configuration, default 0.9. Inspect entropy/policy gradients, effective value-loss coefficients, reward clipping and gradient clipping. Scale rewards once at the learner boundary. Retain n-step returns without introducing GAE; MA2C updates once per rollout.

Replace linear `iqll` with independent DQL networks: `wave 128 → LSTM 64 → Q`. Use distinct online/target networks, uniform replay, Huber loss and epsilon-greedy exploration; greedy evaluation. Do not add Double DQN, dueling heads or prioritized replay.

| Parameter | MA2C | PPO | DQL |
|---|---:|---:|---:|
| Learning rate | 5e-4 | 3e-4 | 1e-4 |
| Discount | 0.99 | 0.99 | 0.99 |
| Optimizer | RMSProp | Adam | Adam |
| LSTM | 64 | 64 | 64 (16-step window) |
| Advantage normalization | On | On | Not applicable |
| Contiguous rollout length | 120 joint control steps | 120 joint control steps | Not applicable; append transitions to replay |
| Samples per intersection per optimization | 120 valid samples from the current rollout; final segment may be shorter | The same 120 valid rollout samples; final segment may be shorter | 64 uniformly sampled replay endpoints, each with up to 16 history steps |
| Updates | Once/rollout | Four epochs on the same rollout; each currently uses the entire valid sequence | After warm-up, one optimization per intersection every four joint control steps |
| Entropy | 0.01 | 0.01 | Not applicable |
| PPO clip | — | 0.2 | — |
| Replay | — | — | 50,000 joint transitions |
| Warm-up | — | — | 5000 steps |
| Target synchronization | — | — | Every 1000 learning steps |
| Gradient-norm limit | 40 | 40 | 10 |

Preserve existing MA2C/PPO encoder widths. Initial DQL epsilon decreases linearly from 1.0 to 0.01 over 500,000 steps.

### 3.1 Which settings should match

**Do not force identical rollout/batch sizes, update frequencies or gradient-clipping thresholds.** Retain `120 / 120 / 64` and `40 / 40 / 10` as development starting points. Numerical differences alone do not imply unfairness, and these defaults are not established optima.

- MA2C/PPO's 120 denotes a newly collected contiguous sequence, corresponding to 600 seconds at five-second control intervals. DQL's 64 denotes historical replay samples, not 320 contiguous seconds or 64 episodes. Padding is excluded from valid MA2C/PPO samples.
- MA2C uses the current rollout once. PPO's clipped objective supports repeated optimization of the same data; this implementation uses four epochs. DQL uses replay and a target network, with a different update schedule. These mechanisms have algorithmic support, but `120`, `64` and `4` are project choices, not universally optimal settings prescribed by the papers. [PPO paper](https://arxiv.org/abs/1707.06347), [DQN paper](https://deepmind-media.storage.googleapis.com/dqn/DQNNaturePaper.pdf).
- A learning step means one five-second joint network control step, without multiplying by intersections or confusing it with an optimizer update. After warm-up, a complete 1320-step episode produces 11 MA2C, 44 PPO or 330 DQL optimizations per intersection. With 64 replay samples per update, DQL draws approximately 16 samples per newly collected step; this does not mean each transition is replayed exactly 16 times.
- The gradient-norm limit controls stability. MA2C/PPO clip combined actor/critic gradients per intersection; DQL clips online-Q gradients per intersection, without optimizing the target network. Different objectives, parameter counts, reward scales and optimizers mean identical thresholds do not imply identical parameter changes. Retain `40 / 40 / 10`; record unclipped norms, gradient-clipping frequency, scaling factors and losses during development. Distinguish PPO's policy-ratio clip fraction from gradient-clipping frequency. Diagnose scaling and numerical problems before preregistering any additional sensitivity study.

Match simulator/action interfaces within each network, environment-interaction budgets, validation traffic/realizations, tuning-candidate counts and evaluation rules. Within a controller, all five continuation methods use identical networks and optimization settings. Report per-intersection/network parameter counts, optimizer updates, wall-clock, aggregate job RSS and validation cost. Actual interaction counts vary with stopping. Report these and computation costs separately. Retain the existing four development candidates per network/controller rather than silently expanding the search.

### 3.2 Must the neural networks match?

DQL now uses the same wave encoder widths and LSTM gate equations as PPO and MA2C. MA2C/PPO retain separate actor and critic networks; DQL retains separate online and target Q networks.

| Controller | Current per-intersection network | Observation and memory |
|---|---|---|
| MA2C | Wave 128; fingerprint 64; LSTM 64; separate policy/V outputs | Neighbor fingerprints and recurrent state across the episode |
| PPO | Wave 128; LSTM 64; separate policy/V outputs | No fingerprints; recurrent state across the episode |
| DQL | Wave 128; LSTM 64; Q per action | No fingerprints; bounded 16-step history |

DQL samples 64 transition endpoints uniformly and reconstructs each contiguous history from fixed-capacity replay. Histories stop at terminal boundaries or the oldest retained transition. Left padding is masked. Online and target windows each start from zero state, matching bounded-window inference; the target window shifts the history and appends the next observation. Only the endpoint receives a Huber TD loss, with gradients through the valid history (no separate detached burn-in). `sequence_length=16` is a configurable project choice, not a tuned optimum. Full checkpoints preserve inference histories and replay; architecture identity rejects old feedforward checkpoints.

Report online trainable parameters separately from storage including the target network. Shared LSTM widths do not make the systems identical: MA2C has fingerprints, and DQL has finite history and replay while MA2C/PPO carry episode state. Algorithm-only conclusions still require matched-input and memory ablations.

Both Grid and Monaco use **wave-only traffic observations** for MA2C, DQL, PPO and WCE. Controllers retain local/neighbor wave inputs and MA2C retains neighbor fingerprints. WCE concatenates only padded per-node wave vectors: Grid CNN input is 25 × 6 = 150 (6 channels, previously 300 / 12 channels); Monaco GCN input is 28 × 11 = 308 (each node padded to 11 monitored lanes). Wave retains the existing measurement semantics: detector-area vehicle counts in Grid and lane vehicle counts in Monaco, divided by `norm_wave` and clipped by `clip_wave`. Waiting time is not an input; queue-based rewards are unchanged. The six controller INIs enforce `observation = wave`; unused `num_ft`, `norm_wait`, `clip_wait` and `fast_wait_metric` settings have been removed. New observation signatures reject checkpoints from the previous observation schema; restart training and regenerate dependent WCE checkpoints.

| Reward divisor | MA2C | PPO | DQL |
|---|---:|---:|---:|
| Grid | 2000 | 3000 | 3000 |
| Monaco | 200 | 1000 | 1000 |

Learner clipping threshold: 2. Evaluation is unscaled and unclipped. MA2C uses neighborhood reward while the other controllers use global reward, so identical divisors are not required.

Four development candidates per network/controller, each for 1000 episodes: MA2C rates `{5e-4,2.5e-4}` × entropy `{0.01,0.001}`; PPO rates `{3e-4,1.5e-4}` × the same entropy values; DQL rates `{1e-4,3e-4}` × replay `{50000,20000}`. Select development-validation best, breaking ties toward table defaults. Publication seeds and final tests do not tune parameters. All continuation methods share controller settings.

## 4. Validation, best and plateau stopping

### 4.1 Fixed DR sequence and one realization

Initial training and all five continuations validate at episode zero, every 100 complete episodes and at stopping; reuse an identical validation already completed at the stopping endpoint.

| Item | Revised controller validation |
|---|---|
| Demand groups per network | 11 pregenerated DR mixtures |
| Group duration | Provisionally 600 seconds, or 120 joint control steps |
| Complete rollout | All 11 groups in fixed order: 6600 seconds / 1320 joint control steps |
| Routine realizations | One; each starts empty and runs the complete sequence |
| Work per validation round | One complete rollout, totaling 6600 simulated seconds; no reset between the eleven blocks |
| Traffic/recurrent state | Reset only at the start of a complete rollout; retain vehicles, queues and LSTM state across group switches |
| Stopping | Minimum episodes reached and ten consecutive routine evaluations without meaningful improvement; no extra confirmation |

If 60 seconds per group is intended, explicitly select a 660-second total or repeat the eleven-group sequence ten times to reach 6600 seconds before freezing. Never encode `11×60` as 6600 in documentation or manifests. The following formula and budgets assume `11×600` provisionally.

Groups are **mixtures of the eleven original profiles**, not simply those profiles played individually. By default retain the existing DR distribution: generate `w_b ~ Dirichlet(1,…,1)` offline per network for `b=1,…,11`, with eleven nonnegative weights summing to one. Combine normalized demands as `d_b = Σ_k w_bk d_k`; introduce no additional total-flow multipliers, spatial noise or peak settings in this revision. Eleven groups form a small fixed development/selection benchmark, not guaranteed coverage of the entire DR space; generalization claims require the independent final tests.

Freeze all weights, group IDs/order, interval boundaries, profile names/order, network/profile/generator hashes, design seed and traffic/SUMO/policy seeds for the single routine realization (derived from manifest hash, scenario ID, realization index and stream name). Profile storage order can differ between networks: bind weights to names rather than copying unchecked array positions across maps. All checkpoints, algorithms, publication training seeds and continuation methods reuse the same validation inputs within each network, without redrawing weights or reordering groups. Tuning uses a separate development manifest with the same dimensions but different groups and random streams; do not repeatedly tune against publication selection inputs.

Routine validation uses realization index 0, reusing the eleven demand-rate groups, order and fixed arrival/vehicle streams, SUMO seed and MA2C/PPO policy seed; DQL is greedy. Materialize/share traffic artifacts per complete rollout. Vehicles carry across group boundaries, so a group's queue reflects earlier traffic; the eleven blocks are not independent experimental samples. Validation neither trains WCE nor allows it to change this external DR sequence. Isolated models and random streams leave training parameters, optimizers, replay and counters unchanged.

### 4.2 Metric and plateau stopping

Select using mean network-total queue over one complete rollout, including every measurement from second 1 through 6600:

\[
J_{\mathrm{val}}=\frac1R\sum_{r=1}^{R}\frac1T\sum_{t=1}^{T}\sum_{\ell\in L}q_{\ell,r}(t),\qquad R=1,\quad T=6600.
\]

Equal 600-second blocks make this equivalent to equally averaging their block means. Routine best uses this single overall score; retain realization and block means diagnostically. One fixed realization reduces cost but does not estimate variability across realizations; the current stage uses only training seed 101 and does not estimate cross-training-seed variance; final tests retain ten evaluation realizations. Routine validation every 100 complete 6600-second training episodes adds approximately 1% in simulated seconds, excluding episode zero, stopping evaluations; this is not a wall-clock estimate.

Save every strict improvement; retain the earlier checkpoint on ties. Throughput, pending vehicles, teleports, collisions and tail metrics are diagnostic only. Incomplete rollouts have no valid score.

| Stage | Minimum episodes | Maximum episodes |
|---|---:|---:|
| Initial | 1000 | 5000 |
| Offline WCE | 500 | 3000 |
| Continuation | 1000 | 5000 |

Stop when both conditions hold: the stage minimum episode count has been reached and ten consecutive routine evaluations have produced no meaningful improvement. At one evaluation per 100 episodes this normally means 1000 episodes. Episode zero establishes the baseline and does not count as a failed improvement. Patience accumulates before the minimum and is preserved across checkpoint recovery.

Meaningful improvement is `max(anchor×0.01,0.1)`: a decrease for controllers, an increase for WCE. Reaching this cumulative improvement updates the anchor and resets the counter. Every strict improvement still saves best; ties keep the earlier checkpoint. Smaller improvements do not reset patience.

Range, slope and current-to-best gap are diagnostic only, as are training entropy, loss and gradients. Remove all stability gates and additional confirmation evaluations. Select historical best at stopping, even if the last model has deteriorated. All methods follow the same stage limits and may stop at different episodes.

Patience produces `early_stopped`; parents/offline WCE without meaningful improvement may also stop, but report `no_learning`. Hitting the budget first produces `max_budget_reached` (or `no_learning` for unimproved parents/offline WCE). Errors are `failed`; unfinished segments are `interrupted`. Early stopping does not prove convergence or global optimality. Automatic publication branching uses historical best from `early_stopped` ancestors; no-learning and budget-limited runs require inspection.

### WCE

Freeze the best parent. Retain Grid CNN-A2C and Monaco GCN-A2C. Output eleven demand weights every 600 seconds and update once per full episode. Every 100 episodes, evaluate full 6600-second mean queue on one fixed realization (index 0), maximizing the result. Do not add confirmation evaluations; use the same minimum-episode and ten-evaluation patience rule.

This evaluates WCE's own adversarial demand policy: **do not replace its actions with the fixed eleven-group DR sequence**, which would eliminate the distinction between WCE candidates. Offline WCE uses the same realization index at episode 0 and every 100 completed episodes. Its score is the mean queue under the candidate’s own demand policy. Controller selection/stopping in online-WCE continuation still uses the external fixed DR sequence above.

Fix external random streams and demand constraints, not WCE-generated vehicle schedules. Index random streams by block, OD and variable type. Retain existing learning-rate, reward-scale and entropy starts; WCE advantage normalization is enabled by default (`adv_norm = true`): each update normalizes actor advantages over its eleven macro-step samples using `(A - mean(A)) / (std(A) + 1e-8)`, with population standard deviation. Critic return targets and reward scaling retain their existing definitions. This applies to offline and online WCE learning; fixed WCE performs inference only. Diagnostics record advantage means and standard deviations before and after normalization. Reference the frozen parent by hash. Publication fixed/online branches share the same patience-stopped best WCE.

### Continuation and fairness

All methods branch from the same best parent. Random group samples uniform one-hot profiles; DR samples `Dirichlet(1,…,1)`; fixed freezes WCE; online updates it every episode. DQL retains replay/target/optimizer but resets epsilon to 0.1 at branching, decaying to 0.01 over 132,000 steps. Resume must not repeat this reset.

Include the parent as the episode-zero best candidate; continuation may fail to improve it. The 1000-episode minimum controls early stopping. At termination, freeze the best over the entire validation history. Exact ties keep the earliest checkpoint. Online stopping uses external validation, not the changing adversary's training reward. Recovery bundles contain simultaneous controller/WCE states, never a combination of unrelated best moments.

## 5. Memory and data reduction

Continue per-second queue measurement. Use float64 streaming sums, squared sums, maxima, vehicle counters and timestamp validation; retain the current lane vector and at most 600 seconds of total queue. Accumulate WCE block rewards directly. Write one episode summary and aggregate learner diagnostics per demand block. Replace `batch_lengths` growth with histograms; full sequences are test-only.

Use fixed NumPy replay arrays indexed by joint transition. Observations/rewards are float32, actions integer, terminal flags boolean. Omit unused value/policy fields. Serialize in chunks without copying the full buffer; no float16 or lossy compression.

Inference-only validation creates no optimizer or replay; DQL loads only online weights. Pause training and run sequential validation in a separate process. Frozen parents also use inference mode.

| Output mode | Retained data |
|---|---|
| Training | Episode summaries, block diagnostics, demand weights/hashes, validation history |
| Routine validation | Rollout scalars, scenario scores, candidate identities |
| Final evaluation | Scalars, minute curves, lane statistics for predefined windows |
| Audit | Per-second lanes, control records and necessary SUMO data |

Disable routine per-episode NPZ, per-second JSONL, per-step rewards, per-agent/update gradients, automatic figures and unnecessary tripinfo. Full audits are restricted to integration tests and the first Uniform/peak_1.5 realization for each final model. Where tripinfo is needed, parse temporary local XML incrementally into summaries and remove it. Retain bounded failure log tails.

Generate training traffic per 600-second block, retaining generator state, weights, versions and hashes. Generate fixed evaluation traffic once on demand and share compressed artifacts; vehicles reference route tables. Seeds accompany generator, network, profile and traffic hashes. Write compressed result shards; derive CSVs, plots and TensorBoard offline. Cross-version bitwise equivalence is not promised.

## 6. Checkpoints and archives

Full checkpoints include parameters, optimizer, replay/pending state, RNG, schedules, counters, selection state and online WCE. Lightweight snapshots contain inference parameters/identity and cannot resume training.

Save recovery points every ten episodes, retaining the latest two. Keep the current best in full, the latest five validation candidates as lightweight snapshots, the selected best as an inference snapshot and the final state in full. Pin downstream-referenced parents/WCEs. Reuse files across roles. Remove only unreferenced intermediates after validating replacements; saving best continuously does not retain every historical best replay forever.

Write temporary files → parameters/state → hashes → completion manifest → verify destination → update references → collect unused objects. For cross-filesystem transfers, publish atomically within the destination filesystem after copying. Never remove the previous recovery point on failure. Cleanup is restricted to new managed outputs.

## 7. Local CPU execution and recovery

The current target is a local Linux workstation. Slurm/Apptainer files remain optional migration materials, not local prerequisites. One task is a map × controller × training seed × stage/method. Each owns its Python process, TF session, SUMO connection, streams and output directory. No shared replay, asynchronous gradients or distributed learner is introduced.

Start with one task, then compare two/four measured tasks. TF intra/inter, BLAS, OpenMP and MKL default to one thread; set them before importing scientific libraries. DQL, MA2C/PPO and WCE share effective session configuration. Keep GPU disabled and TF_ENABLE_ONEDNN_OPTS=0. A validating trainer remains resident while its independent inference worker runs sequential realizations.

Reserve at least 20% total RAM. Admit tasks only when the sum of measured whole-process-tree peaks times 1.3 fits the experiment budget, while checking physical-core and disk budgets. Include full replay, validation and checkpoint publication. Unknown classes run alone; short trainer-only RSS does not certify a formal task. Five-second measurements and coverage-aware resource export are implemented. Default disk reserve is 10 GiB plus checkpoint headroom.

`main.py local prepare/run/status/resume` freezes a persistent DAG and reuses the stage CLIs. It records task identities, argv, outputs, resource classes, statuses, receipts and logs. One scheduler lock per run root and one lock per output prevent duplicates. On restart, reconcile supervisor PID/start-time identities and durable receipts before launching more work. Parents release their own WCE and baseline/random/DR branches; fixed/online also require their matching WCE. Automatic upstream admission requires successful exit, verified full best and early_stopped. Budget exhaustion, no_learning and failures remain visible.

The local entry disables time segmentation by explicitly passing zero, with optional --segment-seconds; the direct legacy CLI keeps its 23-hour default. A first stop signal requests an episode boundary, while force interruption rolls back an unfinished episode. Preserve patience, RNG, counters and DQL exploration. Normal segments resume automatically, stopping after three attempts without progress; unexpected failures are not retried indefinitely.

Use isolated Python 3.10, TF CPU 2.15.1, NumPy 1.26.4 and released SUMO/TraCI/sumolib 1.26.0 plus the full hash-locked dependency set. `main.py runtime-lock` records a local profile with exact Python patch/packages, binary hash, system/CPU and effective threads. `verify.py --extended` must pass in that same locked environment. Development-system correctness evidence cannot bypass the publication runtime gate. Retain the container-lock path for future deployments.

Source/configuration/documents belong in local Git; CBWCE_DATA_ROOT holds immutable shared inputs, CBWCE_RUN_ROOT active recovery/models/results, and CBWCE_CACHE_ROOT rebuildable traffic/evaluation caches. Use separate temporary directories. Verify stage archives before cleaning inactive copies, retain ancestry references, and back up critical results to another physical disk/NAS. An archive on the same disk is not an independent hardware-failure backup. Current work does not deploy Hábrók.

## 8. Acceptance and delivery

Acceptance covers relocation without the original tree; both networks × three controllers; streaming/audit agreement; learning invariance under reduced logging; bounded replay/monitoring; isolated validation; full recovery and snapshot rejection; interruption/corruption/duplicate-job protection; reference-safe retention; RSS/bytes/file-count/write-time measurements; bilingual consistency; and refusal of publication runs without a frozen manifest.

Implemented validation changes: separate `dr_sequence_v1` schema, eleven 600-second blocks, one routine rollout/eleven intervals, `patience_v1` stopping without confirmation, unchanged 3600-second final tests and one fixed realization per routine WCE check. `validation-manifest` generates weights from an explicit design seed and binds profile names and input hashes; random streams derive from the manifest hash and realization index. `pilot-manifest` generates an unfrozen development sequence. Traffic and recurrent state persist across all blocks; diagnostics retain block means while best uses the overall mean. Short development runs scale all eleven blocks. The new schema, manifest hashes, source fingerprint and run protocol invalidate old selection/cache/gate identities: start new runs and revalidate instead of resuming old scores or best/patience.

## 9. Single-seed task delivery and evaluation updates

Development remains 24 candidates ×1000 episodes using seed 9001. Formal work is six parents, six WCEs and thirty continuations. The thirty selected checkpoints require 6900 main evaluations in total. Default training_seeds is [101]; extensions use a new campaign with --training-seeds and preserve old run identities. Reports require the declared matrix. Single-seed training_seed_sd is null; ten evaluation realizations are not ten training seeds.

Tasks: [baseline](tasks/01_baseline_training_en.md), [WCE](tasks/02_wce_training_en.md), [continuation](tasks/03_continuation_training_en.md), [evaluation](tasks/04_evaluation_comparison_en.md), each with a separate Chinese counterpart, storage frequencies and parallel execution requirements.

Final tests use test_suite_v1 and evaluation_metrics_v2, retaining eleven seen plus twelve test scenarios. Explicit pre/peak/post windows, p95, final-600-second slope and completed-trip summaries are included. Parse temporary tripinfo XML and save each rollout atomically as a compressed shard. Full recovery also preserves frozen-policy sampling RNGs. The queue covers tuning, formal dependencies, scenario evaluation and resource export. Blockage proposals remain an appendix and are not implemented incidents.
