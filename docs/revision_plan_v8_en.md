# Revision Plan v8.1: Standalone Project, Validation-Driven Training and Hábrók Deployment

Date: 2026-10-27. Chinese counterpart: [revision_plan_v8_zh.md](revision_plan_v8_zh.md).

This document specifies the protocol and acceptance criteria. Code availability is not acceptance evidence. Actual verification is recorded separately in `implementation_status.md`; publication training, container construction and deployment are not claimed complete.

## 1. Objectives and experimental matrix

Create the compact `cbwce_standalone` project for Groningen's Hábrók cluster, retaining MA2C, DQL and PPO with validation selection, plateau stopping, bounded outputs and resumable Slurm jobs.

| Item | Protocol |
|---|---|
| Networks | 5×5 Grid and the Monaco subnet used by the current revised runner; do not substitute the repaired network |
| Publication seeds | 101, 202, 303; development seed 9001 |
| Training episode | 6600 seconds, 1320 joint network decisions |
| Control/demand interval | 5/600 seconds |
| Initial training | Existing fixed sequence of eleven demand profiles |
| Continuations | `baseline`, `random_group`, `domain_randomization`, `fixed_wce`, `online_wce` |
| Selection | Minimize mean queue for controllers; maximize frozen-parent mean queue for WCE |
| Endpoints | Common-budget best and best at individual stopping |

**Validation scenario count, composition and generation parameters remain user decisions.** Require an explicit frozen manifest; the supplied Uniform development fixture must not replace publication validation. Separate development, selection and final-test identities and random streams.

Publication totals: 18 parents, 18 offline WCE runs and 90 continuations. Final evaluation provisionally retains eleven seen and twelve generalization scenarios, ten realizations each: 20,700 rollouts per endpoint. Identical selected checkpoint hashes may reuse results.

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

Replace linear `iqll` with independent DQL networks: `128 ReLU → 64 ReLU → Q`. Use distinct online/target networks, uniform replay, Huber loss and epsilon-greedy exploration; greedy evaluation. Do not add Double DQN, dueling heads or prioritized replay.

| Parameter | MA2C | PPO | DQL |
|---|---:|---:|---:|
| Learning rate | 5e-4 | 3e-4 | 1e-4 |
| Discount | 0.99 | 0.99 | 0.99 |
| Optimizer | RMSProp | Adam | Adam |
| LSTM | 64 | 64 | None |
| Advantage normalization | On | On | Not applicable |
| Rollout/batch | 120 | 120 | 64 |
| Updates | Once/rollout | Four epochs/rollout | Once/four learning steps |
| Entropy | 0.01 | 0.01 | Not applicable |
| PPO clip | — | 0.2 | — |
| Replay | — | — | 50,000 joint transitions |
| Warm-up | — | — | 5000 steps |
| Target synchronization | — | — | Every 1000 learning steps |
| Gradient-norm limit | 40 | 40 | 10 |

Preserve existing MA2C/PPO encoder widths. Initial DQL epsilon decreases linearly from 1.0 to 0.01 over 500,000 steps.

| Reward divisor | MA2C | PPO | DQL |
|---|---:|---:|---:|
| Grid | 2000 | 3000 | 3000 |
| Monaco | 200 | 1000 | 1000 |

Learner clipping threshold: 2. Evaluation is unscaled and unclipped. MA2C uses neighborhood reward while the other controllers use global reward, so identical divisors are not required.

Four development candidates per network/controller, each for 1000 episodes: MA2C rates `{5e-4,2.5e-4}` × entropy `{0.01,0.001}`; PPO rates `{3e-4,1.5e-4}` × the same entropy values; DQL rates `{1e-4,3e-4}` × replay `{50000,20000}`. Select development-validation best, breaking ties toward table defaults. Publication seeds and final tests do not tune parameters. All continuation methods share controller settings.

## 4. Validation, best and plateau stopping

Validate controllers at episode zero, every 100 complete episodes and at stopping. Use three fixed realizations per scenario, 3600 seconds each, starting empty. MA2C/PPO use fixed stochastic-policy seeds; DQL is greedy. Pair traffic artifacts and SUMO seeds across methods. Independent models/random streams must not alter learner parameters, optimizer, replay or counters.

The score is the equally weighted scenario mean of network-total queue:

\[
J=\frac1G\sum_g\frac1R\sum_r\frac1{3600}\sum_{t=1}^{3600}\sum_{\ell\in L}q_\ell(t),\quad R=3.
\]

Save every strict improvement; retain the earlier checkpoint on ties. Throughput, pending vehicles, teleports, collisions and tail metrics are diagnostic only. Incomplete rollouts have no valid score.

| Stage | Minimum episodes | Maximum episodes |
|---|---:|---:|
| Initial | 1000 | 5000 |
| Offline WCE | 500 | 3000 |
| Continuation | 1000 | 5000 |

Plateau candidacy requires the minimum episode count; five validation intervals (500 episodes) without meaningful improvement; latest-five range ≤ `max(mean,1)×2%`; absolute fitted slope ≤ `max(mean,1)×0.5%` per validation interval; current-to-best gap ≤ `max(best,1)×2%`. Initial training must improve meaningfully over episode zero at least once.

Meaningful improvement is `max(anchor×0.01,0.1)`. Reset the anchor/patience only after cumulative improvement reaches that threshold; smaller improvements still update best.

Confirm the latest five candidates and best on seven additional independent fixed realizations, ten total per scenario. Recheck spread, trend and current-to-best gap without redefining routine best ranking.

Terminal outcomes are `plateau_confirmed`, `max_budget_unconfirmed`, `no_learning` and `failed`. An unfinished segment is `interrupted`, not converged. An empirical plateau is not global optimality; exhausting the budget does not imply convergence.

### WCE

Freeze the best parent. Retain Grid CNN-A2C and Monaco GCN-A2C. Output eleven demand weights every 600 seconds and update once per full episode. Every 100 episodes, evaluate full 6600-second mean queue on ten seed combinations, maximizing the result. Add twenty combinations for confirmation, thirty total.

Fix external random streams and demand constraints, not WCE-generated vehicle schedules. Index random streams by block, OD and variable type. Retain existing learning-rate, reward-scale and entropy starts; explicitly disable WCE advantage normalization. Reference the frozen parent by hash. Publication fixed/online branches share the same plateau-confirmed best WCE.

### Continuation and fairness

All methods branch from the same best parent. Random group samples uniform one-hot profiles; DR samples `Dirichlet(1,…,1)`; fixed freezes WCE; online updates it every episode. DQL retains replay/target/optimizer but resets epsilon to 0.1 at branching, decaying to 0.01 over 132,000 steps. Resume must not repeat this reset.

Include the parent as the episode-zero best candidate; continuation may fail to improve it. Freeze `best_at_common_budget` at 1000 episodes/1,320,000 additional learning steps; later checkpoints cannot replace it. Online stopping uses external validation, not the changing adversary's training reward. Recovery bundles contain simultaneous controller/WCE states, never a combination of unrelated best moments.

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

Save recovery points every ten episodes, retaining the latest two. Keep the current best in full, the latest five validation candidates as lightweight snapshots, the common-budget best as inference state and the final state in full. Pin downstream-referenced parents/WCEs. Reuse files across roles. Remove only unreferenced intermediates after validating replacements; saving best continuously does not retain every historical best replay forever.

Write temporary files → parameters/state → hashes → completion manifest → verify destination → update references → collect unused objects. For cross-filesystem transfers, publish atomically within the destination filesystem after copying. Never remove the previous recovery point on failure. Cleanup is restricted to new managed outputs.

## 7. Hábrók

Local Git stores code. `/scratch/<account>/cbwce` stores active inputs/checkpoints/results; `$TMPDIR` stores extracted code, SUMO temporaries and caches; `/projects/<account>/cbwce` stores verified archives; RDMS/an independent backup preserves important long-term artifacts.

Scratch is not backed up; projects storage is not directly available to compute nodes. Persist to scratch during jobs, then archive from the login side. Inspect actual space/inode quotas with `hbquota`, rather than assuming documentation examples apply to the account.

Apptainer runtime: Python 3.10, TensorFlow CPU 2.15.1 via compat.v1, NumPy 1.26.4, SUMO/TraCI/sumolib 1.26.0. Lock the Python patch, base-image digest, dependencies and SIF checksum during construction. Reverify the new runtime; do not claim equivalence with historical TF1.12 runs. All publication jobs use the same image and install nothing online during execution.

CPU pilot defaults: regular partition, one task, two CPUs, 8 GiB, single-threaded numerical libraries, 24-hour segments, array concurrency four. The memory request is provisional: measure RSS, episode and write durations, then request at least 1.3× measured peak rounded upward. Disable oneDNN optimization as part of fixed runtime settings.

Forward Slurm signals to workers. Stop starting episodes before the deadline; discard unfinished episodes if necessary. Resume preserves budgets, patience and exploration. Periodically persist to scratch instead of relying on exit copies. Downstream stages require job success and application `plateau_confirmed`; unfinished normal segments resume their current stage. Verify archives before removing inactive scratch copies.

Official references: [storage](https://wiki.hpc.rug.nl/habrok/migration/migration_habrok), [quotas](https://wiki.hpc.rug.nl/habrok/data_management/quota), [Apptainer](https://wiki.hpc.rug.nl/habrok/examples/apptainer), [Slurm](https://wiki.hpc.rug.nl/habrok/job_management/scheduling_system).

## 8. Acceptance and delivery

Acceptance covers relocation without the original tree; both networks × three controllers; streaming/audit agreement; learning invariance under reduced logging; bounded replay/monitoring; isolated validation; full recovery and snapshot rejection; interruption/corruption/duplicate-job protection; reference-safe retention; RSS/bytes/file-count/write-time measurements; bilingual consistency; and refusal of publication runs without a frozen manifest.

Deliver standalone code, bilingual plans, runtime locks/build definitions, input releases, Slurm templates, verification evidence, resource reports and archival tools. Actual trained models and result archives are produced by subsequent authorized, gated experiments; they cannot be fabricated in advance.
