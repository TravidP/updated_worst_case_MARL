# 03 Five-method continuation and recovery task

[Local automation and workflow console](05_local_orchestration_en.md)

Online branches show separate controller and WCE training rewards. Only the controller is validated under external demand. Fixed WCE is inference-only.

> Task specification. Formal training defaults to seed 101; development uses seed 9001. See [implementation evidence](../implementation_status.md). This document does not claim completed formal training.

## Objective and branch definitions

Two maps × three algorithms × seed 101 × five methods give thirty continuation branches. All five branches for a map/algorithm start from the same full best parent; fixed/online also start from the same full best offline WCE. Formal upstream runs must exit successfully, pass integrity checks and have early_stopped status.

| Method | Demand every 600 seconds | Controller updates | WCE updates |
|---|---|---|---|
| baseline | Original eleven profiles in fixed order, one-hot weights | Yes | None |
| random_group | Uniform choice among eleven profiles | Yes | None |
| domain_randomization | Dirichlet(1,…,1) mixture of eleven profiles | Yes | None |
| fixed_wce | Selected WCE observes the current state and samples weights | Yes | No; frozen parameters |
| online_wce | Same initial WCE observes and samples weights | Yes | Once per complete episode |

Fixed refers to parameters; it still responds to state and samples using its RNG, rather than replaying a fixed traffic sequence. Reuse the [baseline task](01_baseline_training_en.md) environment/state/action/reward/networks/optimizer settings and the [WCE task](02_wce_training_en.md) architecture. No new waiting-time input or enlarged network is introduced.

## Additional budget and RL update frequency

An episode is 6600 seconds or 1320 joint control steps. All five branches use the same episode limits and stopping rule. Actual interaction counts can differ. One learning step is a five-second network decision, not multiplied by junction count or interpreted as an optimizer update.

| Item | MA2C | PPO | DQL |
|---|---|---|---|
| Trigger | Every 120 consecutive steps | Every 120 steps, four whole-sequence epochs | Every four steps after warm-up |
| Data per junction | 120 valid samples | Same 120 samples reused for four epochs | 64 uniform replay endpoints with up to sixteen history steps |
| Updates per junction per full episode | 11 | 44 | 330 after warm-up |
| Memory | Episode LSTM64 | Episode LSTM64 | Finite sixteen-step LSTM64 window |

Online WCE additionally updates once after eleven macro steps; fixed WCE never updates. Preserve each algorithm's batch and clipping settings, while matching them across its five branches. Report interaction/optimization counts, elapsed time and whole-process-tree memory. Equal interactions are not equal computation.

DQL branches preserve online/target networks, replay and optimizer. Epsilon starts at 0.1 and falls to 0.01 over 132000 new steps. Branch only once; recovery preserves the schedule origin and does not reset exploration. Other controllers retain parent optimizer/scheduler state.

## Selection, stopping and recovery

The parent is an episode-zero best candidate; continuation may fail to improve it. Always validate controllers on the fixed exogenous eleven-by-600-second DR sequence, realization zero. Online selection does not use training reward against a changing adversary. Validate every 100 episodes, minimum 1000 and maximum 5000; patience is ten with threshold max(anchor×0.01,0.1), without extra confirmation.

At termination, freeze the controller with the lowest validation mean queue across all checks from episode 0 through stopping. An exact tie keeps the earlier checkpoint. The selected checkpoint may be the starting parent or a later improvement, including one after episode 1000. Record both the selected episode and the number of episodes trained.

Full checkpoints include parameters, optimizer, replay/pending data, inference history, RNG, schedules, counters and patience. Online saves controller and WCE from the same instant, not a splice of their individual best states. Fixed saves frozen WCE sampling state; parameters alone cannot reproduce continuation. `--resume` restores latest full state in the original directory; initial branching reads parent/WCE best. Keep those operations distinct.

```bash
"$CBWCE_PYTHON" main.py continue --network grid --controller dql --seed 101 --method online_wce --parent "$CBWCE_PARENT_BEST" --wce "$CBWCE_WCE_BEST" --validation "$CBWCE_DATA_ROOT/grid_validation.json" --gate "$CBWCE_RUN_ROOT/verification/gate.json" --output "$CBWCE_RUN_ROOT/manual_grid_dql_online"
# Resume: same command/directory plus --resume; baseline/random_group/DR omit --wce
```

## Saving and retention

| Artifact | Frequency | Retention |
|---|---|---|
| `manifest.json`, provenance and parent identity | Start | Immutable identity; inputs, runs and caches outside source |
| `blocks.jsonl.gz` | Every 600-second demand block | Weights, traffic hash, scheduled vehicles, raw queue and aggregated diagnostics; 11 rows/episode |
| `episodes.jsonl.gz` | Every complete episode | Queue, throughput, elapsed time and steps; trainer RSS is not whole-job memory |
| `validation.jsonl.gz` | Episode 0, every 100 episodes and terminal endpoint | Scores, block diagnostics, candidate hash and stopping diagnostics; reuse an identical endpoint |
| Full recovery | Every 10 episodes, validation, normal segment/end | Latest two; parameters, optimizer, replay, RNG, counters and patience |
| Best | Every strict validation improvement | Current full best and inference snapshot; earlier candidate wins a tie |
| Validation snapshots | Every validation | Latest five lightweight candidates; inference snapshots cannot resume training |
| Stage archive | Stage end | Final state, referenced ancestry, manifests and summaries; verify before archiving |

Routine training does not retain per-second lanes, per-step rewards, episode NPZs or tripinfo. Training completed counters retain raw SUMO arrivals; final evaluation uses TripInfo to separate normal completions from removals, so training throughput summaries are not final completion rates. The first stop signal requests a durable episode boundary; a forced interruption discards the unfinished episode and restarts from full recovery. Attempts distinguish rolled-back records; offline export removes duplicates. `CBWCE_DATA_ROOT` holds shared inputs, `CBWCE_RUN_ROOT` runs, and `CBWCE_CACHE_ROOT` rebuildable caches. A separate disk/NAS provides an independent backup.

Retain the selected inference snapshot for evaluation and the full best for training dependencies. Pin referenced ancestors. Archives include the reference inventory and all parents and WCEs still used by branches.

## Local multitask execution

Release thirty branches by dependency. After a parent completes, baseline/random_group/DR can run alongside its offline WCE. Fixed/online wait for that WCE. Interleave maps/algorithms instead of waiting for an entire group of five. Measure the online resource class separately, including both trainable models and the validation worker.

Use separate processes, output directories, SUMO connections and random streams. All training and evaluation tasks share one local queue/resource ceiling. Default TF intra/inter and BLAS/OpenMP/MKL thread counts are one, on CPU.

The dashboard workflow admits concurrent workers from startup, with 12 tuning slots and up to 24 formal slots subject to host budgets. See the [workflow guide](05_local_orchestration_en.md). The following measurement procedure applies to the standalone campaign CLI.

Measure one complete process tree first, then compare two/four concurrent measured tasks. An unmeasured resource class, or DQL without full-replay coverage, runs alone. Reserve at least 20% of total RAM; the sum of measured active peaks multiplied by 1.3 must fit the budget. Check CPU and space for overlapping old/new checkpoints. Disk reserve defaults to 10 GiB and can be increased. RSS/CPU are sampled every five seconds; state transitions persist immediately.

```bash
"$CBWCE_PYTHON" main.py local run --campaign "$CBWCE_CAMPAIGN" --max-parallel 1
"$CBWCE_PYTHON" main.py local status --campaign "$CBWCE_CAMPAIGN"
# Stop the queue, export measurements, then increase concurrency for covered classes
"$CBWCE_PYTHON" main.py local export-resources --campaign "$CBWCE_CAMPAIGN" --output "$CBWCE_RESOURCES"
"$CBWCE_PYTHON" main.py local resume --campaign "$CBWCE_CAMPAIGN" --max-parallel 2 --resources "$CBWCE_RESOURCES"
```

`CBWCE_CAMPAIGN` is the directory containing campaign.json; `CBWCE_RESOURCES` is a new JSON path. Five-second samples may miss brief peaks; 1.3x is operational headroom, not a worst-case guarantee. Short pilot RSS cannot certify full replay. Local runs have no time segmentation by default; `--segment-seconds` is optional. Normal segments resume automatically, stopping after three attempts without progress. Failures are recorded, not retried indefinitely. File locks reject duplicate run ownership.

## Acceptance and later seed expansion

Check identical parent hashes across five branches and WCE hashes across fixed/online; matching controller structure/hyperparameters; correct DQL branch/recovery exploration; uninterrupted versus resumed tensors and next updates; and selection of the historical best through stopping. Keep failures, no_learning and budget exhaustion visible.

To add 202/303 later, prepare a new extension without modifying old manifests or scientific configuration:

```bash
"$CBWCE_PYTHON" main.py local prepare --output "$CBWCE_RUN_ROOT/extended_queue" --extend "$CBWCE_RUN_ROOT/formal_queue" --training-seeds 101 202 303 --validation-grid "$CBWCE_DATA_ROOT/grid_validation.json" --validation-monaco "$CBWCE_DATA_ROOT/monaco_validation.json" --gate "$CBWCE_RUN_ROOT/verification/gate.json"
```

With the same run root, seed-101 tasks with successful receipts and validated identity/checkpoints are reused. Each added seed must complete the whole matrix, not only favourable methods. Source/input/runtime changes cannot masquerade as the same recovered experiment.

Sources: [runner](../../experiments/runner.py), [DQL branch](../../agents/dql.py), [checkpoints](../../experiments/checkpoint.py), [stopping](../../experiments/selection.py), [local queue](../../experiments/campaign.py). Chinese: [matching task](03_continuation_training_zh.md).
