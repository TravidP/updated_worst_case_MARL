# 01 Initial baseline training task

[Local automation and workflow console](05_local_orchestration_en.md)

Training and validation log mean immediate controller reward across junctions and control steps. The console also plots mean queue and the best validation point.

> Task specification. Formal training defaults to seed 101; development uses seed 9001. See [implementation evidence](../implementation_status.md). This document does not claim completed formal training.

## Training budget and stopping time

| Individual run | Training episodes | Stop point |
|---|---:|---|
| Development candidate (seed 9001) | Exactly 1000 | Each of the 24 candidates completes its budget before configuration selection |
| Formal baseline parent (seed 101) | 1000–5000 | Validate at episode 0 and every 100 episodes. The earliest stop is episode 1000 and the cap is episode 5000 |

One episode covers 6600 simulation seconds, eleven demand blocks and 1320 joint decisions. Each formal parent trains for 1.32–6.60 million joint steps, plus validation. For a wall-clock estimate, time 100 episodes between the ends of consecutive routine validations on the target machine with the same map, algorithm and concurrency. Multiply that duration by the remaining episode count divided by 100. Calculate a range using the earliest possible stop and the cap. This includes periodic validation and checkpoint writes.

Patience counts validations before the episode minimum as well. For example, after a meaningful improvement at episode 800, ten consecutive checks without meaningful improvement at episodes 900 through 1800 stop the run at 1800. The selection section below defines the improvement threshold. [Training protocol](../../config/protocol.json) · [Stopping logic](../../experiments/selection.py)

## Objective, environment and deliverables

Train six parents: Grid / Monaco × MA2C / PPO / DQL × seed 101. Initial baseline episodes play the eleven normalized original profiles in fixed order. Baseline continuation is a separate stage and supplies the comparator under the same stopping rule.

Grid has 25 controlled junctions on a 5×5 layout; the current revised Monaco subnet has 28. Do not substitute another repaired network. Each episode lasts 6600 seconds: 1320 joint network control steps. SUMO advances at one second; control uses five-second intervals with the existing two-second yellow/three-second green transition logic, retaining same-phase behavior. Demand changes every 600 seconds without clearing vehicles or queues. Base rates are 3000 veh/h for Grid and 2383.3333 for Monaco; realized vehicle counts are random.

State includes local and neighbour wave: detector-area vehicle counts on Grid and lane vehicle counts on Monaco, with norm_wave=5 and clip_wave=2. MA2C discounts neighbour wave by coop_gamma=0.9 and adds each neighbour's previous policy vector excluding its final probability. PPO/DQL also receive neighbour wave, without fingerprints. Waiting time is not an input. Actions are discrete indices into each junction's available phase list, translated by the existing environment. Junction action sizes can differ.

| Parameter | MA2C | PPO | DQL |
|---|---:|---:|---:|
| Optimizer / initial learning rate | RMSProp / 5e-4 | Adam / 3e-4 | Adam / 1e-4 |
| Gamma | 0.99 | 0.99 | 0.99 |
| Wave encoder / LSTM | 128 / 64 | 128 / 64 | 128 / 64 |
| Additional input | Neighbour fingerprint →64 | No fingerprint | No fingerprint |
| Update data | 120 consecutive steps, once | 120 steps, four whole-sequence epochs | 64 uniform replay endpoints, up to 16 history steps each |
| Update interval | Every 120 joint control steps | Every 120 joint control steps | Every four steps after 5000-step warm-up |
| Entropy / PPO ratio clip | 0.01 / — | 0.01 / 0.2 | — |
| Gradient norm limit | 40 | 40 | 10 |
| Grid reward divisor | 2000 | 3000 | 3000 |
| Monaco reward divisor | 200 | 1000 | 1000 |

MA2C/PPO use separate actor and critic networks and preserve LSTM memory within episodes, including demand switches. MA2C retains n-step returns without GAE. DQL uses Huber TD loss, separate online/target networks and uniform replay of 50000 joint transitions. Target synchronization occurs every 1000 environment learning steps. Finite windows start with zero recurrent state and mask padding. No Double/Dueling/prioritized replay is introduced. Initial epsilon falls linearly from 1 to 0.01 over 500000 steps; evaluation is greedy. MA2C/PPO evaluation samples with a fixed policy seed.

Rewards are divided once at the learner boundary and clipped to [-2,2]. MA2C uses negative local queue plus 0.9-weighted negative neighbour queues; PPO/DQL replicate global negative queue to each junction. Controller rewards use five-second endpoints, whereas WCE/evaluation integrate per-second measurements. MA2C/PPO normalize valid-sample advantages. Report PPO ratio clipping separately from gradient clipping.

## Measured interfaces and parameter counts

| Map / algorithm | State width per junction | Action count | Trainable online parameters, whole network | Per-junction range |
|---|---|---|---:|---|
| grid / ma2c | 26/36/46 | 5 | 3,511,190 | 137,990–142,086 |
| grid / ppo | 18/24/30 | 5 | 2,647,830 | 104,070–107,142 |
| grid / dql | 18/24/30 | 5 | 1,327,165 | 52,165–53,701 |
| monaco / ma2c | 6–50 | 2/3/4/5/6 | 3,848,314 | 133,571–143,045 |
| monaco / ppo | 5–34 | 2/3/4/5/6 | 2,900,986 | 100,547–108,101 |
| monaco / dql | 5–34 | 2/3/4/5/6 | 1,452,638 | 50,306–54,148 |

Grid MA2C has 6 local wave inputs plus 6 wave inputs and 4 previous policy probabilities per neighbour. Corners, non-corner boundary junctions and interior junctions have 2, 3 and 4 neighbours, giving input widths of 26, 36 and 46.

Ranges do not imply every intermediate width occurs. The [machine-readable inventory](model_inventory.json) lists every junction, state/action/fingerprint width and parameter count. Reproduce with `python scripts/inspect_models.py --output <new-json-path>`. DQL target has the same parameter count as online but is not trainable. Full 50000-transition replay arrays occupy 263250000 / 198600000 bytes on Grid / Monaco, excluding TF and SUMO.

## Development tuning: the 24 candidates

Run all commands from the project root, after configuring the interpreter and path variables in the local-environment section below.

Before formal training, seed 9001 trains four configurations for each map/algorithm, 1000 complete episodes per candidate. This is 24 development jobs and 24000 episodes, separate from the six formal parent jobs.

| Algorithm | Two learning rates | Two values of the other parameter | Candidates per map |
|---|---|---|---:|
| MA2C | 5e-4, 2.5e-4 | Entropy 0.01, 0.001 | 4 |
| PPO | 3e-4, 1.5e-4 | Entropy 0.01, 0.001 | 4 |
| DQL | 1e-4, 3e-4 | Replay 50000, 20000 | 4 |

Select each map/algorithm's lowest development-validation best queue; exact ties use the lower tie_rank, preferring the default starting point. Version the six winners, rerun verification and freeze the formal campaign. All five continuation methods reuse the selected controller configuration. Development manifests must not share formal selection/test streams.

```bash
# From project root; explicitly choose development seeds distinct from formal designs
"$CBWCE_PYTHON" main.py validation-manifest --network grid --design-seed "$CBWCE_DEV_GRID_SEED" --output "$CBWCE_DATA_ROOT/grid_dev.json"
"$CBWCE_PYTHON" main.py validation-manifest --network monaco --design-seed "$CBWCE_DEV_MONACO_SEED" --output "$CBWCE_DATA_ROOT/monaco_dev.json"
"$CBWCE_PYTHON" main.py local prepare-tuning --output "$CBWCE_RUN_ROOT/tuning_queue" --validation-grid "$CBWCE_DATA_ROOT/grid_dev.json" --validation-monaco "$CBWCE_DATA_ROOT/monaco_dev.json"
"$CBWCE_PYTHON" main.py local run --campaign "$CBWCE_RUN_ROOT/tuning_queue" --max-parallel 1
"$CBWCE_PYTHON" main.py local select-tuning --campaign "$CBWCE_RUN_ROOT/tuning_queue" --output "$CBWCE_DATA_ROOT/selected_configs"
"$CBWCE_PYTHON" main.py local prune-tuning --campaign "$CBWCE_RUN_ROOT/tuning_queue" --selection "$CBWCE_DATA_ROOT/selected_configs"
```

Selection requires all 24 candidates to finish 1000 episodes. It exports six INIs and selection.json without silently overwriting source configuration. Version the winning INIs under `config/revised`, then reverify. Development jobs are explicitly pilot runs and cannot serve as formal parents.

After selection, run the prune-tuning command above. It rechecks the 24 completed candidates, six validation winners and hashes of the exported INIs before removing candidate checkpoints including DQL replay. All candidate configs, training and validation logs, results and selection artifacts are retained. The queue records the original payload bytes in `tuning_cleanup.json`. Rerun the same command after an interruption. Export any development resource measurements before pruning.

## Local environment and formal launch

Use an isolated Python 3.10 environment and the hash-locked `packaging/requirements.lock`: TF CPU 2.15.1, NumPy 1.26.4, SUMO/TraCI/sumolib 1.26.0 and the complete dependency lock. These commands are execution templates, not a claim that the current host matches:

```bash
python3.10 -m venv .venv
export CBWCE_PYTHON="$PWD/.venv/bin/python"
"$CBWCE_PYTHON" -m pip install --require-hashes -r packaging/requirements.lock
export CBWCE_ASSETS_ROOT="$PWD/assets"
export CBWCE_DATA_ROOT="$(dirname "$PWD")/cbwce_data"
export CBWCE_RUN_ROOT="$(dirname "$PWD")/cbwce_runs"
export CBWCE_CACHE_ROOT="$(dirname "$PWD")/cbwce_cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_ENABLE_ONEDNN_OPTS=0
export PATH="$PWD/.venv/bin:$PATH"
export CBWCE_RUNTIME_LOCK="$CBWCE_DATA_ROOT/runtime.local.json"
"$CBWCE_PYTHON" main.py runtime-lock --output "$CBWCE_RUNTIME_LOCK"
"$CBWCE_PYTHON" scripts/verify.py --extended --output "$CBWCE_RUN_ROOT/verification"
```

SUMO requires local TraCI sockets. The formal lock records interpreter/dependency identities, the SUMO executable SHA-256, OS/CPU and effective threads. A complete gate must match the current source/environment. Missing/development dependencies permit development checks only; do not fabricate a container lock or bypass publication checks with pilot mode.

```bash
"$CBWCE_PYTHON" main.py validation-manifest --network grid --design-seed "$CBWCE_VALIDATION_GRID_SEED" --frozen --output "$CBWCE_DATA_ROOT/grid_validation.json"
"$CBWCE_PYTHON" main.py validation-manifest --network monaco --design-seed "$CBWCE_VALIDATION_MONACO_SEED" --frozen --output "$CBWCE_DATA_ROOT/monaco_validation.json"
"$CBWCE_PYTHON" main.py local prepare --output "$CBWCE_RUN_ROOT/formal_queue" --validation-grid "$CBWCE_DATA_ROOT/grid_validation.json" --validation-monaco "$CBWCE_DATA_ROOT/monaco_validation.json" --gate "$CBWCE_RUN_ROOT/verification/gate.json"
```

Explicitly choose the formal design seeds and do not regenerate after freezing. Prepare creates a queue without starting training: six parents, six WCEs, thirty continuations and 690 scenario tasks for the controllers selected through stopping. `--training-only` prepares just the 42 training jobs. Variables refer to new campaign directories; existing runs are not overwritten.

## Validation, selection and stopping

Controller validation uses eleven pregenerated DR mixtures bound to profile names, each lasting 600 seconds, continuously covering 6600 seconds in fixed order. Reset the network and recurrent memory only at rollout start. Routine validation uses realization zero. Development and formal selection use different manifests/streams; final tests never tune parameters.

The primary score is the full-horizon mean of per-second total queue over controlled incoming lanes: 150 Grid lanes and 116 Monaco lanes, not every lane in the physical network. Lower is better. Metrics are unscaled/unclipped. Validate every 100 episodes. Every strict improvement saves best; only cumulative improvement of at least `max(anchor×0.01,0.1)` resets patience. Episode zero does not count as a non-improvement. After the stage minimum, ten consecutive routine validations without meaningful improvement stop the run, without extra confirmation. Earlier candidates win ties.

Parents have a 1000-episode minimum and 5000 maximum. No meaningful learning gives no_learning; budget exhaustion gives max_budget_reached. Only early_stopped upstream runs automatically release dependants. Early stopping is not proof of convergence.

## Saving and retention

Keep one block summary per 600 simulation seconds (120 control steps) to compare queue lengths across the eleven demand profiles. A continuous run of 1000 episodes appends 11,000 block records and 1000 episode records to two compressed log files. At 5000 episodes the same two files hold 55,000 and 5000 records. Model and optimizer checkpoints follow their own schedule below.

| Artifact | Frequency | Retention |
|---|---|---|
| `manifest.json` | Once for a new run | Record map, controller, seed, configuration and source identity. Resume reads and verifies this file |
| `blocks.jsonl.gz` | Every 600-second demand block | Weights, traffic hash, scheduled vehicles, raw queue and aggregated diagnostics; 11 rows/episode |
| `episodes.jsonl.gz` | Every complete episode | Queue, throughput, elapsed time and steps; trainer RSS is not whole-job memory |
| `validation.jsonl.gz` | Episode 0, every 100 episodes and terminal endpoint | Scores, block diagnostics, candidate hash and stopping diagnostics; reuse an identical endpoint |
| Full recovery | Every 10 episodes, validation, normal segment/end | Latest two; parameters, optimizer, replay, RNG, counters and patience |
| Best | Every strict validation improvement | Current full best and inference snapshot; earlier candidate wins a tie |
| Validation snapshots | Every validation | Latest five lightweight candidates; inference snapshots cannot resume training |
| Stage archive | Stage end | Final state, referenced ancestry, manifests and summaries; verify before archiving |

Routine training does not retain per-second lanes, per-step rewards, episode NPZs or tripinfo. Training completed counters retain raw SUMO arrivals; final evaluation uses TripInfo to separate normal completions from removals, so training throughput summaries are not final completion rates. The first stop signal requests a durable episode boundary; a forced interruption discards the unfinished episode and restarts from full recovery. Attempts distinguish rolled-back records; offline export removes duplicates. `CBWCE_DATA_ROOT` holds shared inputs, `CBWCE_RUN_ROOT` runs, and `CBWCE_CACHE_ROOT` rebuildable caches. A separate disk/NAS provides an independent backup.

## Local multitask execution

Development has 24 independent candidates; after configuration freeze there are six independent parent jobs. Measure distinct classes such as Grid/DQL and Monaco/MA2C before increasing concurrency. A completed parent can release its own dependants without waiting for the other five parents.

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

## Acceptance and code evidence

Check actual state/action sizes, parameter counts, single reward scaling, valid-sample masks, update counts, validation isolation, full recovery and refusal to resume an inference-only snapshot. Deliver six selected parent identities and explicit statuses; report no_learning/budget exhaustion separately.

Sources: [controller](../../agents/controller.py), [DQL](../../agents/dql.py), [environment](../../envs/experiment_env.py), [configuration](../../config/revised/), [tuning](../../experiments/tuning.py), [queue](../../experiments/campaign.py). Chinese: [matching task](01_baseline_training_zh.md).
