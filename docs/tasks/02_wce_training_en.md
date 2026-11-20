# 02 Offline WCE training task

[Local automation and workflow console](05_local_orchestration_en.md)

The console shows episode means of the eleven raw and learner block rewards alongside single-realization validation points. Higher raw WCE reward means a larger mean queue.

> Task specification. Formal training defaults to seed 101; development uses seed 9001. See [implementation evidence](../implementation_status.md). This document does not claim completed formal training.

## Prerequisites and deliverables

Train one offline WCE for each map/controller/seed-101 best parent: six jobs. The formal parent must pass environment/source/input checks, exit successfully with early_stopped status, and supply its full best checkpoint. Bind WCE to its exact parent hash. no_learning/budget exhaustion do not automatically release downstream jobs.

The parent is inference-only: no parameter, optimizer or replay updates. Keep the baseline maps, 6600-second episodes, five-second signal decisions and 600-second demand intervals. WCE selects mixtures of eleven normalized demand profiles; it does not control signals, alter the map or raise the base total rate.

## WCE state, action, reward and network

| Item | Grid | Monaco |
|---|---|---|
| Input | 25 nodes ×6 wave =150 | 28 nodes ×11 padded wave =308 |
| Encoder | 5×5×6; 3×3 CNN32 same →3×3 CNN64 valid →flatten →FC128 | Symmetric self-looped degree-normalized adjacency; GCN64 →GCN64 →node max pooling →FC128 |
| Heads | Gaussian μ11, trainable log_std11, V1; shared encoder | Same |
| Trainable parameters | 95671 | 14679 |
| Reward divisor | 3000 | 1000 |

Wave normalization/clipping matches the controller, with explicit node order and padding. No waiting-time or fingerprint input. Sample `z=μ+std×Normal(0,1)` and set `weights=softmax(z)`: eleven nonnegative values summing to one, bound to profile names/order. Initial log_std is -0.5. This is neither a Dirichlet policy nor an LSTM.

Act every 600 seconds. Reward is the interval mean of per-second network-total controlled-approach queue, maximized by WCE. Divide once at the learner boundary and clip to [-2,2]; evaluation metrics remain raw.

| Learning parameter | Value |
|---|---|
| Optimizer / learning rate | RMSProp / 5e-4, constant |
| Gamma / entropy / value coefficient | 1.0 / 0.01 / 0.5 |
| RMSProp alpha / epsilon | 0.99 / 1e-5 |
| Gradient norm limit | 40 |
| Update frequency | Once after eleven macro transitions, one complete 6600-second episode |
| Advantage normalization | true; population std over eleven samples, epsilon=1e-8; actor advantages only |

Critic returns and reward scaling retain their definitions. Log pre/post normalization mean/std, losses, entropy, gradients and reward clipping. A complete episode has eleven demand actions, one WCE update and zero parent optimizer updates.

## Validation, stopping and execution

At episode zero, every 100 episodes and terminal endpoints, evaluate the frozen WCE candidate with its frozen parent on one fixed exogenous realization (index 0) covering 6600 seconds. Maximize mean queue. **WCE must generate its actions during validation**; the controller's fixed DR sequence cannot replace them. Fix the block/OD/random-type streams and demand constraints; candidate traffic trajectories can differ.

The stage minimum is 500 episodes and the cap is 3000. Ten consecutive routine validations without meaningful improvement trigger early stopping. With validation every 100 episodes and episode zero setting the baseline, the earliest such stop is episode 1000. Save every strict best; meaningful improvement uses max(anchor×0.01,0.1), with earlier candidates winning ties. Report no_learning and budget exhaustion separately. Fixed/online continuation share the same early_stopped best WCE.

```bash
# The queue resolves and pins parent best automatically; standalone execution needs its full path
"$CBWCE_PYTHON" main.py wce --network grid --controller ppo --seed 101 --validation "$CBWCE_DATA_ROOT/grid_validation.json" --gate "$CBWCE_RUN_ROOT/verification/gate.json" --parent "$CBWCE_PARENT_BEST" --output "$CBWCE_RUN_ROOT/manual_grid_ppo_wce"
# Resume in the original directory with identical identity arguments and --resume
```

## Saving and retention

| Artifact | Frequency | Retention |
|---|---|---|
| `manifest.json`, provenance and parent identity | Start | Immutable identity; inputs, runs and caches outside source |
| `blocks.jsonl.gz` | Every 600-second demand block | Weights, traffic hash, scheduled vehicles, raw queue and aggregated diagnostics; 11 rows/episode |
| `episodes.jsonl.gz` | Every complete episode | Queue, throughput, elapsed time and steps; trainer RSS is not whole-job memory |
| `validation.jsonl.gz` | Episode 0, every 100 episodes and terminal endpoint | Scores, block diagnostics, candidate hash and stopping diagnostics; reuse an identical endpoint |
| Full recovery | Every 10 episodes, validation, normal segment/end | Latest two; WCE parameters, optimizer, RNG, counters, patience and frozen-parent sampling state |
| Best | Every strict validation improvement | Current full best and inference snapshot; earlier candidate wins a tie |
| Validation snapshots | Every validation | Latest five lightweight candidates; inference snapshots cannot resume training |
| Stage archive | Stage end | Final state, referenced ancestry, manifests and summaries; verify before archiving |

Routine training does not retain per-second lanes, per-step rewards, episode NPZs or tripinfo. Training completed counters retain raw SUMO arrivals; final evaluation uses TripInfo to separate normal completions from removals, so training throughput summaries are not final completion rates. The first stop signal requests a durable episode boundary; a forced interruption discards the unfinished episode and restarts from full recovery. Attempts distinguish rolled-back records; offline export removes duplicates. `CBWCE_DATA_ROOT` holds shared inputs, `CBWCE_RUN_ROOT` runs, and `CBWCE_CACHE_ROOT` rebuildable caches. A separate disk/NAS provides an independent backup.

Full offline-WCE recovery also saves the frozen parent's policy RNG. Frozen parameters do not mean deterministic actions; omitting this state changes post-resume training traffic. WCE RNG, pending transitions, macro steps and update counters are saved as well. Reference the parent instead of copying it. Fixed and online WCE continuations inherit the offline WCE selected by the highest validation mean queue through stopping.

## Local multitask execution

Each of the six offline WCE tasks depends on its own parent and can start as soon as that parent finishes. Resource measurements include the frozen parent, trainable WCE, SUMO and the extra inference process during validation. Run realization 0 once per validation and parallelize independent training tasks.

Use separate processes, output directories, SUMO connections and random streams. All training and evaluation tasks share one local queue/resource ceiling. Default TF intra/inter and BLAS/OpenMP/MKL thread counts are one, on CPU.

The dashboard workflow admits concurrent workers from startup, with 12 tuning slots and up to 24 formal slots subject to host budgets. See the [workflow guide](05_local_orchestration_en.md). The following measurement procedure applies to the standalone campaign CLI.

Measure one complete process tree first, then compare two/four concurrent measured tasks. An unmeasured resource class runs alone. Offline WCE loads only the parent inference graph, without DQL replay, so this stage does not require full-replay coverage. Reserve at least 20% of total RAM; the sum of measured active peaks multiplied by 1.3 must fit the budget. Check CPU and space for overlapping old/new checkpoints. Disk reserve defaults to 10 GiB and can be increased. RSS/CPU are sampled every five seconds; state transitions persist immediately.

```bash
"$CBWCE_PYTHON" main.py local run --campaign "$CBWCE_CAMPAIGN" --max-parallel 1
"$CBWCE_PYTHON" main.py local status --campaign "$CBWCE_CAMPAIGN"
# Stop the queue, export measurements, then increase concurrency for covered classes
"$CBWCE_PYTHON" main.py local export-resources --campaign "$CBWCE_CAMPAIGN" --output "$CBWCE_RESOURCES"
"$CBWCE_PYTHON" main.py local resume --campaign "$CBWCE_CAMPAIGN" --max-parallel 2 --resources "$CBWCE_RESOURCES"
```

`CBWCE_CAMPAIGN` is the directory containing campaign.json; `CBWCE_RESOURCES` is a new JSON path. Five-second samples may miss brief peaks; 1.3x is operational headroom, not a worst-case guarantee. Short pilot RSS cannot certify complete training and single-realization validation resource use. Local runs have no time segmentation by default; `--segment-seconds` is optional. Normal segments resume automatically, stopping after three attempts without progress. Failures are recorded, not retried indefinitely. File locks reject duplicate run ownership.

## Acceptance and code evidence

Check 150/308 input widths, normalized actions, total-demand conservation, unchanged parent tensors, exactly one update per episode, maximizing selection, a single validation realization at each check and identical next sampling/update after recovery. Deliver six selected WCE identities, parent hashes, histories and explicit stopping statuses.

Sources: [WCE state](../../agents/wce.py), [Gaussian CNN/GCN](../../agents/policies.py), [runner](../../experiments/runner.py), [validation worker](../../experiments/validation.py), [WCE configuration](../../config/revised/). Chinese: [matching task](02_wce_training_zh.md).
