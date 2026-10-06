# Endpoint rewards and native training monitoring — protocol 5

[简体中文](TRAINING_MONITORING_zh.md) · [Full protocol](../reviewer_revision_plan.md)

## Effective training settings

Grid and Monaco use endpoint controller rewards: after each five-second action, IA2C/PPO/IQL receive `-Q(t+5)`. MA2C receives negative local endpoint queue plus 0.9-weighted negative neighbor endpoint queues. There is **no controller reward normalization or clipping**. WCE remains `mean(Q over 600 seconds)/100`. Evaluation averages the uncapped per-second queues. These share the measurement source but intentionally use different aggregation/scaling.

IA2C, MA2C and PPO use 120-transition batches on both networks. IA2C/MA2C collect ordered transitions; PPO reuses its rollout for four epochs. IQL retains minibatches of 20, replay capacity 1,000, and ten minibatch updates per agent every twenty learning transitions. Other parameters are unchanged. Monaco IA2C/MA2C now update one-third as often as with the historical 40-transition batches.

Training episodes still last **6,600 seconds**, not 600. They contain eleven 600-second demand blocks and 1,320 joint controller transitions. Unscaled rewards are an experimental choice, not proof that value underfitting is solved.

## Commands

Use the `deeprlsc` environment from the repository root. Pilot seed 9001 and publication seed 101 remain separate.

```bash
conda activate deeprlsc
# A short mechanism check; periodic interval shortened for this pilot only.
python main.py experiment parent --network monaco --controller ma2c \
  --seed 9001 --pilot --visualization --steps 160 \
  --monitor-every 1 --monitor-rollouts 1

# The gate is written only if all checks pass. Do not edit source/configs during this command.
CBWCE_VERIFY_DIR="$PWD/runs_eval/revised/verification/endpoint_$(date +%Y%m%d_%H%M%S)"
python main.py experiment verify --workers 4 --output "$CBWCE_VERIFY_DIR" \
  && export CBWCE_GATE="$CBWCE_VERIFY_DIR/gate.json"

# Eight fresh parent jobs, sequentially. No WCE/continuation campaign is launched.
unset CBWCE_RESUME CBWCE_STEPS CBWCE_EPISODES
bash scripts/training/08_all_parents.sh --mode publication --seed 101 \
  --gate "$CBWCE_GATE" --no-visualization

# New native events; the historical exporter may still use port 6006.
tensorboard --logdir runs/revised --host 127.0.0.1 --port 6007
```

Publication settings require `--monitor-every 50 --monitor-rollouts 3` (the defaults). Shell wrappers also accept `CBWCE_MONITOR_EVERY` and `CBWCE_MONITOR_ROLLOUTS`; command-line arguments take precedence. Monitoring is observational: it records paired rollouts, TensorBoard metrics and checkpoints but does not automatically terminate a run when performance degrades relative to the initial monitor. Final quality decisions use the complete paired evaluation. Resume with the exact selected new-protocol checkpoint and the same monitoring settings. Existing reward-protocol checkpoints are rejected for continuation; do not select by newest filename.

## Which TensorBoard plots to read

| Tag | Horizontal axis | Meaning and direction |
|---|---|---|
| `train/reward_by_learning_step` | Actual cumulative learning step | One actual reward per transition; **closer to zero is better**. Shared reward once for IA2C/PPO/IQL; agent-mean neighborhood reward for MA2C. |
| `train/episode/mean_total_queue` | Learning step at episode end | Average over all 6,600 seconds; **lower is better**. |
| `train/episode/mean_reward`, `reward_sum` | Learning step at episode end | Actual episode reward aggregates; reward sum depends on duration. |
| `train/partial_episode/*` | Learning step | Final budget-truncated episode, with its actual duration; not mixed into full-episode curves. |
| `block/mean_total_queue_recent_600_seconds` | Learning step; WCE macro step in frozen stages | Recent demand-block diagnostic only. |
| `learner/*/{mean,min,max}` | Learning step | Actual update diagnostics across agents/minibatches, excluding padding. |
| `monitor/mean_total_queue`, `monitor/mean_current_wait_seconds` | Learning step | Fixed Uniform monitoring; lower is better. `_sd` is sample SD across realizations, not a confidence interval. |
| `monitor/queue_by_second`, `monitor/current_wait_by_second` | Simulation seconds 1–600 | Per-round detail runs under `monitoring/round_*/tensorboard/`. |
| `monitor/queue_waiting` | Learning step | Image tab: within-test plots with mean ± sample SD. |
| `wce/*`, `wce_episode/*` | WCE macro transitions | WCE rewards/losses and episode statistics; no controller-learning reward points are written. |

Learner diagnostics include actor/value loss, entropy, predictions, targets, advantages, pre-clipping actor/critic gradient norms and clipping factor; PPO adds clipping fraction. IQL reports TD loss, Q predictions/targets, gradient norm, epsilon and replay occupancy. Raw agent/minibatch records remain in `learner_metrics.jsonl`. TensorBoard may downsample its display; raw records retain all actual values. Events are buffered and flushed periodically and at episode boundaries, not synchronously on every step.

Reward magnitudes are not directly comparable across controller families because MA2C uses neighborhood rewards. Use the canonical mean total queue to compare traffic performance. Monitoring SUMO runs are headless; `--visualization` controls the training environment.

During adversarial continuation, training demand can become harder. A worse training reward alone does not prove worse control; read the fixed Uniform monitoring curves alongside the training curves. These short monitoring tests do not replace final generalization evaluation.

Loss and gradient curves diagnose learning mechanics; a lower optimizer loss alone does not establish better traffic control. Prefer lower fixed-monitoring queue together with completed/pending vehicle counts when assessing progress.

## Fixed monitoring tests and saved data

For parent and all five continuation methods, monitor before learning, every 50 complete episodes, and at the final budget. Three sequential 600-second Uniform tests start empty with no warm-up. Grid uses 3,000 vehicles/hour; Monaco uses 2,383.3333. Arrival seeds are 53001–53003, SUMO seeds 63001–63003, and policy seeds 73001–73003. Traffic artifacts are cached by network/profile hashes under `runs_eval/revised/monitoring_inputs/`, independent of final evaluation data.

Frozen model instances and separate SUMO connections isolate monitoring from training. Monitoring does not advance learning counters, replay, schedules or training random streams. Start/middle/end checkpoints are not selected by monitoring performance. A failed test stops the campaign after saving the training checkpoint and creates a failed attempt record; it never becomes a valid zero-queue result.

Each run keeps `tensorboard/`, `episode_metrics.jsonl`, `learner_metrics.jsonl`, and `monitoring/round_<episode>/`. Each round stores an exact checkpoint reference, three `rollout_*/timeseries.csv` files, JSONL raw records, lane-level NPZ, `summary.json`, and PNG/SVG plots. Completed monitoring rounds are reused on resume; failed rounds are retried in the new attempt.

Each CSV has exactly 600 rows. `queue` is the stopped count on unique controlled incoming lanes (150 grid; 116 Monaco). `current_wait_mean_seconds` is mean current consecutive waiting time among vehicles presently on those lanes, including moving vehicles in the denominator. Empty monitored lanes use zero with `monitored_vehicles=0`. It is not completed-trip waiting time. `current_wait_sum_vehicle_seconds` sums ongoing waiting spells; `cumulative_stopped_vehicle_seconds` integrates queue over time and retains past contributions.

`rollout.npz` contains `time`, `lanes`, and time-by-lane `queue`; `waiting.npz` contains matching `time`, `lanes`, and `current_wait`. Control JSONL records actual reward vectors, cumulative `learning_steps`, stage simulation steps, and whether learning was enabled. TensorBoard plots use the same records. Test timing is charged separately under `monitoring`; it is included in total pipeline wall time.

## Compatibility and evidence

Controller signatures use `learner_boundary_scaled_v3`; WCE signatures include protocol version 6. Raw `-queue` remains in control and traffic records, while learner rewards are scaled only at the learner boundary. Exact pre-edit snapshots and their checksum index are preserved under `docs/history/endpoint_monitoring_20260916T095506Z/`. Historical checkpoints/results are retained at their original locations and are intentionally incompatible with the new training workflow.

The new verification runs reward fixtures, TensorBoard/raw-data comparisons, monitoring/no-monitoring learning equivalence, frozen-state checks, resume reuse, and C01–C10 across both networks and all four controller families. The authoritative pass/fail state is the newly generated gate, not an older historical report. The immediate full campaign consists of eight parents only; WCE and the forty continuations are available for later explicit launches.

The fresh full eight-case verification passed. See the [new gate](../runs_eval/revised/verification/endpoint_monitoring_20260916_v1/gate.json) and [implementation/campaign status report](ENDPOINT_MONITORING_REPORT.md).
