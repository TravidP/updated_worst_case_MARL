# Endpoint reward and monitoring integration — 16 September 2026

[English operating guide](TRAINING_MONITORING.md) · [中文操作指南](TRAINING_MONITORING_zh.md)

## Implemented behavior

- Grid and Monaco controller rewards use the queue at the end of each five-second action, without division by 100. IA2C/PPO/IQL use shared negative queue; MA2C retains local and 0.9-weighted neighbor costs. WCE retains positive 600-second mean queue divided by 100.
- Monaco IA2C/MA2C use 120-transition batches. Other learning settings and the 6,600-second training episode remain unchanged.
- Native buffered TensorBoard events include `train/reward_by_learning_step`, full/partial episode metrics and actual learner diagnostics. Raw rewards and step identifiers are retained.
- Parent/continuation monitoring uses separate model and SUMO instances: initialization, every 50 complete episodes, and the final budget. Publication rounds use three paired Uniform 600-second realizations. Monitoring records include CSV, lane NPZ, JSONL, PNG/SVG and TensorBoard detail curves.
- CLI, shell launchers and bilingual local workspace forward monitoring options. Publication settings enforce 50 episodes and three realizations. Historical checkpoint signatures are rejected under protocol 5.

## Preservation

The exact pre-edit snapshot contains 1,366 files. All copies matched the [checksum index](history/endpoint_monitoring_20260916T095506Z/checksums.json). Historical checkpoints, demand and result directories remain in place. No Git commit or push was performed.

## Verification evidence

The [fresh verification directory](../runs_eval/revised/verification/endpoint_monitoring_20260916_v1/) contains deterministic tests, per-case logs and `checks.json` records. The [fresh gate](../runs_eval/revised/verification/endpoint_monitoring_20260916_v1/gate.json) passed all eight cases. The ten deterministic correction tests and fifteen workflow/visualization/monitoring tests passed. Publication launch status is recorded below.

The full matrix exercises both networks and four controllers, all five continuation methods, frozen WCE, checkpoint recovery, exact budgets, C01–C10, monitoring/raw-TensorBoard consistency, resume reuse and monitoring/no-monitoring parameter equivalence. Its pilot interval is one complete episode with one monitoring realization to exercise scheduling efficiently.

Separate [Monaco three-realization evidence](../runs_eval/revised/verification/endpoint_development/monaco_three_rollouts/focused_checks.json) verifies the publication-sized monitoring round, mean/sample-SD calculations, 600 timestamps and plotted data. Short pilots are mechanism checks, not evidence of improved robustness.

Additional checks passed: shell launcher/follow-up tests, Python/JavaScript/shell syntax, local page and asset responses, documentation links, prepared-input consistency and the snapshot checksum audit. This does not claim that every historical script is bug-free.

## Live services and retraining

- Native training TensorBoard: <http://127.0.0.1:6007/> (`runs/revised`).
- Historical exported curves: <http://127.0.0.1:6006/>.
- Local workspace: <http://127.0.0.1:8765/>; [中文](http://127.0.0.1:8765/zh.html).

After a passing gate, launch eight fresh parents sequentially: grid then Monaco, IA2C, MA2C, IQL-LR, PPO; seed 101; 1,000,000 learning steps each. Full offline WCE and continuation campaigns are outside this launch.

Unscaled rewards and larger Monaco batches are requested experimental settings. They do not guarantee better traffic performance. Read fixed-monitoring queues and waiting measurements alongside rewards and learner diagnostics.
