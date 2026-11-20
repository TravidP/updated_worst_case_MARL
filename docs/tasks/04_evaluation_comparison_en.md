# 04 Final evaluation, storage and comparison task

[Local automation and workflow console](05_local_orchestration_en.md)

> Task specification. Formal training defaults to seed 101, retaining ten evaluation realizations per scenario. Chinese: [matching task](04_evaluation_comparison_zh.md). See [implementation evidence](../implementation_status.md); development checks are not formal results.

## Objective, admission and counts

Compare two maps × three controllers × five continuation methods. Each of the thirty branches contributes its lowest-validation-queue checkpoint through stopping. The 23 scenarios and 10 paired realizations yield 690 scenario jobs and 6,900 main rollouts. Prescribed audits are additional.

Models must have finished training and supply a verifiable selected checkpoint and provenance. Formal execution requires the local/container runtime lock; parent/WCE dependency gates still require early_stopped. Load frozen controllers only: no controller/WCE learning and no test-driven checkpoint selection. Initial parents may support development diagnostics; the main baseline comparator is baseline continuation under the same stopping rule.

Retain the earlier scenario semantics but use independent new manifests/streams. Do not mix old IA2C/IQLL results, old single-seed data or artifacts. For each network/scenario/realization, all methods share the traffic artifact, SUMO seed and policy-stream identity. Training seed does not change this evaluation pairing. MA2C/PPO sample using fixed policy seeds; DQL is greedy.

## The 23 scenarios

Every rollout starts empty and covers 3600 seconds with one-second measurements and five-second control, or 720 joint decisions. No warm-up or extra drain period. Vehicles and LSTM memory persist across demand blocks.

| Family | Values / scenarios | Generation rule |
|---|---|---|
| Seen | Eleven original profiles | Each runs for 3600 seconds, normalized to the map's base total |
| Mixture | mixture_1 / 2 / 3 | Generation seeds 41004 / 41005 / 41006 draw Dirichlet(1,…,1) weights bound to eleven profile names; constant for the hour |
| OD redistribution | σ=0.25 / 0.5 / 0.75 | Seeds 41001 / 41002 / 41003 multiply positive Uniform ODs by LogNormal(0,σ), then restore the original total; zero ODs stay zero |
| Switching | 300 / 900 / 1200 seconds | Seeds 41007 / 41008 / 41009 draw from eleven profiles, disallowing adjacent repeats; twelve / four / three blocks |
| Peak | 1.1 / 1.25 / 1.5 | Generation labels 41010 / 41011 / 41012; base [0,1200), multiplier [1200,2400), recovery [2400,3600) |

Mixtures are held-out compositions/interpolations, not strict extrapolation. Peaks change intensity; redistribution preserves OD support; switching changes the ordering of seen demand. The 300-second case is the fastest of these three; 900/1200-second cases are not second-scale incidents. Map profile orders differ: bind weights by names rather than interpreting another map's array positions.

The `test_suite_v1` manifest stores all 23 exact definitions, profile names/order/hash, network/detector hashes and generator hash. Arrival/SUMO/policy seeds derive from manifest hash, scenario ID, realization index 0–9 and stream name. Shared compressed traffic includes a route table, departures and speed factors. Identical inputs are generated once. Strict composition checks prevent changed or missing scenarios masquerading as the registered test.

## Execution steps

1. Freeze training configuration, validation design and selection rules. The formal campaign already writes both test manifests.
2. Check the terminal training status, best inference checkpoint, model hash, source/runtime and validation/test hashes. Record the selected episode and completed episodes so differences in training duration are visible.
3. Partition by model and scenario; run ten realizations sequentially within each task. Atomically save every completed rollout. Record failures; resume only missing entries after validating saved shards.
4. Audit the first seen_Uniform and peak_1.5 realization for every selected model, checking all deterministic metrics against compact evaluation.
5. Generate reports only after all requested cells pass. Missing scenarios, realizations, methods or expected training seeds fail completeness checks.

```bash
# Generate separately only when needed; prefer the manifests already in formal_queue
"$CBWCE_PYTHON" main.py test-manifest --network grid --output "$CBWCE_DATA_ROOT/grid_test.json"
# One scenario; omit --scenario for all 23; add --resume to recover
"$CBWCE_PYTHON" scripts/evaluate_suite.py --run "$CBWCE_SELECTED_RUN" --manifest "$CBWCE_DATA_ROOT/grid_test.json" --selection best --scenario peak_1.5 --output "$CBWCE_RUN_ROOT/manual_eval_peak"
# Collect the complete best-through-stopping evaluation matrix
"$CBWCE_PYTHON" scripts/report.py --campaign "$CBWCE_RUN_ROOT/formal_queue" --output "$CBWCE_RUN_ROOT/report_single_seed"
```

Do not launch manual evaluations alongside a queue to bypass its resource ceiling. Formal batches share the queue. Outputs must be new or explicitly resumed; never overwrite old results. Missing files, corrupted hashes and identity mismatches are not repaired by silently resampling traffic.

## Metrics and storage option B

Primary `mean_queue` is the mean of per-second total controlled-incoming-lane queue over 3600 seconds, in vehicles, lower being better. There are 150 monitored lanes on Grid and 116 on Monaco. This is neither a per-junction mean nor a measure over every physical lane.

| Content | Measurement / write frequency | Storage and requirements |
|---|---|---|
| Mean/integrated/peak/p95 queue and variance | Per-second accumulation, rollout summary | Raw values, unscaled/unclipped; integrated≈3600×mean |
| Last-600-second trend | Linear slope over the final 600 total-queue samples | Vehicles/s; diagnostic only, not model selection |
| Mean speed, scheduled/inserted/arrived/completed/removed, pending/remaining, teleports/collisions | Per-second accumulation, final summary | Save active_vehicle_seconds denominator; completed/inserted is null when inserted=0 |
| Minute curves | Every sixty seconds | Sixty points with interval mean/peak; no duplicate full per-second lane storage |
| Lane windows | Online accumulation | Three disjoint 1200-second windows: pre_peak, peak, post_peak; separate lane_full_mean |
| Completed-trip metrics | Streaming XML parse after closing SUMO | Count/sums/means of duration, waitingTime, timeLoss, departDelay; delete temporary XML after saving |
| Formal rollout shard | Every completed realization | `rows/<scenario>_<index>.json.gz` with model/test/metrics identity and payload hash |
| Completion index | Scenario task completion | `result.json` lists shard SHA-256 values, count and status; manifest records scope |
| Full audit | Uniform / peak_1.5 realization zero per model | Compressed NPZ of per-second lanes and five-second actions/rewards, plus integrity certificate |
| Aggregate report | After matrix acceptance | CSV and summary.json; never replace failures with zero |

Metrics use `evaluation_metrics_v2`. The earlier window zero represented the whole rollout and cannot be labelled pre-peak. New fields explicitly identify window names/bounds and counts [1200,1200,1200]; old caches are invalid. Final evaluation temporarily retains only 3600 float64 network-total queue samples for exact p95, about 28.8KB. Training still avoids a complete per-second lane matrix.

Trip metrics mean actual insertion-to-arrival duration, traffic waiting at speed ≤0.1m/s, time lost relative to ideal driving, and scheduled-to-actual departure delay. Include only completed, non-removed vehicles within the horizon. Save the completed sample denominator; zero completions give null means. Remaining/pending expose unfinished/not-yet-departed vehicles. A lower completed-trip mean does not establish overall improvement when the slowest vehicles remain unfinished. Definitions: [SUMO TripInfo](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html). Preserve the raw SUMO arrival count as arrived: it may include teleport-removal records. completed counts only normal TripInfo completions; report removed separately and use completed/inserted for completion_rate. All TripInfo records with nonnegative arrival must reconcile with raw arrived, otherwise evaluation fails.

Share content-addressed traffic under `CBWCE_DATA_ROOT/<network>/traffic/`. `CBWCE_CACHE_ROOT/final_evaluations/` is rebuildable; canonical shards/audits live under the run root, never only in cache. Model, test, source/runtime or metrics-schema changes invalidate cache identity. Archives retain selected-model references, test manifests, traffic hashes, shards, reports and checksums; verify before backing up to an independent disk/NAS.

Keep recovery checkpoints during formal training. After evaluation of all 30 selected controllers finishes, branches designated for inference only can be archived with their selected inference checkpoint, upstream references, configs and logs. Store identical checkpoint hashes once. Verify the archive and resolve every reference before retiring unreferenced full states and replay for those branches.

## Comparisons and single-training-seed interpretation

Compare within each network and controller; do not combine raw queues from the two maps into one ranking. Compare every method with baseline, plus online with random_group, DR and fixed. Difference is `method−comparator`, negative being better. Relative improvement is `100×(comparator−method)/comparator`; a zero comparator mean gives null.

For ten paired realizations per scenario, report means, sample SDs, differences, relative improvement, wins/ties and a deterministic 10000-resample paired-bootstrap 95% interval. Within a family or seen/test split, weight scenarios equally, aggregate each realization first and then pair. Intervals are conditional on these fixed scenarios and trained models, not confidence over every possible demand distribution.

Currently `n_training_seeds=1`, with `training_seed_sd=null` (blank in CSV), not zero. Do not report a cross-training-seed confidence interval. Ten evaluations describe vehicle/SUMO/policy sampling randomness, not ten independent training experiments. Later campaigns explicitly list additional seeds and require their full matrices before computing cross-seed means and sample SDs.

Outputs are seeds.csv, comparison.csv, paired_differences.csv, paired_training_seeds.csv, auxiliary_metrics.csv and summary.json. The evaluation manifest and seeds.csv record best_episode and completed_episodes for each selected controller. Pool trip metrics using sums/counts across rollouts, retaining completed sample counts. Queue remains primary; inspect throughput, pending and teleports so uninserted/removed vehicles are not mistaken for improvement. No new significance tests or old results-site changes are included by default.

## Local multitask execution

The dashboard workflow admits concurrent workers from startup, with 12 tuning slots and up to 24 formal slots subject to host budgets. See the [workflow guide](05_local_orchestration_en.md). The following measurement procedure applies to the standalone campaign CLI.

Thirty selected models × twenty-three scenarios produce 690 scenario jobs, each sequentially running ten realizations. These jobs share the training resource ceiling. Measure one task, then compare two/four concurrent covered resource classes. Defaults are CPU and one thread per computation library. Evaluation depends on its continuation's terminal selection; repeated evaluations with identical identities can reuse cached results.

Generate/reuse shared traffic and parallelize different models/scenarios. Atomic traffic writes and cache locks prevent concurrent corruption; separate temporary directories and TraCI connections isolate SUMO. Sample tree RSS/CPU every five seconds, reserve at least 20% RAM and apply 1.3x measured-memory headroom. Include temporary tripinfo/audit disk space. Unmeasured classes run alone.

```bash
"$CBWCE_PYTHON" main.py local status --campaign "$CBWCE_RUN_ROOT/formal_queue"
"$CBWCE_PYTHON" main.py local resume --campaign "$CBWCE_RUN_ROOT/formal_queue" --max-parallel 2 --resources "$CBWCE_RESOURCES"
```

Only one scheduler can own a run-root resource lock. Restart checks active workers and exit receipts. Failed evaluation retains completed shards; after diagnosing the failure, explicitly resume its evaluation directory. Do not choose retries based on favourable scores.

## Appendix: temporary blockage proposals, not implemented

| Candidate | Question | Implementation considerations |
|---|---|---|
| Temporary lane / short-edge access restriction | Local capacity loss, detours and spillback | Save/restore permissions; distinguish vehicles already inside from later entrants; permissions do not necessarily stop existing vehicles immediately |
| Brief all-red at a chosen junction | Recovery after loss of signal service | An event layer overrides requested actions, records requested/executed signals and restores normal transitions; prevent the next five-second action overwriting the event |
| Temporary speed / capacity reduction | A milder reversible bottleneck | Define severity separately; speed restriction and hard closure are different experiments |

A first study could use one all-red junction while preserving OD, routes and topology. This is a discussion proposal, not a frozen design. Second-scale events should trigger in the per-second simulation layer independently of the five-second policy cycle. Durations such as 5/10/20 seconds are possible discussion candidates, not registered values. Location, start time, duration, rerouting and recovery criteria remain undecided.

Pair every event version with a no-event version using the same traffic/realizations. Freeze an event manifest and inspect local/network excess queue AUC, peak queue, throughput loss and post-release recovery. Log start/end, affected lanes, actual signals/permissions, pending and teleports. All-red is not physical road closure; rerouting experiments remain separate from the main 23 scenarios.

Technical references: [SUMO road/lane restrictions](https://sumo.dlr.de/docs/Simulation/Rerouter.html#closing_a_lane), [TraCI signal control](https://sumo.dlr.de/docs/TraCI/Change_Traffic_Lights_State.html). Future implementation must verify restoration and vehicle behavior against the locked SUMO version. Do not manufacture improvements by deleting demand, skipping unreachable ODs or increasing teleporting.

## Acceptance and code evidence

Check all scenario definitions, named weights, total rates, random-stream pairing, 3600 continuous samples, sixty minute points, window boundaries, trip denominators, shard hashes, duplicate-free recovery, complete matrices and null single-seed SD. Audit and compact modes must agree on all deterministic metrics. Source/environment/metric changes invalidate old caches.

Sources: [scenarios](../../experiments/scenarios.py), [worker](../../experiments/validation.py), [metrics](../../experiments/metrics.py), [tripinfo](../../experiments/tripinfo.py), [restartable evaluation](../../experiments/evaluation.py), [reporting](../../experiments/reporting.py), [queue](../../experiments/campaign.py).
