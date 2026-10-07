# Hangzhou and Monaco external demand: location, evaluation, and website integration

Date: 2026-10-06. This is an implementation plan. File discovery and static checks are complete; no training, formal evaluation rollouts, or website changes were performed. [Chinese plan](external_test_plan_zh.md), [machine-readable audit](external_demand_audit.json).

## 1. Located inputs and readiness

| Item | Grid / candidate Hangzhou profile | Monaco / reconstructed MoST profile |
|---|---|---|
| Demand | [data_traffic/demand_5x5_sparse.csv](../../data_traffic/demand_5x5_sparse.csv) | [real_net_subnet/demand_groups/Real_Life_Monaco.csv](../../real_net_subnet/demand_groups/Real_Life_Monaco.csv) |
| Legacy Group 12 entry | [eval_signal_controllers.py](../../eval_signal_controllers.py), final profile | [eval_signal_controllers_real.py](../../eval_signal_controllers_real.py), final profile |
| Local original source | No original Hangzhou 4×4 input or reproducible mapping identified | [real_net/data/in/most_0.rou.xml](../../real_net/data/in/most_0.rou.xml) |
| Active network | [large_grid/data/exp.net.xml](../../large_grid/data/exp.net.xml) | [real_net_subnet/data/in/most.net.xml](../../real_net_subnet/data/in/most.net.xml) |
| Positive OD pairs | 140 | 14 |
| Total demand | 2,983 veh/h | 2,383.333332 veh/h |
| Static directed connectivity | 140/140 reachable | 13/14 reachable |
| Required before formal testing | Verify provenance and live SUMO routes | Resolve route and network provenance mismatch |

The manuscript statement is in [paper/main.tex](../../paper/main.tex), line 591. The legacy list identifies the Grid CSV as the local Group 12 candidate, but neither its filename nor its total flow proves Hangzhou derivation.

The authors' CoLight repository contains [data/Hangzhou/4_4](https://github.com/wingsweihua/colight/tree/master/data/Hangzhou/4_4), including `roadnet_4_4.json` and flow variants `anon_4_4_hangzhou_real.json`, `anon_4_4_hangzhou_real_5734.json`, and `anon_4_4_hangzhou_real_5816.json`. Identify the exact input version before claiming replication. The Monaco upstream is the authors' [MoSTScenario repository](https://github.com/lcodeca/MoSTScenario); the local extraction still requires upstream version and transformation provenance.

The revised main suite defines 11 seen and 12 generated test scenarios. These legacy Group 12 inputs are not explicit external scenarios in that suite. The manuscript's 11+1 protocol and the current 11+12 suite need separate descriptions.

## 2. Dataset-specific preparation

### Grid

First recover the historical source and mapping from backups or experiment records. Reproduce and compare the CSV SHA-256: `1a2bdc51cf5656cf7ea348bdf8fee0271345feeac9f9ee732de02500ab71b9c0`.

If the mapping cannot be recovered, create a newly versioned pipeline: pin upstream commit and input hashes; extract boundary OD, direction, and timing; normalize boundary coordinates between layouts; export an explicit edge mapping with documented splits/merges; specify conservation and scaling rules before evaluation; validate every positive OD using live SUMO routing; freeze conversion code and mapping/artifact manifests. Keep the existing 5×5 network unchanged. This proposed mapping cannot retrospectively establish the manuscript's original procedure.

Until provenance is verified, label the existing input `Grid sparse external candidate`. First test its native 2,983 veh/h demand. Optionally add a separately named 3,000 veh/h version using `3000/2983 ≈ 1.00570`, to distinguish spatial shift from load differences.

### Monaco

Inspect [scenario_metadata.json](../../real_net_subnet/demand_groups/scenario_metadata.json), [audit summary](../../real_net_subnet/demand_groups/Real_Life_Monaco_audit_summary.json), and [flow audit](../../real_net_subnet/demand_groups/Real_Life_Monaco_route_audit.csv). Historical auditing used a 272-edge set patched with `10180#0` and `10180#1`; the active subnet has 270 noninternal edges.

The current directed connection graph cannot route `-10051#2 → 10043`, carrying 108.333333 veh/h, approximately 4.55% of demand. Historical `88/88 ok` therefore does not certify the active network. This independently confirms the issue noted in [the earlier revision audit](../../reviewer_revision_audit_2026-09-14.md).

Prefer a physically justified demand projection onto the unchanged active network, recording changed endpoints and preserving traffic mass and timing. If no valid projection exists, do not silently remove the flow. A separately repaired network requires checking junctions, lanes, phases, observation dimensions, and training compatible models. Existing checkpoint signatures include network asset hashes; bypassing compatibility checks would invalidate the frozen-model comparison.

Distinguish `monaco_most_od_stationary_v1` (14 aggregate OD rates, stochastic 3600 s realization) from `monaco_most_temporal_v1` (88 original flows with timing and route constraints preserved). The local original flows cover 0–3300 s, while the CSV aggregates contributions over a 3600 s denominator. Stationary resampling changes timing and potentially routes. A temporal version should retain the final 3300–3600 s interval without added source flows, while keeping a 3600 s evaluation horizon.

## 3. Independent evaluation matrix

| Parameter | Specification |
|---|---|
| Controllers | IA2C, MA2C, IQLL, PPO |
| Methods | baseline, random_group, domain_randomization, fixed_wce, online_wce |
| Checkpoints | Exact `parent` from `runs_eval/revised/publication_seed101_v1/<network>/<controller>/<method>/suite.json` |
| Training seed and budget | Seed 101; existing final models with 2,320,000 learning steps |
| Evaluation | Frozen policy, empty network, 3600 s, 5 s control, 1 s logs, 720 decisions |
| Policy behavior | Preserve sampled IA2C/MA2C/PPO actions and greedy IQL actions |
| Arrival / SUMO seeds | 51001–51010 / 61001–61010 |
| Policy seeds | `int(digest(['evaluation-policy',101,i])[:8],16)`, i=0..9 |
| Pairing | Identical complete vehicle artifact across all 20 combinations within each network/scenario/rollout |
| Runtime | Match originating Python, TensorFlow, and SUMO versions and record discrepancies |

One scenario per network requires `4 × 5 × 10 = 200` rollouts; the first two scenarios total **400**. Each optional normalized or temporal scenario adds 200. Grid can complete independently while Monaco awaits route validation. Never fine-tune on external demands or use their results to select checkpoints.

Audit all selected training manifests for held-out status. Revised inputs explicitly contain eleven profiles; legacy Grid loading could include sparse demand, so legacy results are not automatically valid unseen-distribution evidence.

## 4. Execution and analysis

Create a separate external manifest containing campaign, network, scenario, external split, family, source files/hashes, mapping version, network hash, rate and temporal schedule, routing validation, horizon, arrival seed, and artifact hash. The completed static check does not validate passenger lane permissions or live TraCI routes. These remain mandatory before formal materialization.

The current [scenario implementation](../../experiments/scenarios.py) and CLI support seen/test/validation, without an external suite. Add an isolated supplementary materializer/orchestrator that reuses [demand materialization](../../experiments/demand.py) and preserves frozen main training definitions. This script is proposed and has not been implemented.

Generate ten complete artifacts per scenario before evaluations, including departures, complete route edges, and speed factors. Perform route-loading smoke checks and at least one full controller evaluation per family. Keep smoke results separate, preserve failed attempts, and never rerun selectively because of poor performance.

Proposed directory:

```text
runs_eval/revised/external_group12_seed101_v1/
  campaign.json
  artifacts/<network>/<scenario>_<arrival_seed>.json
  <network>/<controller>/<method>/external/<scenario>/rollout_01/attempt_001/
```

The current single-rollout CLI accepts a complete external artifact. After validation, replace the placeholders in this template; do not add the unsupported `--suite external`:

```bash
python main.py experiment evaluate \
  --network grid --controller ppo --method online_wce --seed 101 \
  --parent <EXACT_FINAL_CHECKPOINT_FROM_SUITE_JSON> \
  --artifact <VALIDATED_COMPLETE_EXTERNAL_ARTIFACT_JSON> \
  --sumo-seed 61001 --policy-seed <POLICY_SEED_FOR_INDEX_0> \
  --output <NEW_EXCLUSIVE_ROLLOUT_ATTEMPT_DIRECTORY>
```

Substitute Monaco's network, checkpoint, and artifact for its campaign. The current evaluate branch does not require the publication training gate, but checkpoint provenance, integrity, signatures, and final learning steps remain enforced. The orchestrator should preflight them and record source hashes.

Use mean network queue over all 3600 s as the primary metric. Also report integrated/peak queue, speed, inserted/completed vehicles, remaining/pending demand, teleports, and collisions. Preserve undefined speed as missing. Report ten-rollout mean/SD, paired differences with confidence intervals, and win counts against baseline and between online/fixed WCE. Improvement is `100 × (baseline-method)/baseline`; undefined if baseline is zero. Apply multiple-comparison handling if formal significance tests are used.

Interpret queue together with throughput and uninserted/teleported vehicles. Analyze networks and native/normalized or stationary/temporal scenarios separately. Ten rollouts of one training seed quantify evaluation randomness, not robustness to training seeds. Broader algorithm claims require additional complete training seeds with matched budgets.

## 5. Results website integration

The current [exporter](../../docs/evaluation_workbook/grid_results_site/export_network_data.py) hardcodes 4600 rollouts, 23 scenarios, 460 groups, and seen/test splits per network. The [frontend](../../docs/evaluation_workbook/grid_results_site/dist/app.js) also requires 460 summary rows and hardcodes coverage text. Copying 200 new runs into the main tree or using the existing input override alone will fail these checks.

Add an **Evaluation set** selector (`Main v7` / `External Group 12`) independently of the existing network selector. Derive allowed splits and expected counts from campaign/catalog metadata, preserving strict completeness and pairing validation. Export supplementary catalogs, metrics, rollout tables, and time series into:

```text
dist/data/supplementary/group12_v1/grid/
dist/data/supplementary/group12_v1/monaco/
```

Registry entries should include campaign, network, base, expected rollouts/groups, provenance, and availability. Initial supplementary counts are 200 rollouts, 20 groups, and one scenario per network. Main counts remain 4600/460/23. Combined completed coverage would be 9600 rollouts, explicitly separated by evaluation set.

Validate all 20 combinations, ten distinct paired seeds, continuous 3600 timestamps, NPZ shapes, finite nonnegative queues, demand hashes, SUMO/policy pairing, checkpoint hashes, route/network versions, and source provenance. Export to a temporary directory, validate, then update the registry. Grid can publish separately; Monaco can remain `pending route validation`. Failed attempts remain auditable and excluded from completed averages.

Browser acceptance checks: evaluation-set/network switching, external scenario menu, 20 curves, ten-run statistics, CSV downloads, bilingual labels, source disclosures, independent URL state, and unchanged main data hashes/values. Verify the directory served on port 8878 before updating assets, since it may serve an older snapshot.

## 6. Recommended sequence and manuscript wording

1. Complete Grid provenance and live routing; evaluate its native profile (200 runs).
2. Resolve Monaco projection/network consistency; evaluate stationary reconstructed OD (200 runs).
3. Extend the isolated exporter and evaluation-set UI; import each accepted campaign independently.
4. Optionally add normalized Grid and temporally preserved Monaco (200 additional runs each).
5. Describe main 11 seen + 12 generated tests separately from supplementary external-demand tests.

Until verified, describe Grid as `an external sparse OD profile`. With complete source/mapping evidence, use `a Hangzhou-derived demand profile mapped to the 5×5 grid`. For the Monaco CSV use `an OD demand profile reconstructed from the local MoST route file`; for preserved segments use `a temporally preserved MoST-derived demand scenario`. Match the claim to the actual transformation and avoid describing aggregate-rate resampling as unmodified trajectory replay.
