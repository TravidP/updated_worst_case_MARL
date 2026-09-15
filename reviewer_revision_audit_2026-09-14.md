**Historical audit archived on 14 September 2026**

This document preserves the previous repository audit and seven-method proposal. It is superseded for experiment planning by [the English training procedure](reviewer_revision_plan.md) and [the Simplified Chinese training procedure](reviewer_revision_plan_zh.md). Its historical method matrix, workload totals, and proposed corrections must not be treated as the current protocol. The original text follows unchanged.

# Reviewer revision plan: repository audit and experimental procedure

**Repository:** `/home/sdc_joran/Journal/deeprl_signal_control`  
**Review date:** 14 September 2026  
**Document:** `reviewer_revision_plan.md` in the repository root.

**Status:** Documentation only. This file records the repository audit and the approved experimental procedure. Existing code, configurations, datasets, checkpoints, and results have not been changed. No training or SUMO simulations were launched. The implementation and experiments described below are future work; unchecked acceptance criteria are not claims of completed validation.

## 1. Summary and agreed experimental scope

The repository already implements the three stages needed for your framework:

1. Train a traffic signal controller under a predefined demand schedule.
2. Train CB-WCE against that frozen controller.
3. Retrain the controller while updating CB-WCE.

Your revision should retain this structure, but rebuild the comparisons around a common experimental protocol. Several current implementation differences affect fairness beyond the additional training budget.

The agreed revised study will use:

- **Networks:** the synthetic 5×5 grid and the active Monaco subnet.
- **Controllers:** IA2C, MA2C, IQL-LR (`iqll` in the code), and PPO.
- **Objective:** queue-based controller training, CB-WCE reward, and primary evaluation.
- **Replication:** five independently trained starting controllers per network/controller family, with ten evaluation rollouts per test scenario.
- **Comparisons:** seven training methods, including fixed and updated CB-WCE, random demand groups, domain randomization, and a RARL-style comparator.
- **Preservation:** existing source files, checkpoints, datasets, and results remain historical records. Any later implementation should use separate revision code, configurations, datasets, and output directories.

The evidence below comes from source inspection, configuration files, network XML, checkpoint metadata, and evaluation artifacts. All **45 Python source files passed syntax parsing** during the audit. This establishes syntax validity, not runtime correctness or reproducibility of previous results.

## 2. Current repository organization and protocol

### 2.1 Folder and entrypoint map

| Location | Current purpose |
|---|---|
| `agents/` | Controller models, policies, replay buffers, and CNN/GCN adversary implementations. |
| `envs/` | Shared SUMO environment, grid/Monaco environments, adversarial environments, and coevolution environments. |
| `config/` | Controller hyperparameters, network settings, training budgets, and some resume settings. |
| `main.py`, `utils.py` | Standard controller training and the original evaluation workflow. |
| `train_adversary.py`, `train_adversary_real.py` | CB-WCE training against frozen grid and Monaco controllers. |
| `train_coevolution.py`, `train_coevolution_real.py` | Controller and CB-WCE retraining. |
| `eval_signal_controllers.py`, `eval_signal_controllers_real.py` | Per-demand-group benchmark evaluation of original and retrained controllers. |
| `eval_signal_controller_visualize.py` | Individual SUMO GUI rollouts and queue/speed exports. |
| Root plotting and aggregation scripts | TensorBoard extraction, horizon averages, and comparison figures. |
| `large_grid/` | Active 5×5 network, SUMO assets, builders, and generated runtime files. |
| `real_net_subnet/` | Active Monaco subnet, training demand groups, and demand metadata. |
| `data_traffic/` | Active grid demand directory, currently containing twelve valid CSV profiles. |
| `runs/` | Baseline controller checkpoints, copied configurations, and logs. |
| `output_adversary/`, `output_adversary_monaco/` | Grid and Monaco CB-WCE checkpoints and logs. |
| `output_coevolution/`, `output_coevolution_real/` | Retrained controllers and updated adversaries. |
| `runs_eval/` | Benchmark results, raw rollout data, manifests, plots, and later partial reruns. |
| `real_net/`, `small_grid/` | Legacy/full Monaco assets and the older six-signal benchmark. |
| `demand_5x5_noisy/` | Much heavier grid demand variants; these are not mild-noise test profiles. |
| `data_traffic/data_traffic/` | A separate nested demand variant that the active loader does not load recursively. |
| `data_traffic_real/` | A placeholder directory; it is not the active Monaco demand source. |
| `output_result/`, `real_net_experimental_data/`, `figs/` | Older results, tables, and figures. |
| `deeprl_signal_control/` | A nested output log, not another active codebase. |
| README, introduction, handoff note, PDF | Documentation of different ages. The PDF is the original 2019 MA2C paper, not the revised CB-WCE manuscript. |

The README contains unresolved merge markers, and parts of `introduction.md` describe older settings. For the revised paper, executable settings and frozen run manifests should be the source of truth.

### 2.2 Networks and demand settings

| Property | 5×5 grid | Active Monaco subnet |
|---|---:|---:|
| Signal controllers | 25 | 28 |
| Physical signalized junctions | 25 | 30 |
| Noninternal road edges | 120 | 270 |
| Noninternal road lanes | 180 | 513 |
| Unique controlled incoming lanes | 150 | 116 |
| Controller interval | 5 seconds | 5 seconds |
| Configured yellow interval | 2 seconds | 2 seconds |
| Demand-block duration | 600 seconds | 600 seconds |
| Current loaded training groups | **12** | **11** |
| Current effective training horizon | **7,200 seconds** | **6,600 seconds** |
| Joint controller transitions per full episode | **1,440** | **1,320** |

The distinction between current files and historical experiments is important:

- Grid configurations specify a 3,600-second horizon, but the environment replaces it with `loaded_groups × 600`.
- Historical grid CB-WCE logs record **eleven groups**, giving 6,600-second episodes.
- The current grid directory also contains `demand_5x5_sparse.csv`. Its schema is valid, so the current loader includes it and expands the adversary action dimension to twelve.
- Monaco explicitly loads eleven named groups and excludes additional profiles.

This behavior is visible in [the grid loader](envs/large_grid_env.py) around line 73 and [the Monaco loader](envs/real_net_env.py) around line 171.

The eleven established profiles cover eight directional patterns, periphery-to-center, center-to-periphery, and uniform demand. Their current total rates are approximately:

- **Grid:** 2,869.55–3,046.10 vehicles/hour.
- **Monaco:** 2,383.33 vehicles/hour per group.

Baseline training follows a fixed sequence of groups. Vehicle realizations remain stochastic through Poisson counts, sampled departure times, and other injection randomness.

### 2.3 Current training stages

A **controller step** currently means one joint network action followed by five simulated seconds. It does not mean one junction action, one second, or one optimizer update.

| Stage | Current intended setting | Evidence and qualification |
|---|---|---|
| Baseline controller training | `total_step = 1e6` | Stopping is checked at episode boundaries, so saved counts can exceed one million. |
| Offline CB-WCE training | 500 episodes | Historical eleven-group runs record 5,500 adversary decisions and 660,000 frozen-controller transitions. |
| Coevolution | 1,000 episodes | Historical eleven-group grid runs add 1,320,000 controller transitions. |
| Demand action | Every 600 simulated seconds | This is the action frequency; adversary gradient updates depend on the buffer and episode handling. |

Grid CB-WCE uses a **CNN-A2C** policy. Monaco CB-WCE uses **GCN-A2C**. The adversary constructs mixtures of demand groups, rather than simply choosing one group.

The action transformation differs between networks:

- Grid: Gaussian outputs become mixture weights through softmax.
- Monaco: negative outputs are clipped to zero and the remainder normalized, with a uniform fallback.

### 2.4 Current controller hyperparameters

| Controller | Learning rate | Grid batch/rollout length | Monaco batch/rollout length |
|---|---:|---:|---:|
| IA2C | 0.0005 | 120 | 40 |
| MA2C | 0.0005 | 120 | 40 |
| PPO | 0.0003 | 120 | 120 |
| IQL-LR | 0.0001 | 20 | 20 |

Other current settings include:

- Controller discount factor: 0.99.
- IA2C/MA2C LSTM size: 64.
- MA2C neighborhood discount: 0.9.
- PPO clipping ratio: 0.2; four optimization epochs.
- IQL-LR replay capacity: 1,000; ten minibatch updates per backward call.
- IQL exploration: epsilon decreases from 1 to 0.01 over half the configured training horizon.

These settings should be copied into the revision manifest. Changes to reward construction, normalization, and experiment control must be documented separately.

### 2.5 What the saved benchmark manifests show

The following are the checkpoint steps referenced by existing benchmark manifests, not a claim that they are the latest checkpoints anywhere in the repository.

| Network | Controller | Baseline | Retrained | Additional steps |
|---|---|---:|---:|---:|
| Grid | IA2C | 1,000,920 | 2,320,920 | 1,320,000 |
| Grid | MA2C | 1,000,560 | 2,320,560 | 1,320,000 |
| Grid | IQL-LR | 1,000,560 | 2,320,560 | 1,320,000 |
| Grid | PPO | 1,000,560 | 2,320,560 | 1,320,000 |
| Monaco | IA2C | 1,001,040 | 2,321,040 | 1,320,000 |
| Monaco | MA2C | 1,000,560 | 1,858,560 | 858,000 |
| Monaco | IQL-LR | 1,000,560 | 1,648,680 | 648,120 |
| Monaco | PPO | 1,000,920 | 2,309,040 | 1,308,120 |

Other Monaco branches contain different checkpoints, including a later MA2C checkpoint at 2,320,560. Selecting “latest” from a folder is therefore insufficient to establish which experiment produced a paper result.

### 2.6 Current evaluation

The newer benchmark scripts generally evaluate:

- Four controller families, each with original and retrained checkpoints.
- Twelve demand profiles.
- Ten rollouts per controller/profile.
- A 3,600-second horizon.
- One profile repeated across six 600-second blocks.
- Stochastic IA2C/MA2C/PPO actions and greedy IQL actions under the default policy mode.
- Queue and speed recorded every second.

Both benchmark directories contain 192 primary raw queue/speed CSVs, corresponding to eight controllers × twelve profiles × two metrics. However, partial reruns have mixed different sessions in the same directories.

The original `main.py evaluate` workflow is different: it evaluates the environment’s sequential demand program. The manuscript should identify which evaluation workflow was used.

## 3. Reviewer checklist and required corrections

| Reviewer requirement | Current position | Revised procedure |
|---|---|---|
| Record training wall time and CB-WCE overhead | Diagnostic timings exist, but no complete structured cost ledger was found. | Record stage totals, component times, simulation interactions, and hardware. |
| Match controller-training steps | Original/retrained comparisons use unequal budgets. | Give every continuation method exactly the same additional controller transitions. |
| Fixed versus online CB-WCE | No explicit controlled ablation exists. | Branch from the same controller and CB-WCE checkpoints; vary adversary updating only. |
| Random groups and literature baselines | Existing experiments do not provide the complete comparison. | Add categorical random groups, mixture domain randomization, fixed demand, and RARL-style training. |
| Additional unseen demand groups | Current grid loader can ingest newly added CSVs automatically. | Use explicit train/validation/test manifests and physically separate datasets. |
| Align CB-WCE reward and primary metric | Queue definitions, sampling, clipping, and scaling differ. | Use one canonical queue measurement and aggregation. |
| Rollout metrics and uncertainty | Repeated data exist, but summaries mainly report point estimates; bands are min–max. | Export rollout-level outcomes and distinguish rollout variability from training variability. |
| Network heatmaps | Existing GUI and temporal plots do not implement quantitative spatial comparisons. | Record lane/edge measurements and draw maps using common scenarios, windows, and scales. |

### Corrections that must precede long revised experiments

1. **Evaluation seeds are not currently applied as intended.**  
   The two benchmark scripts initialize test seeds but do not set `env.train_mode = False`. Environment resets consequently select training seeds. See [seed selection during reset](envs/env.py) around line 659.

2. **Demand and controller actions share NumPy randomness.**  
   Identical SUMO seeds alone do not produce identical exogenous traffic across controllers. Evaluation must freeze complete demand realizations and separate demand, policy, and simulator random streams.

3. **Coevolution changes controller reward handling.**  
   The standard environment constructs shared or neighborhood-weighted rewards. Coevolution bypasses this path and supplies raw local rewards. Demand selection is therefore not the only difference between current baseline and robust training. See [standard reward handling](envs/env.py) around line 805.

4. **MA2C fingerprints are not updated consistently.**  
   Standard training updates them; the adversarial/co-evolution loops omit equivalent updates.

5. **IQL learning frequency differs.**  
   Standard training updates after a block of twenty transitions. Grid coevolution can update after every transition once replay is populated. Matching environment steps without matching optimizer cadence would leave a substantial confound.

6. **Monaco IQL has incompatible calls in the current source.**  
   Frozen-controller inference uses an actor-critic calling convention, and coevolution assumes replay-buffer attributes that IQL does not expose.

7. **Reward and metric definitions differ.**  
   Grid training uses queue plus waiting time. Monaco training caps lane queues at ten. Evaluation uses uncapped lane halting counts. Monaco offline and online CB-WCE also use different reward scales. See [current reward measurement](envs/env.py) around line 394.

8. **Resume is not complete state restoration.**  
   Existing checkpoints do not establish restoration of optimizer slots, replay state, random streams, and all counters. Some Monaco load failures can fall through to random initialization.

9. **Incomplete evaluations can be hidden by padding.**  
   Short trajectories are padded with their final value; empty trajectories can become zeros. Revised results must flag incomplete rollouts instead.

10. **Output provenance is mixed.**  
    The real benchmark’s current demand manifest lists only `Real_Life_Monaco`, while its summary contains the earlier full comparison. Revised runs need unique directories and immutable manifests.

These are proposed corrections. None has been applied during this review or the creation of this document.

## 4. Revised common experimental contract

### 4.1 Canonical queue metric

Use the **time-averaged total number of stopped vehicles on controlled incoming lanes**.

Let \(L\) be the globally deduplicated set of controlled incoming lanes:

\[
Q(t)=\sum_{\ell\in L}q_\ell(t),
\qquad
J_Q=\frac{1}{H}\sum_{t=1}^{H}Q(t).
\]

Definitions:

- \(q_\ell(t)\): uncapped lane halting count, sampled every simulated second.
- \(H\): the fixed evaluation horizon in seconds.
- \(J_Q\): mean total queue, in vehicles.
- Lower \(J_Q\) is better.

SUMO defines a halted vehicle using speed below 0.1 m/s. [SUMO lane-value documentation](https://sumo.dlr.de/docs/TraCI/Lane_Value_Retrieval.html)

The primary monitored domain contains 150 grid lanes and 116 Monaco lanes in the current network files. Record the exact lane lists and hashes.

Use the manuscript label **“mean total queue on controlled approaches.”** This avoids implying that the metric measures every road lane or directly measures travel delay.

### 4.2 Controller and CB-WCE rewards

For controller training:

1. Accumulate lane queues over all five seconds of each controller transition.
2. Form each junction’s negative mean local queue.
3. Apply the controller family’s established shared/global or MA2C neighborhood reward construction consistently across every training method.
4. Use a fixed reward normalization of 100 and disable reward clipping in the revised configuration.
5. Retain the controller discount of 0.99 and the existing architecture settings.

For a 600-second CB-WCE action block:

\[
r_{\mathrm{WCE},k}
=
\frac{1}{100}
\left[
\frac{1}{600}
\sum_{t\in k}Q(t)
\right].
\]

CB-WCE maximizes this reward. Evaluation minimizes the same unscaled queue measure.

Use adversary discount **1.0** for the finite, eleven-block episode. With equal-duration blocks, maximizing their summed mean queues is equivalent to maximizing the episode’s mean queue. Apply no additional adversary reward normalization or clipping.

The MA2C neighborhood reward remains a learning construction. CB-WCE reward must be calculated directly from canonical measurements, not by summing already transformed controller rewards.

### 4.3 Fixed data and timing definitions

For the revision:

- Explicitly allowlist the original **eleven training groups** on each network.
- Normalize each grid training profile to **3,000 vehicles/hour**.
- Normalize each Monaco training profile to **2,383.3333 vehicles/hour**.
- Preserve within-profile OD proportions.
- Use 600-second demand blocks and 6,600-second training episodes.
- Keep 5-second controller actions and the existing yellow-phase behavior.
- Exclude sparse and additional profiles from the new training manifest.

Normalization is a deliberate revision to the input data. Store the revised copies separately and retain the original CSVs unchanged.

### 4.4 Shared training machinery

All seven methods must use the same controller interaction and update path. Only the demand-selection strategy and whether the adversary learns may differ.

Required behavior includes:

- Correct MA2C fingerprint updates.
- Correct recurrent-state resets.
- Identical reward transformations within a controller family.
- IQL backward calls every twenty newly collected controller transitions.
- Identical PPO epochs and actor-critic update cadence across methods.
- Exact transition counting and partial-batch handling.
- Explicit failure on missing or incompatible checkpoints.
- No silent route dropping or random-model fallback.

For revised CB-WCE, use the same architecture and observation construction across fixed, online, and RARL-style methods:

- Grid: existing CNN structure with consistently ordered local wave/wait features.
- Monaco: GCN with local normalized lane-wave features, fixed padding/masks, and fixed network adjacency; exclude controller-specific fingerprints from adversary inputs.
- Both: eleven Gaussian outputs transformed by softmax into mixture weights.
- Adversary learning rate: 0.0005.
- Adversary update batch: eleven macro transitions, one full episode.

Thus the **updated CB-WCE selects demand every 600 seconds and updates parameters between training episodes**.

### 4.5 Proposed interfaces and records

A later implementation should add a separate revision runner accepting an experiment manifest with:

- Network and controller family.
- Training method and stage.
- Ordered demand manifest.
- Training seed and independent RNG stream identifiers.
- Exact parent controller/adversary checkpoint paths.
- Required controller and adversary transition budgets.
- Reward definition, scale, and monitored lane set.
- Output directory and checkpoint interval.

Checkpoint metadata must record actual counters rather than reconstructing them from filenames or episode indices.

New revision checkpoints should include model parameters, optimizer state, schedules, IQL replay state, and recoverable RNG state. Save regular resumable checkpoints at episode boundaries; an interrupted episode restarts from the last complete checkpoint, with discarded work recorded in the timing ledger. Also save the exact-budget final checkpoint at a declared stage truncation boundary, even when the budget ends mid-episode. Flush pending on-policy samples with the correct bootstrap before saving; all continuation methods then begin a fresh episode with reset environment and recurrent state.

## 5. Detailed training procedure

### Step 1 — Freeze the experiment inputs

Create a new revision workspace containing copies or references with hashes for:

- Source revision and any working-tree changes.
- Network assets.
- Eleven normalized training profiles.
- Controller and CB-WCE configurations.
- Validation/test manifests.
- Runtime and hardware information.

Use the existing `deeprlsc` environment as the starting runtime. It contains Python 3.6.13 and the legacy TensorFlow stack. Record the actual installed versions and SUMO build; the current system reports `1_26_0+0455-77b9dbc222e`.

Do not launch experiments using the default shell’s Python 3.13 environment.

### Step 2 — Validate the revised runner

Before full training, run short correctness checks for all eight network/controller combinations.

These checks must establish:

- Valid route generation.
- Correct queue/reward equality.
- Working MA2C and IQL paths.
- Correct optimizer cadence.
- Exact stopping.
- Checkpoint restoration.
- Reproducible demand and policy streams.

Use pilot seeds outside the publication seed sets. Pilot results are debugging evidence and do not enter the reported comparison.

### Step 3 — Train independent baseline parents

Use five master training seeds:

`101, 202, 303, 404, 505`.

For each network/controller/seed combination:

1. Initialize a fresh controller.
2. Train using the original sequential demand-group scheme, with the revised queue reward and explicit eleven-profile manifest.
3. Stop at exactly **1,000,000 joint controller transitions**.
4. Flush a final partial on-policy batch with correct truncation bootstrapping.
5. Save the complete parent training state.

This produces **40 independent baseline parents**.

For IQL, retain the Stage-I exploration schedule: epsilon reaches 0.01 by 500,000 controller transitions and remains there during continuation.

### Step 4 — Train CB-WCE against each frozen parent

For each parent:

1. Freeze controller parameters and disable all controller learning.
2. Reset recurrent state at every episode.
3. Train one CB-WCE for **5,500 macro transitions**: 500 episodes × eleven demand actions.
4. Count the associated **660,000 frozen-controller transitions** separately.
5. Save the final adversary state and its parent-controller identity.

This produces **40 pretrained CB-WCE models**.

Use the same pretrained adversary as the starting point for fixed CB-WCE, updated CB-WCE, and RARL-style continuation.

### Step 5 — Branch into seven methods

Every method starts from the same parent controller state for its network/controller/seed combination.

Each receives exactly:

\[
B_{\mathrm{additional}}=1{,}320{,}000
\]

controller-learning transitions, giving:

\[
B_{\mathrm{total}}=2{,}320{,}000.
\]

| Method ID | Demand during continuation | Adversary learning |
|---|---|---|
| `nominal_continue` | Original fixed sequential schedule of eleven groups. | None. |
| `fixed_profile` | The normalized Uniform profile throughout every episode. | None. |
| `random_group` | Uniformly sample one of eleven groups every 600 seconds; use one-hot weights. | None. |
| `domain_randomization` | Sample weights from `Dirichlet(1,…,1)` every 600 seconds. | None. |
| `fixed_wce` | Pretrained CB-WCE generates state-dependent mixtures. | Parameters remain frozen. |
| `online_wce` | CB-WCE generates mixtures while the controller learns. | Update after each eleven-block controller-training episode. |
| `rarl_style` | Alternate controller-learning and adversary-learning episodes. | Update only in the adversary-learning episodes. |

**Fixed CB-WCE means fixed parameters, not fixed actions.** Its demand choices can still respond to traffic state. This distinguishes it from the fixed-demand-profile baseline.

The domain-randomization method is a traffic-demand adaptation of simulator randomization. Its randomized variables are mixture weights, rather than physical parameters. [Peng et al., dynamics randomization](https://arxiv.org/abs/1710.06537)

For `rarl_style`, repeat:

1. One 6,600-second episode with the controller learning and adversary parameters frozen.
2. One 6,600-second episode with the controller frozen and the adversary learning.

Run 1,000 such pairs. During frozen-controller episodes, disable optimizer calls, replay insertion, and scheduler advancement. Count the final adversary episode consistently, even though it does not change the evaluated controller.

This is **RARL-style alternating adversarial-demand training**, an adaptation rather than a reproduction of the original disturbance-force experiments. [Pinto et al., Robust Adversarial Reinforcement Learning](https://proceedings.mlr.press/v70/pinto17a.html)

### Step 6 — Enforce comparison fairness

Within each controller family, verify that all seven methods have:

- Identical parent controller state.
- Identical additional controller-learning transitions.
- Identical controller optimizer-call schedule and PPO epochs.
- Identical reward handling.
- Identical checkpoint cadence.
- Identical training horizon and allowed demand-group set.

Fixed and updated CB-WCE also share the same initial adversary, observation structure, action transformation, and sampling rule.

RARL-style training has additional frozen-controller simulation. Report it explicitly; equal controller-training budgets do not imply equal total simulation or wall-clock budgets.

### Step 7 — Record computation cost

Use a monotonic timer and structured records for:

- Baseline pretraining.
- Offline CB-WCE training.
- Controller continuation.
- Adversary-only simulation.
- Environment startup/reset.
- Controller and adversary inference.
- Controller and adversary optimization.
- Checkpoint/logging overhead.
- Failed or discarded attempts.

Report both stage times and total pipeline time:

\[
T_{\mathrm{method}}
=
T_{\mathrm{baseline}}
+
T_{\mathrm{offline\ WCE,\ if\ used}}
+
T_{\mathrm{continuation}}.
\]

Charge offline CB-WCE training to every method that requires it when reporting standalone method cost, even when the actual experiment shares that checkpoint.

Report hardware, thread counts, concurrent workload, and simulation interactions alongside wall time. No reliable full-run hour estimate can be derived from the current logs alone.

The agreed matrix contains **40 baseline runs, 40 offline CB-WCE runs, and 280 continuation runs**.

## 6. Unseen-demand generation and evaluation procedure

### 6.1 Separate three evaluation categories

Maintain distinct results for:

1. **Seen profiles, new realizations:** the eleven training demand profiles with held-out arrival and simulator seeds.
2. **New demand profiles/schedules:** a frozen twelve-scenario test suite.
3. **Supplementary historical profiles:** existing sparse/noisy/reconstructed profiles whose provenance or severity needs separate explanation.

Never place validation or test CSVs in the current automatically loaded training directory.

### 6.2 Default twelve-scenario test suite per network

Use four families, three scenarios each:

| Family | Prespecified construction | Interpretation |
|---|---|---|
| OD-rate redistribution | Multiply positive rates of the Uniform profile by seeded lognormal factors with log-space standard deviations 0.25, 0.50, and 0.75; renormalize to reference demand. | New spatial rate allocations at fixed total load. |
| Convex mixtures | Generate three fixed `Dirichlet(1,…,1)` weight vectors using held-out generation seeds. | Compositional generalization within the mixture family. |
| Temporal changes | Alternate N→S and W→E demand every 300, 900, or 1,200 seconds across the 3,600-second horizon. | Generalization to new switching schedules. |
| Peak intensity | Apply multipliers 1.10, 1.25, or 1.50 to Uniform demand during seconds 1,200–2,400; use reference demand outside that interval. | Intensity extrapolation beyond the training rate. |

Use test generation seeds `41001–41012`, assigned in table order.

Convex mixtures are already within the demand family accessible to CB-WCE. Describe them as compositional tests, not automatically as out-of-distribution demand. Likewise, new rate allocations do not necessarily introduce new OD pairs.

Create six separate validation scenarios using generation seeds `31001–31006`: two rate redistributions, two mixtures, one temporal switch scenario, and one 1.15× peak scenario. Validation and final-test artifacts must have distinct hashes.

### 6.3 Demand validation and provenance

For every generated scenario:

1. Validate finite, nonnegative rates.
2. Check edge existence and vehicle-class-compatible route connectivity.
3. Record requested total demand and its temporal schedule.
4. Store the generation parameters and seed.
5. Materialize vehicle counts, departure times, OD pairs, route edge sequences, and speed factors.
6. Hash the completed artifact.

Keep the same complete exogenous realization across all methods. Realized insertion may differ because congestion differs; record that difference rather than changing the requested traffic.

Do not discard a valid high-demand test because it produces congestion or gridlock.

The current `Real_Life_Monaco` profile should remain outside the main test suite. Its audit refers to a 272-edge patched network, while the active subnet contains 270 edges. A static connectivity check also identified an unreachable positive-demand OD pair. Resolve this provenance mismatch before any supplementary use; do not silently drop the affected flow.

The existing `demand_5x5_noisy` profiles total approximately 13,787–17,379 vehicles/hour. Treat them as extreme-load scenarios, not ordinary noise robustness tests.

### 6.4 Evaluation execution

For each final controller checkpoint:

- Evaluate all eleven seen profiles and twelve new scenarios.
- Use ten paired rollout realizations per scenario.
- Keep the horizon at **3,600 seconds**.
- Start from an empty network.
- Include startup in the primary average.
- Use no drainage extension in the primary metric.
- Retain sampled IA2C/MA2C/PPO actions and greedy IQL actions.

Use separate seed namespaces for demand generation, arrival realizations, SUMO, and policy sampling. For example, reserve `51001–51010` for arrival realizations, `61001–61010` for SUMO, and a documented training-seed/rollout-derived policy stream.

Explicitly set evaluation mode and record the effective SUMO seed.

Evaluate the checkpoint saved at the exact final budget. Do not select a checkpoint using final-test performance.

The primary seven-method matrix requires **64,400 evaluation rollouts** across the two networks: 2 networks × 4 controller families × 5 training seeds × 7 continuation methods × 23 scenarios × 10 rollouts. This count covers final continuation checkpoints; the one-million-step parent checkpoints are not an additional evaluated method. Report seen and new-scenario outcomes separately.

### 6.5 Required rollout outputs

Write one summary row per rollout containing:

- Network, controller, method, training seed, and exact checkpoint hash.
- Scenario, split, demand hash, and all RNG identifiers.
- Intended and completed horizon.
- Mean queue, integrated queue, and peak queue.
- Mean speed, with its aggregation definition.
- Scheduled, inserted, and completed vehicle counts.
- Pending departures and vehicles remaining at the horizon.
- Teleports, collisions, and execution failures.
- Completed-trip travel time, waiting time, time loss, and departure delay, with denominators.
- Evaluation wall time.

Retain underlying time series and trip records.

For speed, use a vehicle-time-weighted mean; an entirely empty rollout has an unavailable speed metric rather than an invented zero-speed observation.

A valid rollout must contain the expected horizon. Do not pad truncated records into apparently complete results.

### 6.6 Statistical reporting

For each training seed and scenario:

1. Calculate \(J_Q\) separately for each of ten rollouts.
2. Report the mean and **sample standard deviation** of those ten outcomes.

For each controller family, method, and network separately:

1. Average the scenario estimates within each training seed, giving equal weight to the twelve new scenarios.
2. Obtain five seed-level values.
3. Report their mean, sample SD, and 95% t-interval:

\[
\bar J \pm 2.776\,\frac{s}{\sqrt{5}}.
\]

For method comparisons, first form differences between matched training-seed results, then calculate the interval over the five paired differences.

This separates rollout variability from independent-training variability. Time samples within a rollout are not independent experimental replicates. Interval reporting is supported by the RL evaluation recommendations of [Agarwal et al.](https://arxiv.org/abs/2108.13264)

Also report:

- All five training-seed points.
- Separate results for each demand family.
- Seen-profile performance.
- The exploratory **worst tested scenario mean**, defined as the maximum scenario mean after averaging over training seeds and rollouts.

Keep grid and Monaco results separate. The confidence interval is conditional on the specified test suite; it is not a guarantee over all possible traffic demands.

### 6.7 Heatmaps

For the predefined 1.25× and 1.50× peak scenarios:

1. Record per-lane queue measurements during every rollout.
2. Aggregate over the common peak window **1,200–2,400 seconds**.
3. Map monitored lanes onto the SUMO network geometry.
4. Produce absolute queue maps for all methods.
5. Produce `online_wce − comparator` difference maps.

Use:

- One shared absolute color scale within each network/scenario.
- A symmetric, zero-centered difference scale.
- Consistent spatial extent, labels, and units.
- Gray for unmonitored roads.
- Means across the same training seeds and rollout realizations.

Caption these as queue maps of controlled approaches across the network. Negative difference values indicate lower queues under updated CB-WCE.

Do not choose a different peak time or demand scenario for each controller.

## 7. Deliverables and acceptance criteria

### Planned document and experiment outputs

Use a separate revision structure when implementation is undertaken. Only the Markdown document has been created at this stage:

```text
reviewer_revision_plan.md
revision/
  configs/
  datasets/
    train/
    validation/
    test/
  manifests/
  runs/
  evaluation/
  figures/
  tables/
```

The reporting package should contain:

- Current-versus-revised settings table.
- Eight-item reviewer response matrix.
- Exact training budgets and computation-cost table.
- Seven-method comparison tables.
- Fixed-versus-updated CB-WCE ablation.
- Seen/new-demand results with both uncertainty levels.
- Peak-demand heatmaps and paired difference maps.
- Frozen run, checkpoint, network, and demand manifests.

### Acceptance tests before publication runs

- [ ] Every method stops at exactly the prescribed controller-learning budget.
- [ ] Controller optimizer cadence is identical across methods within each family.
- [ ] CB-WCE reward can be reconstructed from the recorded per-second queue measurements.
- [ ] Queue lanes are counted once, without the existing Monaco cap.
- [ ] Fixed CB-WCE weights remain unchanged; updated CB-WCE weights change.
- [ ] Frozen-controller stages do not update weights, replay, or schedules.
- [ ] MA2C fingerprints and recurrent-state resets behave correctly.
- [ ] Both Monaco IQL paths execute correctly.
- [ ] Repeated seeded evaluation reproduces the same demand realization and policy sampling.
- [ ] Changing the controller does not change scheduled demand.
- [ ] Missing checkpoints, invalid routes, and incomplete rollouts fail visibly.
- [ ] Training manifests contain no validation/test profiles.
- [ ] Checkpoint restoration preserves the state required for continuation.
- [ ] Statistical summaries use rollout and training-seed units correctly.
- [ ] Heatmap totals reconcile with the primary metric over the same lanes and window.
- [ ] Every output directory’s manifest matches the results actually stored there.

The existing results remain useful historical evidence and may support qualified descriptive summaries. The revised matched-budget and generalization claims should come from the new, controlled experiment set described above.
