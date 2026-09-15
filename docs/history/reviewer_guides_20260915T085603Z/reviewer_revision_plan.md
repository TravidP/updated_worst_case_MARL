# CB-WCE training and evaluation procedure

[简体中文版](reviewer_revision_plan_zh.md) · [Historical repository audit](reviewer_revision_audit_2026-09-14.md)

**Protocol version:** 3

**Date:** 14 September 2026

**Repository:** `/home/sdc_joran/Journal/deeprl_signal_control`

**Implementation status:** This guide defines the revised study. The corrected implementation is integrated into `agents/`, `envs/`, and `experiments/`. Use `python main.py experiment ...`; see the [project guide](README.md). C01–C10 have passed the documented pilot checks for that path. Existing legacy entrypoints remain unchanged. The full publication training and evaluation matrix has not been launched.

## 1 Experiment overview

We will compare one baseline with four demand-training strategies on the 5×5 grid and the Monaco subnet. Each strategy is applied to IA2C, MA2C, IQL-LR (`iqll`), and PPO, using five independent training seeds.

| Method ID | Demand during each 600-second block | WCE parameter learning |
|---|---|---|
| `baseline` | Follow the original fixed sequential order of eleven training profiles. | Not applicable |
| `random_group` | Select one profile uniformly; use one-hot mixture weights. | Not applicable |
| `domain_randomization` | Sample a mixture from `Dirichlet(1,…,1)`. | Not applicable |
| `fixed_wce` | Let the pretrained WCE select a state-dependent mixture. | Disabled |
| `online_wce` | Let the same pretrained WCE select a state-dependent mixture. | Enabled after each full episode |

The baseline uses the original sequential training scheme throughout. There is no separate Uniform-only baseline or RARL comparison. There are **five methods and four controller families**; these are different dimensions of the experiment.

Every final controller receives **2,320,000 controller-learning steps**: a common 1,000,000-step parent followed by 1,320,000 additional steps. A controller-learning step is one joint network decision followed by five simulated seconds, collected for controller learning. Frozen-controller simulation and optimizer updates are counted separately. Equal step budgets do not imply equal wall-clock time.

```text
Stage 0  Set up inputs and pass C01–C10 verification
    |
Stage I  Train a common controller parent for 1,000,000 steps
    |                                      |
    |                              Stage II  Freeze a copy
    |                                        Train and save WCE
    |                                             |
Stage III  Continue controller copies for 1,320,000 steps each
    +-- baseline                                  |
    +-- random_group                              |
    +-- domain_randomization                      |
    +-- fixed_wce <--------- same pretrained WCE --+
    +-- online_wce <-------- same pretrained WCE --+
    |
Stage IV  Evaluate all five final controllers on paired demand
```

The WCE is trained against the **1,000,000-step parent**, not the final extended baseline. All five continuation methods inherit the same controller state for a given network, controller family, and training seed.

## 2 Environment and dataset setup

### Runtime and input records

1. Work from the repository root on the `revision` branch and record the exact source commit and any uncommitted changes used for experiments.
2. Use the existing `deeprlsc` environment as the starting runtime. The audit identified Python 3.6.13 and a legacy TensorFlow stack; the default shell Python 3.13 environment is not the established training runtime.
3. Record actual Python, TensorFlow, NumPy, TraCI, SUMO, OS, hardware, thread, and concurrent-workload settings. The audited SUMO build was `1_26_0+0455-77b9dbc222e`.
4. Record hashes of network assets, controller configurations, the monitored lane set, and ordered demand manifests.
5. Create dedicated revision datasets and output locations. Preserve existing networks, original CSVs, checkpoints, and historical results.

These existing commands inspect the starting setup; they do not launch training:

```bash
cd /home/sdc_joran/Journal/deeprl_signal_control
conda activate deeprlsc
python --version
sumo --version
git branch --show-current
git rev-parse HEAD
git status --short
```

Historical entrypoints retain legacy behavior. The corrected stage runner is available through `python main.py experiment ...`; see README.md for executable commands.

### Networks and training profiles

| Setting | Grid | Monaco |
|---|---:|---:|
| Signal controllers | 25 | 28 |
| Unique controlled incoming lanes | 150 | 116 |
| Explicit training profiles | 11 | 11 |
| Reference demand per profile | 3,000 veh/hour | 2,383.3333 veh/hour |
| Controller interval | 5 seconds | 5 seconds |
| Yellow interval | 2 seconds | 2 seconds |
| Demand-selection interval | 600 seconds | 600 seconds |
| Training episode | 6,600 seconds | 6,600 seconds |
| Controller-learning steps per full episode | 1,320 | 1,320 |

Use the grid network under `large_grid/data/` and the Monaco network under `real_net_subnet/data/`. Confirm the actual network files referenced by each SUMO configuration before recording their hashes.

Create normalized copies of the eleven established directional, center/periphery, and Uniform training profiles. Preserve OD proportions and store the original total and normalization factor. Initial training and baseline continuation use the original profile order: the grid's established eleven-profile alphabetical order and Monaco's explicit eleven-profile sequence. Record the filenames in order; do not infer them anew on each run.

The current grid loader discovers **twelve** valid profiles, including `demand_5x5_sparse.csv`. An explicit eleven-profile training manifest is required so adding a test CSV cannot alter episode duration or WCE output dimension. Monaco also uses an explicit eleven-profile manifest in the revision.

### Dataset validation and output organization

Keep training, validation, and test datasets separate. Validate finite nonnegative rates, edge existence, and vehicle-class-compatible route connectivity. Failed routes must not silently disappear from the requested traffic.

Use unique run and attempt identifiers. The integrated project uses these locations; historical files remain in place:

```text
config/revised/
data_traffic/revised/{train,validation,test}/
real_net_subnet/demand_groups/revised/{train,validation,test}/
runs/revised/<network>/<controller>/seed_<seed>/<stage>/<run_id>/
output_adversary/revised/                 # grid WCE
output_adversary_monaco/revised/          # Monaco WCE
output_coevolution/revised/               # grid continuations
output_coevolution_real/revised/          # Monaco continuations
runs_eval/revised/                       # evaluation and verification
output_result/revised/                   # tables and report exports
figs/revised/                            # scientific plots
```

Each run manifest records its method, stage, network, controller, seed streams, input hashes, exact parent checkpoints, step budgets, reward definition, and output location. Outputs record the effective values actually used.

## 3 Mandatory corrections and verification

**All ten corrections must be implemented and verified before the full publication training matrix starts.** Each acceptance check below is a requirement, not a completed test. Save its result and evidence under the verification run identifier. Historical source locations and observations are retained in the [archived audit](reviewer_revision_audit_2026-09-14.md).

### C01 Apply evaluation seeds correctly

**Required change.** Set `env.train_mode = False` before evaluation resets. Pass the intended rollout index and record the effective SUMO seed used by each simulation.

**Acceptance check.** Request two different evaluation seeds and verify that the simulator receives them. Repeating a seeded evaluation must reproduce its exogenous demand realization.

### C02 Separate demand and controller randomness

**Required change.** Establish independent random streams for demand generation, controller action sampling, WCE sampling, replay sampling, and SUMO. For evaluation, materialize vehicle counts, departure times, OD pairs, route edge sequences, and speed factors before comparing controllers. Reuse the same complete artifact across methods.

**Acceptance check.** Changing the controller or its action-sampling seed must not change the scheduled traffic artifact. Record actual vehicle insertion separately because congestion can delay entry. Training methods intentionally select different demand; the identical-artifact requirement applies to paired evaluation.

### C03 Unify controller reward handling

**Required change.** Use one controller reward-construction path for initial training and all five continuation methods. IA2C, PPO, and IQL-LR receive the prescribed shared network reward; MA2C receives the prescribed neighborhood-weighted reward. Apply the Section 4 normalization once and remove additional legacy normalization from the revised path, including unintended extra Monaco scaling.

**Acceptance check.** Identical local queue measurements must yield identical learner reward vectors across methods for the same network/controller family.

### C04 Update MA2C fingerprints consistently

**Required change.** Update fingerprints in initial training, all continuation methods, frozen-controller WCE training, and evaluation. Reset fingerprints and recurrent states at episode boundaries. A frozen controller continues to update inference state and fingerprints while its learned parameters remain fixed.

**Acceptance check.** Check that fingerprints reflect current policy outputs and reset correctly between episodes. Verify that frozen-controller model parameters do not change.

### C05 Match IQL learning frequency

**Required change.** Trigger IQL backward calls after every twenty newly collected controller-learning transitions once replay contains sufficient samples. Preserve replay capacity, sampling rules, and ten minibatch updates per agent per backward call. Frozen-controller simulation must not add replay samples or advance learning counters.

**Acceptance check.** Compare controller-learning transitions, backward calls, and minibatch updates across equal-length runs of all five methods. Counts must match within each network/controller combination.

### C06 Correct Monaco IQL interfaces

**Required change.** Use the IQL inference interface rather than the actor-critic interface. Read replay through supported attributes rather than assuming an `.obs` field. Use greedy IQL inference for evaluation and offline WCE training, and the established exploration schedule during controller learning.

**Acceptance check.** Run short Monaco IQL checks covering frozen-controller WCE training, continuation, checkpoint loading, and evaluation without signature or buffer-attribute errors.

### C07 Align queue measurements and reward scaling

**Required change.** Measure uncapped halting counts every second over globally deduplicated controlled incoming lanes. Use that measurement source for controller local costs, offline and online WCE rewards, and primary evaluation. Remove the revised path's grid queue-plus-wait objective and Monaco queue cap. Apply the formulas in Section 4.

**Acceptance check.** Recompute controller costs and WCE rewards from saved lane measurements. They must agree with logged values after only the specified aggregation and scaling. Include a lane queue above ten vehicles in this check.

### C08 Restore complete training state and reject failed loads

**Required change.** Save model parameters, optimizer state, schedules, relevant buffers, recoverable random state, counters, and parent identifiers. Construct the checkpoint saver after optimizer variables exist. Missing or incompatible required checkpoints must stop the run; do not silently initialize a random model.

Save regular resumable checkpoints at episode boundaries. At the exact-budget parent cutoff, flush pending on-policy samples with the correct bootstrap and save a declared stage-boundary checkpoint even if the episode is incomplete. All continuation methods then begin a fresh episode from that same saved training state. An interrupted episode restarts from the last complete checkpoint, with discarded computation recorded.

**Acceptance check.** Verify restored parameters, optimizer variables, replay contents, schedules, random state, and counters. Compare the next learning update under controlled randomness. Confirm that missing and incompatible checkpoints fail explicitly.

### C09 Reject incomplete evaluation rollouts

**Required change.** Remove last-value and zero padding from the revised evaluation path. Validate timestamps, sample count, and the full 3,600-second horizon. Save failed or incomplete attempts with an explicit status and exclude them from performance summaries; do not convert them to zero-queue outcomes.

**Acceptance check.** Deliberately interrupt a rollout and confirm it is flagged and excluded. A congested simulation that completes the full horizon is still valid. If an infrastructure failure is rerun, retain the failed attempt and reuse the prescribed checkpoint, demand, and seeds.

### C10 Keep output provenance consistent

**Required change.** Use a unique output directory and immutable input manifest for every run, recording network, demand, configuration, checkpoint, and seed identifiers. Partial reruns create separate attempt records and must not overwrite a full-run manifest or mix unrelated checkpoints.

**Acceptance check.** Resolve every result to its actual inputs and checkpoint. Check expected versus completed rollout counts, and reject duplicate identifiers or incompatible manifests.

## 4 Controller and WCE parameters

### Controller settings

| Controller | Learning rate | Grid batch size | Monaco batch size |
|---|---:|---:|---:|
| IA2C | 0.0005 | 120 | 40 |
| MA2C | 0.0005 | 120 | 40 |
| PPO | 0.0003 | 120 | 120 |
| IQL-LR | 0.0001 | 20 | 20 |

For IA2C/MA2C, batch size is an on-policy rollout length. PPO reuses its rollout batch for multiple optimization epochs. IQL-LR samples minibatches from replay; its controller-learning collection interval is separately fixed to twenty steps by C05. Different algorithms need not use equal batches or learning rates. Within one network/controller combination, all five methods must use the same settings and update cadence.

Retain controller discount 0.99 and the existing controller architectures. IA2C/MA2C use LSTM size 64; MA2C neighborhood discount is 0.9. PPO uses clipping ratio 0.2 and four optimization epochs. IQL replay capacity is 1,000, with epsilon decreasing from 1.0 to 0.01 over the first 500,000 controller-learning steps and remaining at 0.01 throughout continuation. Do not restart or stretch that schedule when extending the training budget.

Store all remaining model settings in copied revision configurations. The original INIs remain historical inputs; the queue-only objective, normalization, and WCE settings below are revision targets.

### Queue metric and controller reward

Let $L$ be the fixed unique set of controlled incoming lanes, and let $q_\ell(t)$ be its uncapped halting count sampled every second. Define:

$$
Q(t)=\sum_{\ell\in L}q_\ell(t),\qquad
J_Q=\frac{1}{H}\sum_{t=1}^{H}Q(t).
$$

Report $J_Q$ as **mean total queue on controlled approaches**, in vehicles; lower is better. The audited lane counts are 150 for the grid and 116 for Monaco. Store and validate the exact lane lists. SUMO's lane halting count uses speed below 0.1 m/s. [SUMO lane-value documentation](https://sumo.dlr.de/docs/TraCI/Lane_Value_Retrieval.html)

For each five-second controller transition, average each junction's local queue over all five seconds. For IA2C, PPO, and IQL-LR, supply the negative mean network queue to each agent. For MA2C, supply the negative local mean queue plus the negative neighboring local means weighted by 0.9. Divide the resulting learner reward by 100 **once** and disable reward clipping. Do not retain an additional Monaco-specific divisor in this revised reward path.

### WCE settings

| Parameter | Revised setting |
|---|---|
| Grid architecture | CNN with consistently ordered local wave/wait features |
| Monaco architecture | GCN with controller-independent local lane-wave features, fixed padding/masks, and fixed adjacency |
| Action | Eleven Gaussian outputs converted to mixture weights through softmax |
| Learning rate | 0.0005 |
| Discount | 1.0 |
| Batch | Eleven macro transitions, one full episode |
| Demand action interval | 600 seconds |
| Parameter update interval | After each full episode when WCE learning is enabled |
| Additional reward normalization and clipping | Disabled |

Compute the WCE reward directly from canonical queue measurements, not by summing transformed controller rewards:

$$
r_{\mathrm{WCE},k}=\frac{1}{100}\frac{1}{600}\sum_{t\in k}Q(t).
$$

WCE maximizes positive queue cost; controllers learn from negative queue rewards. Evaluation uses unscaled $J_Q$. With eleven equal-duration blocks and WCE discount 1.0, the summed WCE rewards are proportional to the episode's mean queue.

Fixed and online WCE must share the same initial checkpoint, architecture, observations, and action-sampling rule. Demand-mixture weights may change every 600 seconds in both methods. Only online WCE updates model parameters between episodes.

## 5 Initial baseline training

### Stage 0 Set up and verify

**Prerequisites:** Section 2 inputs and C01–C10.

The integrated runner implements the corrections. After integration, run fresh checks for all eight network/controller combinations. Exercise the five continuation methods and the frozen-controller WCE path. Use pilot seeds outside publication seed sets, record the evidence for each correction, and resolve failures before full training. Pilot results are debugging evidence, not publication observations.

**Output:** Validated input manifests and a verification record showing that every required check has passed. The current [integration report](docs/INTEGRATION_REPORT.md) links the accepted pilot evidence and source-specific gate. Revalidate the gate whenever implementation or experiment inputs change.

### Stage I Train common parent controllers

**Required checks:** C02–C08 and C10 must remain active during training.

Use master training seeds `101, 202, 303, 404, 505`. For each network/controller/seed combination:

1. Initialize a fresh controller with the revised queue reward and fixed configuration.
2. Train under the original sequential schedule of eleven normalized demand profiles.
3. Stop at exactly **1,000,000 controller-learning steps**, including a partial final episode if necessary.
4. Flush the final partial on-policy batch with correct bootstrapping. Save after that update; preserve IQL replay and schedule state as applicable.
5. Save the full common parent checkpoint, cumulative counters, input hashes, seed streams, and stage timing.

**Output:** 40 independent parent checkpoints. Five continuations from one trained parent cannot substitute for five independently trained parents.

This parent is not the final baseline result. The baseline continues to the same final budget as the four comparison methods in Stage III.

## 6 WCE training against frozen controllers

### Stage II Train and save the common WCE

**Required checks:** C02–C04, C06–C08, and C10. C05 counters must confirm that frozen-controller learning remains disabled.

For each Stage I parent:

1. Load the exact **1,000,000-step parent checkpoint** and freeze learned controller parameters.
2. Disable controller optimization, replay insertion, and learning-schedule advancement. Continue correct inference-state and MA2C fingerprint updates, resetting state between episodes.
3. Use sampled policy actions for IA2C/MA2C/PPO and greedy IQL actions while generating WCE training experience.
4. Train WCE for **500 episodes**, giving **5,500 macro transitions**. Each macro transition includes a 600-second block, or 120 frozen-controller simulation steps.
5. Save the final WCE checkpoint, its exact parent identity, counters, and timing. Verify that controller parameters remained unchanged.

**Output:** 40 pretrained WCE checkpoints. Each WCE run adds **660,000 frozen-controller simulation steps**; none is counted toward the controller-learning budget.

Use the same saved WCE as the initial model for `fixed_wce` and `online_wce`. The other three methods do not require WCE training for their standalone operation.

## 7 Baseline continuation and comparison training

### Stage III Train the five continuation methods

**Required checks:** C02–C08 and C10; use the same controller interaction/update path across methods.

1. Create five controller copies from the identical Stage I model, optimizer, schedule, replay, and recorded training state for each network/controller/seed combination. Begin a fresh episode and reset recurrent state consistently.
2. Apply the demand rule from Section 1 for the chosen method.
3. Collect exactly **1,320,000 additional controller-learning steps**, equivalent to 1,000 full revised episodes. Stop at **2,320,000 cumulative steps**.
4. Keep controller reward construction, batch schedules, IQL update frequency, PPO epochs, checkpoint policy, and failure handling equal across methods within that network/controller combination.
5. Save each exact final-budget checkpoint and its parent identity. Record controller-learning steps, optimizer calls, minibatch updates, WCE decisions/updates, and wall-clock components.

**Method behavior:** `baseline` continues the original sequential schedule. `random_group` selects one of eleven profiles with equal probability and uses a one-hot vector. `domain_randomization` samples nonnegative mixture weights summing to one; it combines OD rates from all profiles using `Dirichlet(1,…,1)` weights. Because training profiles share the same total demand, both methods preserve the expected total rate while changing its spatial allocation.

`fixed_wce` performs state-dependent inference but stores no WCE learning transitions and makes no WCE optimizer updates. `online_wce` records WCE transitions and updates after eleven macro transitions at the episode boundary. Its continuation includes 11,000 WCE learning transitions and 1,000 episode updates.

**Output:** 200 final controller checkpoints: 40 extended baselines and 160 controllers across the four comparison methods. The one-million-step parents are not an additional comparison method.

## 8 Evaluation timing and completion checks

### Stage IV Freeze evaluation inputs

**Required checks:** C01, C02, C04, C06–C10. Controller and WCE parameter learning is disabled during evaluation.

Evaluate the exact 2,320,000-step controller checkpoints on eleven seen profiles and twelve new scenarios per network. Use ten paired rollouts per scenario, a 3,600-second horizon, an empty initial network, and no drainage extension. Include startup in the primary average. Sample IA2C/MA2C/PPO actions and use greedy IQL actions, resetting the independent policy stream for every rollout.

Materialize complete demand artifacts before evaluation. Reuse a given scenario/arrival realization across all methods and controller seeds, together with its prescribed SUMO seed. Retain actual insertion, pending departures, and residual traffic as outcomes. WCE does not adapt the held-out test demand during this common evaluation.

### New scenarios and seed assignments

| Family | Test generation seeds | Construction |
|---|---|---|
| OD-rate redistribution | `41001–41003` | Perturb positive Uniform-profile rates with seeded lognormal factors using log-space standard deviations 0.25, 0.50, and 0.75; renormalize to reference demand. |
| Convex mixtures | `41004–41006` | Generate three fixed `Dirichlet(1,…,1)` weight vectors over training profiles. |
| Temporal changes | `41007–41009` | Alternate N→S and W→E every 300, 900, or 1,200 seconds over the horizon. |
| Peak intensity | `41010–41012` | Multiply Uniform demand by 1.10, 1.25, or 1.50 during the interval `[1200, 2400)` seconds; use reference demand outside it. |

Use seeds in the order shown for each network. Convex mixtures test new compositions within the WCE-accessible demand family; they do not automatically establish out-of-distribution robustness. Rate perturbations likewise do not necessarily add previously unseen OD pairs.

Preserve the six validation generation seeds `31001–31006`: two rate redistributions, two mixtures, one temporal-switch scenario, and one 1.15× peak scenario. Freeze validation lognormal standard deviations at 0.35 and 0.65, the temporal switch interval at 450 seconds, and the peak at 1.15× before training or tuning. Validation and final-test artifacts must have distinct hashes, and final-test results must not determine parameters or checkpoint selection.

Use arrival-realization seeds `51001–51010` and SUMO seeds `61001–61010`, with a separately recorded policy seed derived from the training seed and rollout index. Pair the same policy seed within matched method comparisons while keeping it independent from demand generation.

Exclude `Real_Life_Monaco` from the main suite until its network/routing provenance is resolved: the audit describes 272 edges, while the active subnet has 270 and a positive-demand OD pair failed a static connectivity check. Do not silently drop that flow. Existing `demand_5x5_noisy` profiles total approximately 13,787–17,379 veh/hour and belong in a separate extreme-load analysis. Do not discard valid test scenarios because they produce congestion.

### Per-rollout records and uncertainty

Export one summary per complete rollout with its network, controller, method, training seed, scenario/split, checkpoint hash, demand hash, all seeds, policy mode, horizon, sample count, and wall time. Retain queue and traffic time series, lane measurements, and trip records.

Record mean/integrated/peak queue; vehicle-time-weighted mean speed; scheduled, inserted, completed, pending, and remaining vehicle counts; teleports, collisions, failures; and completed-trip travel time, waiting time, time loss, and departure delay with their denominators. An entirely empty rollout has unavailable mean speed, not an invented zero-speed observation. Incomplete attempts remain separately identifiable under C09–C10.

For each training seed and scenario, calculate $J_Q$ for each of ten valid rollouts, then report their mean and sample SD. For each network/controller/method combination, average the twelve new-scenario estimates with equal scenario weights within each training seed. Report the five resulting values, their mean, sample SD, and 95% t-interval:

$$
\bar J\pm 2.776\,\frac{s}{\sqrt{5}}.
$$

For comparisons, form the five matched training-seed differences first, then calculate their interval. Do not count individual seconds as independent replicates or pool different controller families/networks into the five-seed estimate. Report seen-profile performance and the four new-demand families separately. Define the exploratory worst tested scenario mean as the maximum scenario mean after averaging over training seeds and rollouts. Intervals describe uncertainty conditional on the fixed test suite. [Agarwal et al. on RL evaluation](https://arxiv.org/abs/2108.13264)

### Peak-demand heatmaps

For the predefined 1.25× and 1.50× peaks, aggregate lane queues over the same `[1200, 2400)`-second window. Produce absolute maps for the five methods and `online_wce − comparator` maps, averaged over matched training seeds and rollout realizations.

Use a shared absolute scale within each network/scenario, a symmetric zero-centered difference scale, common geometry/extent, and explicit units. Show unmonitored roads in gray. Negative differences mean lower queue under online WCE. Label the monitored domain as controlled approaches and reconcile map aggregates with the same queue measurements. Do not select different peak windows or demand scenarios for different controllers.

### Training time and experiment counts

Use a monotonic clock to measure initial controller training, offline WCE training, continuation, setup/reset, inference, controller/WCE optimization, checkpoint/logging overhead, and failed or discarded work. Keep component timing nonoverlapping when summing it; report stage totals separately from the component breakdown. Record hardware, thread counts, and competing workload.

For `baseline`, `random_group`, and `domain_randomization`, standalone training cost includes parent training plus continuation. For `fixed_wce` and `online_wce`, also include offline WCE training, even if the experimental campaign reused one pretrained WCE for both. Do not claim equal wall time from equal controller-learning steps.

| Work item | Count |
|---|---:|
| Common parent training runs | 40 |
| Offline WCE training runs | 40 |
| Continuation runs including baseline | 200 |
| Final evaluation rollouts | 46,000 |

The evaluation count is `2 × 4 × 5 × 5 × 23 × 10 = 46,000`: networks × controller families × training seeds × methods × scenarios × rollouts. Pilot, validation, and failed attempts are additional and must be recorded separately.

### Publication completion checklist

- [ ] C01–C10 each have a passing verification record and supporting evidence.
- [ ] Five independent parents exist per network/controller family, with complete saved state.
- [ ] Every WCE points to the correct frozen 1,000,000-step parent.
- [ ] All five continuation methods finish at 2,320,000 controller-learning steps with matched controller update cadence.
- [ ] Frozen controllers and fixed WCE parameters remain unchanged; online WCE model parameters change during learning and its intended updates are recorded.
- [ ] Test demand artifacts and seeds match across paired comparisons; training/test separation is verified.
- [ ] Only complete rollouts enter summaries; all failed attempts remain visible.
- [ ] Queue rewards, evaluation metrics, and heatmaps reconcile under the declared lane set and scales.
- [ ] Rollout variability and training-seed uncertainty are reported separately for each network/controller/method.
- [ ] Run counts, checkpoint identities, immutable manifests, and timing records agree.

### Bilingual terminology

| English | Simplified Chinese |
|---|---|
| Baseline training | 基线训练 |
| Controller-learning step | 控制器训练步 |
| Frozen-controller simulation step | 冻结控制器仿真步 |
| Training episode | 训练回合 |
| Demand group | 交通需求组 |
| Demand-mixture weights | 需求混合权重 |
| Model parameters | 模型参数 |
| Domain randomization | 域随机化 |
| Checkpoint | 检查点 |

The English and Chinese guides define the same procedure. Keep method IDs, configuration keys, paths, numerical values, correction identifiers, and equations synchronized when updating either version.
