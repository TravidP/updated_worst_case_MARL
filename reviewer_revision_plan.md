# CB-WCE training and evaluation procedure

[简体中文版](reviewer_revision_plan_zh.md) · [Historical repository audit](reviewer_revision_audit_2026-09-14.md)

**Protocol version:** 3

**Date:** 15 September 2026

**Repository:** `/home/sdc_joran/Journal/deeprl_signal_control`

**Implementation status:** This guide defines the revised study. The corrected implementation is integrated into `agents/`, `envs/`, and `experiments/`. Use `python main.py experiment ...`; see the [project guide](README.md). Historical C01–C10 pilot checks are recorded for that path; the visualization source change requires fresh publication verification. Existing legacy entrypoints remain unchanged. The full publication training and evaluation matrix has not been launched.

**Documentation history:** The earlier 15 September 2026 guide revision was documentation-only. The subsequent visualization update adds CLI/local-workspace display selection; its focused verification is documented in [the visualization report](docs/VISUALIZATION.md). Exact prior guides and their [SHA-256 index](docs/history/reviewer_guides_20260915T085603Z/checksums.sha256) are preserved in [the dated archive](docs/history/reviewer_guides_20260915T085603Z/).

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

### Executable setup and shared shell variables

**Purpose and inputs.** Activate the existing environment from the repository root using the commands above. The established runtime is Python 3.6.13, TensorFlow 1.12.0 and NumPy 1.19.5. The following variables define a worked Grid/IA2C example. For Monaco, set `CBWCE_NETWORK=monaco` and `CBWCE_DATASET_ROOT=real_net_subnet/demand_groups/revised`. Controller IDs are `ia2c`, `ma2c`, `iqll`, `ppo`; use each of the five publication seeds in separate runs. Pilot seed 9001 is never a publication replicate.

**Pilot/publication setup commands.** Input preparation is shared by both modes. `prepare` creates the normalized CSVs, scenario definitions and effective INIs, or checks that existing copies agree. It refuses to overwrite differing prepared files. `--materialize` starts short SUMO route-resolution sessions and creates complete traffic artifacts; it does not train a controller. The two materialization commands are not read-only readiness checks.

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
export CBWCE_NETWORK=grid
export CBWCE_CONTROLLER=ia2c
export CBWCE_SEED=101
export CBWCE_PILOT_SEED=9001
export CBWCE_DATASET_ROOT=data_traffic/revised
export CBWCE_GATE=runs_eval/revised/verification/final_integration_20260914/gate.json

python main.py experiment prepare
python main.py experiment check --gate "$CBWCE_GATE"
python main.py experiment prepare --materialize --network grid
python main.py experiment prepare --materialize --network monaco
```

**Implementation.** [CLI](experiments/cli.py) dispatches to [preparation](experiments/prepare.py), [demand loading](experiments/demand.py), [scenario generation](experiments/scenarios.py) and [SUMO environment](envs/experiment_env.py).

**Outputs and checks.** Each network has eleven training CSVs, eleven seen definitions, twelve test definitions and six validation definitions. Ten realizations per definition produce 290 artifacts per network, 580 total, including validation. Source CSVs remain unchanged. Inspect `check`'s `inputs`, `modules`, `gate` and `ready` fields: `ready=true` alone does not mean the gate passed, and the command can return normally while reporting a failed gate. Revalidate the named gate before publication. A missing prepared manifest/configuration currently falls back to original inputs; always run `prepare` and confirm `prepared=true` for this study.

### Optional SUMO visualization

**Interactive pilots:** the parent, frozen-WCE, continuation, evaluation and resume examples below include `--visualization`. The workspace defaults to visualization **On for pilots** and **Off for publication** when you choose the run type. You can select Off for a manual pilot; saved display choices are retained when reopening the page or switching language. To apply the new pilot default to an older saved form, select Publication and then Pilot again, or choose On directly. Automated `verify` runs stay headless. The CLI itself still defaults to headless when neither display flag is supplied.

Add `--visualization` to `parent`, `wce`, `continue` or `evaluate` to show a separate local SUMO window. Use `--no-visualization`, or omit both flags, to keep it off. The flags are mutually exclusive; do not pass `--visualization false`. These options belong after `python main.py experiment <stage>` and are also available through the `revision.runner` compatibility wrapper.

```bash
# SUMO window on
python main.py experiment parent --network grid --controller ia2c \
  --seed 9001 --pilot --steps 160 --visualization

# SUMO window off (also the default when neither flag is supplied)
python main.py experiment parent --network grid --controller ia2c \
  --seed 9001 --pilot --steps 160 --no-visualization
```

Choose one command, not both for the same intended run. Both are short pilots; publication budgets and learning parameters are unchanged. On the local workspace, select **SUMO visualization → On / Off** before clicking **Check & preview**. The selection persists when switching language and appears in the generated command and job request. It opens on the training computer, not inside the browser. Restart an already-running dashboard server to load the updated backend, then refresh the page; do not interrupt an active training job merely to refresh the interface.

| Effective value | Meaning | Implementation/configuration source | Configurable? |
|---|---|---|---|
| CLI default `false`; workspace pilot default `true` | Headless SUMO; `true` opens `sumo-gui` and auto-starts playback | CLI `--visualization` / `--no-visualization`; `experiments/runner.py`, `envs/experiment_env.py`, `experiments/dashboard.py` | Per invocation / local workspace; not an INI field |

A working `sumo-gui` executable and graphical desktop are required. On Linux, `DISPLAY` must identify an accessible X display; a Wayland-only variable does not provide an X display for SUMO. `check` reports `sumo_gui` and `display`, while GUI launch/preflight rejects missing prerequisites. A stale or inaccessible display can still fail at SUMO startup. Background preparation/route-materialization sessions remain headless. Each selected training/evaluation episode opens a GUI session with `--start --quit-on-end`; use SUMO's View Settings for colors/zoom and its Delay control for playback speed.

The option applies at job start, not as a live switch. On resume, choose either display mode for the new attempt; checkpoint compatibility and original learning budget are unchanged. Keep visualization **off for timed publication comparisons** because rendering and playback delay add wall-clock overhead. The selected boolean is in `<run>/manifest.json` as `visualization`; each `runtime/startup_*.json` records its actual `visualization` and SUMO command. Constructor/bootstrap and suite route-preparation sessions can be headless even for a visualized job. Queue NPZ, traffic JSONL, checkpoints and result paths keep the formats described below.

**Verification status after this change:** the 14 September gates remain historical evidence. Their source hashes no longer match the visualization implementation. Run `python main.py experiment verify --workers 4` and select its new passing gate before publication training. The focused visualization checks do not replace the full eight-case gate.

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

The legacy grid loader discovers **twelve** valid profiles, including `demand_5x5_sparse.csv`. An explicit eleven-profile training manifest is required so adding a test CSV cannot alter episode duration or WCE output dimension. Monaco also uses an explicit eleven-profile manifest in the revision.

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

### Stage-to-output map

| Root | Contents |
|---|---|
| `runs/revised/` | Both networks: parents and `baseline` continuations |
| `output_adversary/revised/` | Grid offline WCE |
| `output_adversary_monaco/revised/` | Monaco offline WCE |
| `output_coevolution/revised/` | Grid four comparison continuations |
| `output_coevolution_real/revised/` | Monaco four comparison continuations |
| `runs_eval/revised/` | Evaluation; `verification/`, `preparation/`, `jobs/` records |
| `output_result/revised/` | Report directories: CSV, JSON, PNG, SVG |
| `figs/revised/` | Separately retained figure copies; not a second automatic report destination |

The stage runner selects paths through [experiments/protocol.py](experiments/protocol.py): `<stage-root>/<network>/<controller>/seed_<seed>/<stage-or-method>/<run_id>/`. The stage component is `parent`, `wce`, or `evaluate`; continuations use the method ID. Generated run IDs contain `pilot_` or `publication_`, a UTC timestamp, and a random suffix. An explicit `--output` must name a new directory. A report with no `--output` currently gets a `pilot_`-prefixed directory even when reading publication data: classify observations by their manifests, not that report-directory prefix.

## 3 Mandatory corrections and verification

**All ten corrections must be implemented and verified before the full publication training matrix starts.** Each acceptance check below is an ongoing requirement; observed results and their exact validation scope are linked in Stage 0. Save its result and evidence under the verification run identifier. Historical source locations and observations are retained in the [archived audit](reviewer_revision_audit_2026-09-14.md).

### C01 Apply evaluation seeds correctly

**Required behavior.** Set `env.train_mode = False` before evaluation resets. Pass the intended rollout index and record the effective SUMO seed used by each simulation.

**Acceptance check.** Request two different evaluation seeds and verify that the simulator receives them. Repeating a seeded evaluation must reproduce its exogenous demand realization.

### C02 Separate demand and controller randomness

**Required behavior.** Establish independent random streams for demand generation, controller action sampling, WCE sampling, replay sampling, and SUMO. For evaluation, materialize vehicle counts, departure times, OD pairs, route edge sequences, and speed factors before comparing controllers. Reuse the same complete artifact across methods.

**Acceptance check.** Changing the controller or its action-sampling seed must not change the scheduled traffic artifact. Record actual vehicle insertion separately because congestion can delay entry. Training methods intentionally select different demand; the identical-artifact requirement applies to paired evaluation.

### C03 Unify controller reward handling

**Required behavior.** Use one controller reward-construction path for initial training and all five continuation methods. IA2C, PPO, and IQL-LR receive the prescribed shared network reward; MA2C receives the prescribed neighborhood-weighted reward. Apply the Section 4 normalization once and remove additional legacy normalization from the revised path, including unintended extra Monaco scaling.

**Acceptance check.** Identical local queue measurements must yield identical learner reward vectors across methods for the same network/controller family.

### C04 Update MA2C fingerprints consistently

**Required behavior.** Update fingerprints in initial training, all continuation methods, frozen-controller WCE training, and evaluation. Reset fingerprints and recurrent states at episode boundaries. A frozen controller continues to update inference state and fingerprints while its learned parameters remain fixed.

**Acceptance check.** Check that fingerprints reflect current policy outputs and reset correctly between episodes. Verify that frozen-controller model parameters do not change.

### C05 Match IQL learning frequency

**Required behavior.** Trigger IQL backward calls after every twenty newly collected controller-learning transitions once replay contains sufficient samples. Preserve replay capacity, sampling rules, and ten minibatch updates per agent per backward call. Frozen-controller simulation must not add replay samples or advance learning counters.

**Acceptance check.** Compare controller-learning transitions, backward calls, and minibatch updates across equal-length runs of all five methods. Counts must match within each network/controller combination.

### C06 Correct Monaco IQL interfaces

**Required behavior.** Use the IQL inference interface rather than the actor-critic interface. Read replay through supported attributes rather than assuming an `.obs` field. Use greedy IQL inference for evaluation and offline WCE training, and the established exploration schedule during controller learning.

**Acceptance check.** Run short Monaco IQL checks covering frozen-controller WCE training, continuation, checkpoint loading, and evaluation without signature or buffer-attribute errors.

### C07 Align queue measurements and reward scaling

**Required behavior.** Measure uncapped halting counts every second over globally deduplicated controlled incoming lanes. Use that measurement source for controller local costs, offline and online WCE rewards, and primary evaluation. Remove the revised path's grid queue-plus-wait objective and Monaco queue cap. Apply the formulas in Section 4.

**Acceptance check.** Recompute controller costs and WCE rewards from saved lane measurements. They must agree with logged values after only the specified aggregation and scaling. Include a lane queue above ten vehicles in this check.

### C08 Restore complete training state and reject failed loads

**Required behavior.** Save model parameters, optimizer state, schedules, relevant buffers, recoverable random state, counters, and parent identifiers. Construct the checkpoint saver after optimizer variables exist. Missing or incompatible required checkpoints must stop the run; do not silently initialize a random model.

Save regular resumable checkpoints at episode boundaries. At the exact-budget parent cutoff, flush pending on-policy samples with the correct bootstrap and save a declared stage-boundary checkpoint even if the episode is incomplete. All continuation methods then begin a fresh episode from that same saved training state. An interrupted episode restarts from the last complete checkpoint, with discarded computation recorded.

**Acceptance check.** Verify restored parameters, optimizer variables, replay contents, schedules, random state, and counters. Compare the next learning update under controlled randomness. Confirm that missing and incompatible checkpoints fail explicitly.

### C09 Reject incomplete evaluation rollouts

**Required behavior.** Remove last-value and zero padding from the revised evaluation path. Validate timestamps, sample count, and the full 3,600-second horizon. Save failed or incomplete attempts with an explicit status and exclude them from performance summaries; do not convert them to zero-queue outcomes.

**Acceptance check.** Deliberately interrupt a rollout and confirm it is flagged and excluded. A congested simulation that completes the full horizon is still valid. If an infrastructure failure is rerun, retain the failed attempt and reuse the prescribed checkpoint, demand, and seeds.

### C10 Keep output provenance consistent

**Required behavior.** Use a unique output directory and immutable input manifest for every run, recording network, demand, configuration, checkpoint, and seed identifiers. Partial reruns create separate attempt records and must not overwrite a full-run manifest or mix unrelated checkpoints.

**Acceptance check.** Resolve every result to its actual inputs and checkpoint. Check expected versus completed rollout counts, and reject duplicate identifiers or incompatible manifests.

## 4 Effective controller and WCE parameters

The settings below describe the current corrected implementation. Read `MODEL_CONFIG` from `config/revised/config_<controller>_<large|real>.ini`, with `large` meaning Grid. [Controller adapter](agents/controller.py), [WCE adapter](agents/wce.py), [policy classes](agents/policies.py) and [measurement/reward code](experiments/core.py) determine the effective behavior. A JSON or INI field is not necessarily a live tuning switch: several publication constants are also enforced in code. Do not change a field and assume the whole protocol changed.

### Controller settings and their meanings

| Effective value | Meaning | Implementation/configuration source | Configurable? |
|---|---|---|---|
| IA2C/MA2C LR `0.0005`; PPO `0.0003`; IQL `0.0001` | Constant step size; no learning-rate decay is applied | `MODEL_CONFIG.lr_init`; `agents/controller.py` | INI; new configuration requires verification |
| IA2C/MA2C batches: Grid `120`, Monaco `40`; PPO `120`; IQL `20` | On-policy sequence length for actor-critic/PPO; replay minibatch size for IQL | `MODEL_CONFIG.batch_size`; `Controller.observe/flush` | INI; cadence also has fixed rules below |
| Actor-critic/PPO: wave FC `128`, wait FC `32`, LSTM `64`; MA2C fingerprint FC `64` | Compact recurrent policies; IQL uses a linear Q policy, without LSTM | `num_fw`, `num_ft`, `num_lstm`, `num_fp`; `agents/recurrent.py`, `agents/policies.py` | Widths: INI; architecture: code |
| IA2C/MA2C: RMSProp; PPO/IQL: Adam | RMSProp decay `0.99`; actor-critic/PPO epsilon `1e-5`; IQL uses Adam defaults | `masked_loss`; `LRQPolicy.prepare_loss`; `rmsp_alpha`, `rmsp_epsilon` | Optimizer choice: code; named RMS fields: INI |
| Controller discount `0.99` | Return/TD discount; separate from MA2C spatial weighting | `Controller.flush`; IQL `prepare_loss(..., .99)` | Fixed in code; changing INI `gamma` alone has no effect here |
| Gradient norm limit `40` | Clips gradient global norm, not queue rewards | `MODEL_CONFIG.max_grad_norm`; controller loss code | INI |
| Entropy coefficient `0.01`; value coefficient `0.5` | Recurrent loss is actor loss + `0.5 * value_coef * masked MSE` − entropy term: effective MSE multiplier `0.25`. IQL uses TD MSE only | `entropy_coef_init`, `value_coef`; `masked_loss` | INI coefficients; no entropy schedule in corrected adapter |
| PPO clip `0.2`; epochs `4`; advantage normalization `true` | Ratio bounded to `[0.8,1.2]`; four passes reuse the same starting recurrent state | `masked_loss`, `Controller.flush`; `ppo_adv_norm` | Clip/epochs: code; advantage normalization: INI |
| MA2C neighborhood weight `0.9` | Negative local queue plus weighted immediate-neighbor queues | `experiments/core.py: QueueMetric.rewards` | Fixed in code |
| IQL replay `1000`; learn every `20`; `10` minibatches per agent | Ring-buffer replacement; sample without replacement; replay readiness is an additional condition | `Controller.observe/_update_iql` | Capacity/cadence/update count: code; minibatch size: INI |
| IQL epsilon `max(0.01, 1 − scheduler_steps / 500000)` | Counter advances only during learning, before action selection; reaches floor at `495000`, remains `0.01` in continuation | `Controller.epsilon/act` | Fixed in code; legacy epsilon INI schedule is not used |
| CPU only; TF intra/inter threads `1/1` | Both controller and WCE sessions disable GPU; BLAS/OMP variables control their own libraries | `Controller.__init__`, `WCE.__init__`; setup shell variables | TF/GPU: code; BLAS/OMP: environment |

Batch meanings differ: IA2C/MA2C collect an on-policy sequence; PPO reuses that sequence for four epochs; IQL samples replay minibatches independently of its twenty-transition collection clock. Differences between algorithms or networks are permitted. All five methods within one network/controller combination must retain the same values and cadence.

**Exploration clarification.** Earlier prose described a decay over the first 500,000 steps. The implemented formula reaches 0.01 at 495,000, and the first sampled learning action uses the already-incremented counter. This guide records that distinction; no schedule has been changed. Frozen WCE-training/evaluation controllers use greedy IQL actions and do not advance the schedule.

**Retained legacy fields.** `reward_norm`, `reward_clip`, `TRAIN_CONFIG.total_step`, `test_interval`, `log_interval`, automatic `resume/resume_step`, learning-rate/entropy-decay fields and IQL epsilon/buffer settings in old INIs do not supply those controls to the corrected loop. PPO clip/epoch INI values currently match the constants but do not drive them. Explicit CLI checkpoint arguments control loading. Model INI contents still participate in checkpoint signatures, so even changing an unused field can invalidate compatibility.

### Timing, budgets and normalization

| Effective value | Meaning | Implementation/configuration source | Configurable? |
|---|---|---|---|
| Controller `5 s` = yellow `2 s` + green `3 s` | One joint decision; queue is measured after every simulated second, including yellow | `envs/experiment_env.py: step/_simulate` | Fixed in corrected code |
| Training `6600 s`; `11 × 600 s`; `1320` joint transitions | Episode and demand-block clocks; evaluation uses `3600 s` / `720` transitions | `experiments/runner.py`; `experiment_env.py` | Protocol records these values; loops also contain constants |
| Queue scale `100`; clipping disabled | Divide learner rewards and WCE block cost exactly once; evaluation remains in vehicles | `experiments/core.py: QueueMetric` | Fixed in code; not legacy `reward_norm/reward_clip` |
| Parent `1000000`; continuation `1320000`; offline WCE `500` episodes | Budgets count real learning transitions; WCE adds frozen simulation only | `config/revised/protocol.json`; runner goal selection | Protocol budgets; pilot-only `--steps/--episodes`; publication checks also enforce fixed totals |

### Queue metric and controller reward

Let $L$ be the fixed unique set of controlled incoming lanes, and let $q_\ell(t)$ be its uncapped halting count sampled every second. Define:

$$
Q(t)=\sum_{\ell\in L}q_\ell(t),\qquad
J_Q=\frac{1}{H}\sum_{t=1}^{H}Q(t).
$$

Report $J_Q$ as **mean total queue on controlled approaches**, in vehicles; lower is better. The audited lane counts are 150 for the grid and 116 for Monaco. Store and validate the exact lane lists. SUMO's lane halting count uses speed below 0.1 m/s. [SUMO lane-value documentation](https://sumo.dlr.de/docs/TraCI/Lane_Value_Retrieval.html)

For each five-second controller transition, average each junction's local queue over all five seconds. For IA2C, PPO, and IQL-LR, supply the negative mean network queue to each agent. For MA2C, supply the negative local mean queue plus the negative neighboring local means weighted by 0.9. Divide the resulting learner reward by 100 **once** and disable reward clipping. Do not retain an additional Monaco-specific divisor in this revised reward path.

### WCE settings and update behavior

| Effective value | Meaning | Implementation/configuration source | Configurable? |
|---|---|---|---|
| Grid CNN: conv `32/64`, FC `128`; Monaco GCN: two `64`-wide layers | Grid uses ordered wave/wait features; Monaco uses normalized lane-wave features without controller fingerprints | `agents/wce.py`; active Gaussian policy classes in `agents/policies.py`; `wce_observation` | Architecture/features: code |
| Action dimension `11`; Gaussian logits → softmax | Dedicated recoverable WCE RNG draws noise; demand-mixture weights are nonnegative and sum to one | `WCE.act`; `Streams` | Fixed in code and eleven-profile contract |
| WCE LR `0.0005`; RMSProp decay `0.99`, epsilon `1e-5` | Optimizer settings; the `0.99` argument in WCE `prepare_loss` is RMSProp decay, not return discount | `WCE.__init__/observe`; Gaussian policy loss | Fixed in code |
| WCE entropy `0.01`; value coefficient `0.5`; gradient norm `40` | Gaussian actor-critic loss and gradient control | `WCE` and Gaussian policy `prepare_loss/backward` | Fixed in code |
| WCE discount `1.0`; batch `11` macro transitions | Reverse cumulative sum of eleven block rewards; one update after a full episode when learning is enabled | `WCE.observe` | Fixed in code; not read from controller INI gamma |
| Actions every `600` seconds; no extra reward scaling/clipping | Fixed WCE parameters do not imply fixed demand-mixture weights; online and fixed use the same sampling rule | `experiments/runner.py`; `QueueMetric.wce` | Fixed in code |

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

### Stage 0 command reference and accepted evidence

**Purpose/prerequisites:** Establish the source-specific C01–C10 gate after preparing inputs. **Input:** current source and dataset/configuration hashes; no parent checkpoint is required. **Pilot command:** the verifier runs all eight network/controller combinations with pilot budgets. **Publication prerequisite command:** `check --gate` validates the selected evidence without starting training.

The final gate from 14 September is linked below. Reuse it only if `check` reports `gate: passed`. Run `verify` again when the gate is stale or new verification is needed, then replace the placeholder with its printed gate path. Do not run the verification matrix merely to read this guide.

```bash
python main.py experiment check --gate "$CBWCE_GATE"
python main.py experiment verify --workers 4

export CBWCE_GATE='REPLACE_WITH_THE_NEW_VERIFICATION_GATE_JSON'
python main.py experiment check --gate "$CBWCE_GATE"
```

**Scripts:** [experiments/verify.py](experiments/verify.py), [correction tests](tests/test_corrections.py), [workflow tests](tests/test_workflow.py), using the same corrected runner. **Output:** `runs_eval/revised/verification/<run_id>/gate.json`, test logs and one case directory per network/controller. **Completion:** all eight cases and all tests pass, and the selected gate matches source, input and evidence hashes.

**Existing evidence:** [integration report](docs/INTEGRATION_REPORT.md), [eight-case gate](runs_eval/revised/verification/integrated_20260914/gate.json), [accepted final gate](runs_eval/revised/verification/final_integration_20260914/gate.json), and [focused validation scope](runs_eval/revised/verification/final_integration_20260914/validation_scope.json). The eight-case pilots include 40 continuations, eight resume checks, 64 complete evaluations and eight deliberate interruptions. Subsequent identity/suite-label/timer/report fixes passed 18 tests, seven HTTP checks and one additional complete evaluation. The full eight-case matrix was not rerun after those focused interface fixes; learner/environment/configuration logic did not change. These are recorded results, not new tests performed by this documentation update.

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

### Stage I commands, scripts and saved state

**Purpose and required inputs:** Train a fresh common parent using the ordered eleven-profile manifest and effective INI. No `--parent`, `--wce` or `--resume` is supplied. Confirm C02–C08/C10 and the selected network/controller/seed.

**Pilot command:** 160 real learning transitions. A 120-step rollout uses one full batch plus 40 real entries in a masked partial batch. The pilot is a mechanics check, not a trained publication parent.

```bash
python main.py experiment parent --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --pilot --visualization --steps 160 --checkpoint-every 1
```

**Publication command:** exactly 1,000,000 learning transitions; do not pass `--steps` or `--episodes`.

```bash
python main.py experiment parent --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" --gate "$CBWCE_GATE"
```

**Scripts:** `main.py` → [experiments/cli.py](experiments/cli.py) → [experiments/runner.py](experiments/runner.py); learning in [agents/controller.py](agents/controller.py), measurements in [envs/experiment_env.py](envs/experiment_env.py).

**Output:** `runs/revised/<network>/<controller>/seed_<seed>/parent/<run_id>/`. The final checkpoint is `checkpoint_000000160` for this pilot or `checkpoint_001000000` for publication. Inspect the run's `result.json`: require `status=complete`, the expected `learning_steps`, and its explicit `checkpoint` field. The CLI also prints the returned checkpoint path. Copy that exact path into the relevant variable below; `REPLACE_...` strings are deliberately non-executable placeholders, not checkpoint names. Set only the mode you have actually trained.

```bash
export CBWCE_PARENT_PILOT='REPLACE_WITH_COMPLETED_PILOT_PARENT_CHECKPOINT'
export CBWCE_PARENT='REPLACE_WITH_COMPLETED_PUBLICATION_PARENT_CHECKPOINT'
python main.py experiment check --checkpoint "$CBWCE_PARENT"
```

**Completion:** the parent contains model and optimizer variables, buffers, counters and RNG state. Never substitute a historical weight-only checkpoint or treat five continuations from one seed as five independent parents.

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

### Stage II commands, scripts and saved state

**Purpose and inputs:** Learn challenging mixtures against the corresponding frozen parent. The selected parent must match network, controller and training seed. Publication WCE must start from the one-million-step parent; the pilot below starts from its 160-step test parent. Disable controller optimization, replay insertion and learning-schedule advancement while retaining inference-state/fingerprint updates.

**Pilot command:** two episodes, 22 macro transitions and 2,640 frozen-controller simulation steps.

```bash
python main.py experiment wce --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --parent "$CBWCE_PARENT_PILOT" --pilot --visualization --episodes 2 --checkpoint-every 1
```

**Publication command:** 500 episodes, 5,500 macro transitions and 660,000 frozen-controller simulation steps.

```bash
python main.py experiment wce --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --parent "$CBWCE_PARENT" --gate "$CBWCE_GATE"
```

**Scripts:** shared `experiments/runner.py`, [agents/wce.py](agents/wce.py), shared controller and environment adapters. **Output:** Grid `output_adversary/revised/<network>/<controller>/seed_<seed>/wce/<run_id>/`; Monaco uses `output_adversary_monaco/revised/`. Final checkpoint suffix is `000002640` for this pilot or `000660000` for publication. The WCE bundle also contains the frozen controller; its controller-learning count remains 160 or 1,000,000 respectively.

Select the exact checkpoint from the completed run's `result.json`, then set:

```bash
export CBWCE_WCE_PILOT='REPLACE_WITH_COMPLETED_PILOT_WCE_CHECKPOINT'
export CBWCE_WCE='REPLACE_WITH_COMPLETED_PUBLICATION_WCE_CHECKPOINT'
```

**Completion:** controller parameters/counters unchanged, WCE update count two or 500, and checkpoint parent hash identifies the exact controller used. Both fixed and online continuation must use this same pretrained WCE for the corresponding parent.

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

### Stage III commands for all five methods

**Purpose/inputs:** Fork five complete controller states from the same selected parent. Fixed/online methods additionally require the same WCE, whose recorded parent hash must match. Other methods must not load WCE. A method's demand choices may differ, but its controller architecture, reward path and update cadence must not.

**Pilot commands:** choose one command at a time. Each adds 2,640 learning steps, reaching 2,800 from the 160-step parent. The fixed/online pilots span two full episodes, allowing the frozen/updated distinction to be checked.

```bash
python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --method baseline --parent "$CBWCE_PARENT_PILOT" \
  --pilot --visualization --steps 2640 --checkpoint-every 1

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --method random_group --parent "$CBWCE_PARENT_PILOT" \
  --pilot --visualization --steps 2640 --checkpoint-every 1

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --method domain_randomization --parent "$CBWCE_PARENT_PILOT" \
  --pilot --visualization --steps 2640 --checkpoint-every 1

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --method fixed_wce --parent "$CBWCE_PARENT_PILOT" \
  --pilot --visualization --steps 2640 --checkpoint-every 1 --wce "$CBWCE_WCE_PILOT"

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --method online_wce --parent "$CBWCE_PARENT_PILOT" \
  --pilot --visualization --steps 2640 --checkpoint-every 1 --wce "$CBWCE_WCE_PILOT"
```

**Publication commands:** choose one command at a time. Each adds exactly 1,320,000 learning steps and finishes at 2,320,000. These commands do not form an automatic campaign queue.

```bash
python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --method baseline --parent "$CBWCE_PARENT" \
  --gate "$CBWCE_GATE"

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --method random_group --parent "$CBWCE_PARENT" \
  --gate "$CBWCE_GATE"

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --method domain_randomization --parent "$CBWCE_PARENT" \
  --gate "$CBWCE_GATE"

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --method fixed_wce --parent "$CBWCE_PARENT" \
  --gate "$CBWCE_GATE" --wce "$CBWCE_WCE"

python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --method online_wce --parent "$CBWCE_PARENT" \
  --gate "$CBWCE_GATE" --wce "$CBWCE_WCE"
```

**Scripts:** all commands share `experiments/runner.py` and `agents/controller.py`; only demand selection and WCE updating vary. **Output:** baseline goes to `runs/revised/`; the other four go to `output_coevolution/revised/` (Grid) or `output_coevolution_real/revised/` (Monaco), followed by `<network>/<controller>/seed_<seed>/<method>/<run_id>/`.

**Completion:** `result.json` records the expected cumulative learning count and `stage_simulation_steps`. Fixed WCE has unchanged parameters; online WCE gains 1,000 updates over publication continuation, from 500 pretrained to 1,500 total. `checkpoint_001320000` counts additional stage steps, not total learning. Choose each method's exact final controller for evaluation; do not evaluate the parent as the equal-budget baseline.

### Checkpoint selection, stopping and resume

The default `--checkpoint-every 10` saves every ten complete episodes and at stage completion. `--checkpoint-every 1` saves every full episode. A parent also saves at its exact-budget cutoff after correctly bootstrapping/masking a partial on-policy batch. A 1,000,000-step parent contains 757 complete 1,320-step episodes plus 760 transitions; a 120-step batch leaves 40 real samples at the cutoff. Padding contributes no learner loss and adds no environment transitions.

Read counters from checkpoint state and `result.json`, not filenames alone. Preserve the originating run's `manifest.json` in the parent directory of `checkpoint_*`; moving only a checkpoint directory loses the origin identity required by the current loader. Keep the complete bundle, not just its TensorFlow weights. Use only trusted locally produced pickle state.

Stopping retains the interrupted attempt and closes its owned SUMO connection. Resume starts a fresh episode in a new output directory, restoring the last complete checkpoint. Work after that checkpoint is discarded but its attempt/timing records remain. There is no mid-SUMO-state resume. If no complete checkpoint exists, restart that stage. The stage, method, network, controller, seed, goal and parent/WCE identities must agree; the original total stage budget is not an extra budget added after resume.

**Pilot and publication online-WCE resume examples:**

```bash
export CBWCE_RESUME_PILOT='REPLACE_WITH_COMPLETE_SAME_STAGE_PILOT_CHECKPOINT'
python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --method online_wce --parent "$CBWCE_PARENT_PILOT" --wce "$CBWCE_WCE_PILOT" \
  --pilot --visualization --steps 2640 --checkpoint-every 1 --resume "$CBWCE_RESUME_PILOT"

export CBWCE_RESUME='REPLACE_WITH_COMPLETE_SAME_STAGE_PUBLICATION_CHECKPOINT'
python main.py experiment continue --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --method online_wce --parent "$CBWCE_PARENT" --wce "$CBWCE_WCE" \
  --gate "$CBWCE_GATE" --resume "$CBWCE_RESUME"
```

For parent resume use `parent --resume ...` without parent/WCE arguments; for offline WCE resume use `wce --parent ... --resume ...` without a pretrained `--wce`. Retain the original pilot `--steps` or `--episodes` when applicable. Do not reuse `--output` from the interrupted run. Select the complete checkpoint explicitly; the dashboard's “last checkpoint” refers to that selected job, not a global latest-file search.

## 8 Evaluation timing and completion checks

### Stage IV Freeze evaluation inputs

**Required checks:** C01, C02, C04, C06–C10. Controller and WCE parameter learning is disabled during evaluation.

Evaluate the exact 2,320,000-step controller checkpoints on eleven seen profiles and twelve new scenarios per network. Use ten paired rollouts per scenario, a 3,600-second horizon, an empty initial network, and no drainage extension. Include startup in the primary average. Sample IA2C/MA2C/PPO actions and use greedy IQL actions, resetting the independent policy stream for every rollout.

Materialize complete demand artifacts before evaluation. Reuse a given scenario/arrival realization across all methods and controller seeds, together with its prescribed SUMO seed. Retain actual insertion, pending departures, and residual traffic as outcomes. WCE does not adapt the held-out test demand during this common evaluation.

### Stage IV commands, scripts and output nesting

**Purpose/prerequisites:** Evaluate one explicitly selected final controller against frozen traffic artifacts. C01/C02/C09/C10 apply to every rollout. Use the completed continuation's `result.json.checkpoint`; keep network/controller/training seed consistent. Set only the applicable variable:

```bash
export CBWCE_FINAL_PILOT='REPLACE_WITH_SELECTED_COMPLETED_PILOT_CONTINUATION_CHECKPOINT'
export CBWCE_FINAL='REPLACE_WITH_SELECTED_COMPLETED_PUBLICATION_CONTINUATION_CHECKPOINT'
```

**Pilot command:** one complete 3,600-second realization of the 1.25× peak; the model can have pilot learning counts.

```bash
python main.py experiment evaluate --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --parent "$CBWCE_FINAL_PILOT" --pilot --visualization --suite test --scenario peak_1.25 --rollouts 1
```

**Publication command:** the controller must have exactly 2,320,000 learning steps; each scenario receives ten realizations. This evaluates one controller, not the whole study. The CLI evaluator checks checkpoint identity/budget but does not itself require/validate `--gate` as training stages do, so run the explicit gate check first. The local launcher requires a valid gate for publication evaluation too.

```bash
python main.py experiment check --gate "$CBWCE_GATE"
python main.py experiment evaluate --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_SEED" \
  --parent "$CBWCE_FINAL" --gate "$CBWCE_GATE" --suite all --rollouts 10
```

**Suite selectors:** `--suite seen` = eleven seen profiles; `--suite test` = twelve new scenarios; `--suite validation` = six independent validation scenarios; `--suite all` = seen + test only. `--scenario` selects an exact ID within that suite, for example `peak_1.25` or `peak_1.5`. `--rollouts` accepts 1–10 for pilots; non-pilot suite evaluation requires ten. Validation example:

```bash
python main.py experiment evaluate --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --parent "$CBWCE_FINAL_PILOT" --pilot --visualization --suite validation --rollouts 1
```

**Single-artifact pilot/retry:** omit `--suite`, explicitly supply the complete artifact and its paired simulator/policy seeds. Below reproduces suite rollout index 0 (arrival 51001, SUMO 61001). For other repetitions use the original artifact, seeds and checkpoint from that attempt's manifest. The direct evaluator's default policy seed `71001` is not the suite-derived policy seed, so do not rely on it for paired retries.

```bash
export CBWCE_ARTIFACT="$CBWCE_DATASET_ROOT/test/artifacts/peak_1.25_51001.json"
export CBWCE_POLICY_SEED="$(python -c 'import os; from experiments.core import digest; print(int(digest(["evaluation-policy", int(os.environ["CBWCE_PILOT_SEED"]), 0])[:8], 16))')"
python main.py experiment evaluate --network "$CBWCE_NETWORK" \
  --controller "$CBWCE_CONTROLLER" --seed "$CBWCE_PILOT_SEED" \
  --parent "$CBWCE_FINAL_PILOT" --pilot --visualization --artifact "$CBWCE_ARTIFACT" \
  --sumo-seed 61001 --policy-seed "$CBWCE_POLICY_SEED"
```

**Scripts:** [experiments/cli.py](experiments/cli.py) enumerates suites and generates/reuses artifacts through [experiments/scenarios.py](experiments/scenarios.py). [experiments/runner.py](experiments/runner.py) runs the frozen controller, exports measurements and trip summaries; [experiments/core.py](experiments/core.py) validates the exact timestamps. The selected continuation's method is inferred from its origin manifest; evaluating `online_wce` therefore labels its results correctly even without an explicit `--method`.

**Output and completion:**

```text
runs_eval/revised/<network>/<controller>/seed_<seed>/evaluate/<run_id>/
  suite.json
  suite_result.json                         # written only on complete suite
  demand_runtime/                           # route-resolution startup/trips
  <seen|test|validation>/<scenario>/rollout_01/attempt_001/
    manifest.json
    environment.json
    progress.jsonl
    rollout.npz
    rollout.jsonl
    rollout.controls.jsonl
    rollout_summary.json
    result.json
    runtime/
```

One selected `all` suite has 230 completed rollouts. Require each rollout's `status=complete`, `sample_count=3600`, exact timestamps 1…3600, correct checkpoint/demand hashes and effective seed. A standalone artifact evaluation writes the rollout files directly in its run directory, without the suite nesting. Failed attempts retain `result.json` and, if measurements exist, `incomplete_attempt.*`; they do not yield valid summary rows.

Training `--resume` is not a suite-skip/resume feature. A new suite attempt starts its chosen scenarios/rollouts from the beginning. To rerun only a particular failed realization, use its single artifact and exact paired seeds in a new `--output` directory under the evaluation root. Never overwrite `attempt_001` or report duplicate valid attempts as extra replicates.

### New scenarios and seed assignments

<a id="traffic-generation"></a>

#### 1. Start from each network's eleven OD profiles

Validation and test traffic are **synthetic scenarios derived from the existing OD-rate profiles**, not new measured traffic recordings and not scenarios produced by the trained WCE. An OD row specifies an origin road edge, a destination road edge, and a requested rate in vehicles/hour. The same generation procedure is applied separately to the two networks:

| Network | Original profile source | Prepared dataset root | Reference total rate |
|---|---|---|---:|
| Grid | `data_traffic/demand_<name>.csv` | `data_traffic/revised/` | 3,000 veh/hour |
| Monaco | `real_net_subnet/demand_groups/<name>.csv` | `real_net_subnet/demand_groups/revised/` | 2,383.3333 veh/hour |

For each source profile, [original_profiles](experiments/demand.py) multiplies every OD rate by `reference_total / original_total`. This preserves its OD proportions while giving all eleven profiles the same total within a network. [prepare](experiments/prepare.py) writes separate normalized training CSVs and an ordered hash manifest; original CSVs remain unchanged. The active road networks are `large_grid/data/exp.net.xml` and `real_net_subnet/data/in/most.net.xml`.

**Ordering matters:** Grid uses alphabetical order of the eleven profile names; Monaco uses `ORDER` in `experiments/demand.py`: `N_to_S`, `S_to_N`, `W_to_E`, `E_to_W`, `NW_to_SE`, `SE_to_NW`, `SW_to_NE`, `NE_to_SW`, `Periphery_to_Center`, `Center_to_Periphery`, `Uniform`. Read each `train/manifest.json` when interpreting an eleven-element mixture vector. Reusing a numeric seed across networks does not imply identical OD allocations, routes or vehicle schedules.

#### 2. Build fixed 3,600-second scenario definitions

[definitions](experiments/scenarios.py) creates the schedule; [block_rows](experiments/scenarios.py) turns each block into OD rates. Every scenario covers seconds `[0,3600)` without gaps. The evaluation block lengths below can differ from the 600-second WCE training blocks.

| Family | Test scenarios / generation seeds | Validation scenarios / generation seeds | Construction and purpose |
|---|---|---|---|
| OD redistribution | `redistribution_0.25`, `redistribution_0.5`, `redistribution_0.75` / `41001–41003` | `redistribution_0.35`, `redistribution_0.65` / `31001–31002` | Multiply Uniform-profile OD rates by lognormal factors with the named log-space SD; renormalize to reference total. Test new spatial allocations at fixed load. |
| Convex mixtures | `mixture_1`, `mixture_2`, `mixture_3` / `41004–41006` | `mixture_1`, `mixture_2` / `31003–31004` | Draw one `Dirichlet(1,…,1)` vector per scenario and combine eleven profiles. Keep that mixture fixed for all 3,600 seconds. |
| Temporal switching | `switch_300`, `switch_900`, `switch_1200` / `41007–41009` | `switch_450` / `31005` | Start with `N_to_S`, alternate with `W_to_E` at the named interval, and stop at 3,600 seconds. Test new demand timing at fixed total rate. |
| Peak intensity | `peak_1.1`, `peak_1.25`, `peak_1.5` / `41010–41012` | `peak_1.15` / `31006` | Uniform demand at reference rate for `[0,1200)`, multiplied by the named factor for `[1200,2400)`, then reference rate for `[2400,3600)`. Test higher intensity. |

For OD redistribution, with normalized Uniform rates $u_i$ and network reference rate $R$:

$$
z_i\sim\operatorname{Lognormal}(0,\sigma),\qquad
r_i=R\frac{u_i z_i}{\sum_j u_j z_j}.
$$

Zero rates remain zero; this does not create new OD connections. Here, sigma is the standard deviation of the underlying normal distribution, not the coefficient of variation of traffic counts. For mixtures, the rate of OD pair $i$ is $r_i=\sum_{g=1}^{11}w_g r_{g,i}$; missing OD entries contribute zero, repeated pairs are summed, and $\sum_g w_g=1$. Mixtures are within the demand family accessible to WCE: describe them as compositional generalization, not automatically out-of-distribution demand.

The temporal and peak schedules are deterministic: their `generation_seed` is a recorded identifier, not a random draw used to choose the switch/peak settings. The redistribution and mixture families actually use that seed for their random factors/weights. Test settings are currently specified in `experiments/scenarios.py`; validation settings and seed lists are recorded in [protocol.json](config/revised/protocol.json). Keep the generated definitions frozen before training/tuning.

#### 3. Turn rates into complete vehicle schedules

A scenario definition is not yet a list of vehicles. [artifact_for](experiments/scenarios.py) creates a fresh `RandomState(arrival_seed)` and passes it through all blocks in order. For every positive-rate OD pair in a block starting at $s$, of duration $d$ seconds, [materialize](experiments/demand.py) does the following:

1. Resolve a route in that network through SUMO `findRoute(..., vType='type1')` and store its ordered road edges. Reject an unavailable or endpoint-inconsistent route, even if the sampled vehicle count would be zero. The environment caches routes for repeated OD pairs; controllers do not choose the routes.
2. Draw the count from $N\sim\operatorname{Poisson}(r\,d/3600)$. A rate is an expectation, not an exact number of vehicles.
3. Draw departure times uniformly within the block, add normal jitter with mean 0 and SD **2 seconds**, clip to `[s+0.01, s+d−0.01]`, and round to two decimal places. The resulting departure distribution includes this jitter/clipping; it is not simply an untouched uniform sample.
4. Draw each vehicle's `speed_factor` from a normal distribution with mean **1.0** and SD **0.1**, redrawing nonpositive values. This is a dimensionless factor, not a target speed in m/s.
5. Save a unique block-prefixed ID, departure time, OD endpoints, full route and speed factor; sort vehicles by departure time and ID.

For example, Grid `peak_1.25` requests 3,000 → 3,750 → 3,000 veh/hour across three 20-minute blocks. Its expected one-hour count is **3,250**, but the existing `peak_1.25_51001.json` schedules **3,335** vehicles. Monaco uses approximately 2,383.3333 → 2,979.1666 → 2,383.3333 veh/hour, with an expected count of **2,581.9444** and **2,632** vehicles in its corresponding saved artifact. These are counts in the existing inputs, not performance outcomes or guaranteed realized insertions.

#### 4. Separate seed roles and reuse paired artifacts

| Seed role | Values | Effect |
|---|---|---|
| Scenario generation | Test `41001–41012`; validation `31001–31006` | Spatial factors/mixture weights, or identifiers for deterministic schedules |
| Arrival realization | `51001–51010` for every scenario on each network | Poisson counts, departure jitter and speed factors; ten complete artifacts per scenario |
| SUMO evaluation | `61001–61010`, paired by rollout index | Simulator randomness after scheduled traffic is fixed; materialization uses an empty route-resolution session with seed `61001` |
| Controller policy | Derived from training seed and rollout index by `experiments/cli.py` | Action sampling only; does not regenerate the traffic artifact |

The arrival seed list is reused across scenarios, splits and networks; the current implementation does **not** assign wholly disjoint RNG streams to validation and test arrivals. Their scenario definitions/parameters and artifacts are separate. The same seed can produce different schedules when OD rows or rates change. For fair comparisons, reuse the exact artifact hash for the same network/scenario/arrival realization across all methods and controller seeds. Congestion can change actual insertion and completion even when the scheduled traffic is identical.

Validation is for development and parameter choices; final test results must not choose parameters or checkpoints. Test mixtures `mixture_1` and `mixture_2` are different from their same-named validation counterparts because seeds and split-specific paths differ. Do not put validation/test files into training inputs. The eleven `seen` scenarios repeat each normalized training profile for one hour with held-out arrival realizations; `--suite all` means eleven seen plus twelve test scenarios, excluding validation.

#### 5. Generate, locate and inspect the saved files

These commands prepare inputs; materialization starts SUMO for route resolution but does not train a controller. They are instructions, not commands executed during this documentation update:

```bash
python main.py experiment prepare
python main.py experiment prepare --materialize --network grid
python main.py experiment prepare --materialize --network monaco
```

Under either prepared dataset root:

```text
train/<profile>.csv
train/manifest.json
validation/validation_scenarios.json
validation/artifacts/<scenario>_<arrival_seed>.json
test/test_scenarios.json
test/seen_scenarios.json
test/artifacts/<scenario>_<arrival_seed>.json
```

There are **60 validation + 120 test + 110 seen = 290 traffic artifacts per network**, **580 total**. Seen artifacts are physically under `test/artifacts/`, but carry `scenario.split = seen`. A generation attempt also writes its artifact index under `runs_eval/revised/preparation/<run_id>/manifest.json`. No controller family/method is embedded in the dataset path because these are shared exogenous inputs.

Scenario JSON contains `id`, `network`, `split`, `family`, `generation_seed`, `horizon` and `blocks`. Each traffic JSON additionally contains the network/profile/scenario hashes, `arrival_seed`, `vehicles`, and its own content `hash`; each vehicle has `id`, `depart`, `origin`, `destination`, `edges` and `speed_factor`. These are full vehicle schedules in JSON, not one new CSV per test scenario. At evaluation, `envs/experiment_env.py` inserts the saved routes and vehicles through TraCI.

`prepare` refuses conflicting prepared content. An existing artifact is reused only after its hash, unique vehicle IDs and expected metadata match. These reuse checks do not rerun every possible route/schema validation; route resolution and schedule checks occur during generation. If inputs change, preserve the old artifacts and version the protocol/output location rather than overwrite or silently reuse them. During this documentation update, all **580 existing artifacts** passed read-only content-hash/ID checks and matched their current scenario definitions; no artifacts were regenerated and no SUMO simulation was launched.

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

### Stage V reports and local dashboard

**Purpose/prerequisites:** Convert explicitly selected complete records into scientific tables/figures. Use the valid manifests, raw lane measurements and original traffic artifacts; report generation verifies their provenance. No controller learns during reporting.

**Pilot command:** the following uses existing pilot evidence and creates a new report. Run it only if the example destination does not already exist; otherwise choose another new name.

```bash
python main.py experiment report \
  --input runs_eval/revised/peak_validation \
  --input runs_eval/revised/verification/integrated_20260914/grid_iqll/baseline \
  --input runs_eval/revised/verification/integrated_20260914/monaco_iqll/baseline \
  --output output_result/revised/pilot_report_20260915_example
```

**Publication command template:** use selected publication attempt directories. Repeat `--input` for additional methods, training seeds, parents or WCE runs. The reporter classifies each row using its manifest, so no `--pilot` flag is accepted by `report`. Do not point it at a broad directory containing alternate valid reruns of the same comparison cell.

```bash
export CBWCE_EVAL_RUN='REPLACE_WITH_ONE_SELECTED_EVALUATION_SUITE_OR_ATTEMPT_DIRECTORY'
export CBWCE_TRAIN_RUN='REPLACE_WITH_SELECTED_COMPLETED_CONTINUATION_RUN_DIRECTORY'
export CBWCE_REPORT_DIR='output_result/revised/REPLACE_WITH_NEW_REPORT_ID'
python main.py experiment report --input "$CBWCE_EVAL_RUN" \
  --input "$CBWCE_TRAIN_RUN" --output "$CBWCE_REPORT_DIR"
```

**Scripts:** [experiments/reporting.py](experiments/reporting.py) collects records, checks pairing, computes summaries, exports queue curves and peak maps. **Output:** all report CSV/JSON/PNG/SVG files go inside the selected report directory. It must be new. Duplicate/unpaired observations cause an explicit failure with `rejected.json`; other rejected summaries are listed separately. Attempts without a rollout summary remain in their original failure records and may not appear in that rejected-summary table.

**Completion:** verify valid rollout counts, expected/missing study cells, paired demand/SUMO/policy seeds, and heatmap lane totals. Five complete prescribed training seeds are needed for a publication 95% interval. Available subset means are not full-study estimates. Heatmaps use the common intersection of available `(seed, arrival_seed)` pairs across all five methods; check the reported pair count and completeness before publication. Figure copies in `figs/revised/` are separate retained copies, not an automatic second output from `report`.

**Existing pilot outputs:** [pilot tables and exports](output_result/revised/pilot_integrated_final_report/), [dashboard JSON](output_result/revised/pilot_integrated_final_report/dashboard.json), [figure copies](figs/revised/pilot_integrated_final_report/). They contain 20 peak rollouts, not a completed publication study. Training curves are sampled progress, not smoothed episode statistics or evaluation performance curves.

**Dashboard command:** choose one of these ports, not both for a single workspace:

```bash
python main.py experiment dashboard
python main.py experiment dashboard --port 8766
```

Open [English](http://127.0.0.1:8765/) or [中文](http://127.0.0.1:8765/zh.html), using the selected port. [experiments/dashboard.py](experiments/dashboard.py) launches jobs with the active Python executable, one at a time. Select the stage, pilot/publication mode, network, controller, seed, method and explicit compatible checkpoints; inspect preflight before starting. Stop preserves the interrupted attempt; resume creates another attempt. Server restarts do not silently restart training. Job request/start/finish/recovery JSON and `console.log` are under `runs_eval/revised/jobs/<job_id>/`; ordinary CLI commands write console output to your terminal unless you capture it.

For plots, open Results, choose “Load local result export” and select the generated `dashboard.json`. Filter network/controller/pilot status; choose a training record or peak map explicitly. The [English HTML](docs/site/dist/index.html), [Chinese HTML](docs/site/dist/zh.html) and [private Sites guide](https://cb-wce-training-workspace.loyal-bowl-4834.chatgpt.site) provide instructions and exported results. Only the local service can launch training; importing a report locally does not automatically republish the private site.

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

## 9 Linked implementation index and artifact dictionary

| Script | Responsibility |
|---|---|
| [main.py](main.py) → [experiments/cli.py](experiments/cli.py) | Public `experiment` commands, argument parsing, suite orchestration |
| [experiments/protocol.py](experiments/protocol.py) / [protocol.json](config/revised/protocol.json) | Shared settings and default output roots; effective INIs are in `config/revised/` |
| [experiments/prepare.py](experiments/prepare.py) / [experiments/demand.py](experiments/demand.py) | Explicit training order, normalized profiles, hashes and traffic materialization |
| [experiments/scenarios.py](experiments/scenarios.py) | Seen/test/validation definitions and paired artifacts |
| [experiments/runner.py](experiments/runner.py) | Single parent/WCE/continuation/evaluation interaction loop |
| [agents/controller.py](agents/controller.py) / [agents/recurrent.py](agents/recurrent.py) | Policy actions, fingerprints, masked batches, update cadence and recurrent state |
| [agents/wce.py](agents/wce.py) / [agents/policies.py](agents/policies.py) | WCE adapter and existing policy architectures |
| [envs/experiment_env.py](envs/experiment_env.py) / [experiments/core.py](experiments/core.py) | SUMO stepping, queue/reward measurement, RNG and run records |
| [experiments/checkpoint.py](experiments/checkpoint.py) | Complete bundles, integrity, restoration and originating-run identity |
| [experiments/reporting.py](experiments/reporting.py) | Scientific collection, statistics, training curves, peak maps and exports |
| [experiments/dashboard.py](experiments/dashboard.py) | Local preflight, launch/status/stop, run catalog and reports |
| [experiments/verify.py](experiments/verify.py) / [tests/test_corrections.py](tests/test_corrections.py) / [tests/test_workflow.py](tests/test_workflow.py) / [tests/test_visualization.py](tests/test_visualization.py) | Integration gates and deterministic correctness tests |

Historical `main.py train/evaluate`, adversary/coevolution scripts and benchmark scripts retain their previous paths and behavior. `revision.*` modules are compatibility wrappers, not a second corrected implementation. Use the `experiment` interface above for this study.

### Input, training and evaluation records

`<dataset>` means `data_traffic/revised` for Grid or `real_net_subnet/demand_groups/revised` for Monaco. `<run>` is the stage directory and `<attempt>` is a single evaluation directory. JSONL contains one complete JSON object per line; JSON is a complete document. CSV uses a header row; nested report values such as cost components are JSON-encoded inside CSV cells.

| Artifact / producer | Location / format | Fields and units | Use |
|---|---|---|---|
| Training CSV / `prepare` | `<dataset>/train/<profile>.csv` · CSV | `origin_edge`, `dest_edge`, `veh_per_hour`; rates in veh/hour; one OD per row | Controller/WCE demand input |
| Training manifest / `prepare` | `<dataset>/train/manifest.json` · JSON list | Ordered entries: `name`, `source`, `source_hash`, `original_total`, `scale`, `prepared`, `prepared_hash`; `rows` is null in this index | Check original rates, normalization factor, order and input integrity |
| Scenario definitions / `prepare` | `<dataset>/test/{seen,test}_scenarios.json`; `<dataset>/validation/validation_scenarios.json` · JSON lists | `id`, `network`, `split`, `family`, `generation_seed`, `horizon`, `blocks`; seconds and dimensionless multipliers/weights | Freeze spatial/temporal scenario construction |
| Traffic artifact / scenario materializer | `<dataset>/<test\|validation>/artifacts/<scenario>_<arrival_seed>.json` · JSON | Network/profile/scenario hashes, `horizon`, `arrival_seed`, `scenario`, `vehicles`, `hash`; vehicle keys `id`, `depart`, `origin`, `destination`, `edges`, `speed_factor` | Complete scheduled traffic; count = length of `vehicles`; departures in seconds, speed factor dimensionless; seen artifacts also live under test |
| Run identity / `RunRecord` | `<run>/manifest.json` · JSON | Stage/method/seed, CLI inputs including `visualization`, parent paths/hashes, training profiles, configuration/source hashes, runtime information, `manifest_hash` | Immutable actual inputs; joins results to their provenance |
| Environment identity / runner | `<run>/environment.json` · JSON | `assets`, `lanes`, `nodes`, `node_lanes`, `neighbors`, `configuration` | Lane order and geometry/network identity; contains copied settings, including legacy fields |
| Completion / `RunRecord.finish` | `<run>/result.json` · JSON | `status`, `manifest_hash`, `wall_seconds`, `component_seconds`; training adds `checkpoint`, `learning_steps`, `stage_simulation_steps`, `backward_calls`, `minibatch_updates_per_agent`, `wce_updates`; failures include error/completed counters | Authoritative completion/cost record; elapsed values in real seconds |
| Progress / runner | `<run>/progress.jsonl` · JSONL | `stage`, `episode`, `simulation_steps`, `learning_steps`, `goal`, `mean_queue`, `wce_updates`, `wall_seconds` | One record every 120 joint transitions; mean queue over the most recent 600 simulated seconds; not one record per optimizer update or an episode mean |
| Demand decisions / runner | `<run>/demand_decisions.jsonl` · JSONL | `episode`, `block`, eleven `weights`, `wce_reward`, `completed_seconds`, `scheduled_vehicles`, `traffic_hash` | One block record; zero-based episode/block indexes; partial final parent block has null WCE reward; training records hashes/counts rather than complete vehicle artifacts |
| Lane arrays / `export_episode` | `<run>/episode_0001.npz` or `<attempt>/rollout.npz` · compressed NumPy archive | `time`: shape `(T,)`, seconds 1…T; `lanes`: shape `(L,)`, ordered lane IDs; `queue`: shape `(T,L)`, stopped vehicles per lane | Full training episode T=6600; evaluation T=3600; L=150 Grid / 116 Monaco; partial parent/incomplete records can be shorter |
| Traffic series / `export_episode` | Same episode/rollout stem + `.jsonl` · JSONL | `time`, `queue`, `active`, `speed_sum`, `inserted`, `completed`, `pending`, `teleports`, `collisions` | Every simulated second; counts in vehicles, `speed_sum` sums active-vehicle speeds in m/s |
| Learner reward series / `export_episode` | Same episode/rollout stem + `.controls.jsonl` · JSONL | `time`, `learner_rewards` vector in controller-node order | Every five simulated seconds; already family-transformed and divided by 100 |
| Rollout outcome / evaluator | `<attempt>/rollout_summary.json` · JSON | Queue metrics, speed and denominator, scheduled/inserted/completed/remaining/pending, events, completed-trip metrics, demand hash, effective SUMO seed, checkpoint identity | Queue in vehicles; integrated queue in vehicle-seconds; speed m/s; travel/wait/time-loss/departure-delay in seconds. Join manifest for method/training/policy seed |
| Runtime records / environment | `<run>/runtime/startup_<attempt>.json`, `trips_<attempt>.xml`, SUMO assets | Startup command, actual `visualization`, seed/version provenance; SUMO `tripinfo` vehicle records | Use the startup command to identify the matching trip file. Completed-trip averages exclude unfinished trips and state their denominator |

`simulation_steps` means five-second joint transitions, not seconds. Progress `episode` is one-based, while demand-decision `episode` is zero-based; NPZ filenames start with `episode_0001`. A final partial episode can finish between progress writes, so use `result.json` and the checkpoint counters for completion. An offline WCE progress curve has constant controller-learning steps while simulation steps and WCE updates grow.

Complete evaluation demand persists all vehicle routes/departures/speed factors. Requested demand is paired across methods; congestion can change actual insertion, pending departures and completed trips. Training demand is generated block by block with its dedicated RNG and recorded hash; a hash is not itself a saved vehicle schedule.

### Checkpoint bundle and numbering

| Publication stage | Final directory | Controller-learning count |
|---|---|---|
| Parent | `checkpoint_001000000` | 1,000,000 |
| Offline WCE | `checkpoint_000660000` | Still 1,000,000 |
| Continuation | `checkpoint_001320000` | 2,320,000 |

```text
<run>/manifest.json
<run>/checkpoint_<nine-digit-stage-steps>/
  manifest.json
  runner.pkl
  controller/
    variables.index
    variables.data-00000-of-00001
    checkpoint
    state.pkl
  wce/                         # present only when WCE is part of the stage
    variables.index
    variables.data-00000-of-00001
    checkpoint
    state.pkl
```

The checkpoint manifest records `version`, model `signatures`, file hashes, `parents` and bundle `hash`; it is written last to mark completeness. TensorFlow files contain model and optimizer variables. Controller `state.pkl` stores real learning/schedule/update counters, pending/replay data, replay cursor, RNG and recurrent states; WCE state stores pending macro transitions, macro/update counters and RNG. `runner.pkl` stores stage/method, goal, stage steps, episode, initial learning count and runner RNG. These pickle files are Python-specific state, not interchange CSVs. Load through the checkpoint manager rather than reconstructing a model from filenames.

### Report outputs and timing interpretation

| Filename inside report directory | Meaning |
|---|---|
| `rollouts.csv` | One valid observation per method/training seed/scenario/arrival realization; metrics and provenance |
| `scenarios.csv` | Ten-rollout mean/sample SD and completeness per training seed/scenario; pilot rows remain marked |
| `seeds.csv` | Equal-scenario mean for each prescribed publication training seed, separately for seen/test |
| `comparison.csv`, `paired_differences.csv` | Five-seed mean/SD/95% interval and paired online-minus-comparator differences; missing intervals remain unavailable |
| `demand_family_summary.csv`, `worst_tested_scenario.csv` | Family-level publication summaries and worst tested scenario only with the required complete records |
| `computation_costs.csv`, `standalone_pipeline_costs.csv` | Recorded stage cost and parent + continuation + required offline-WCE cost; unavailable parent records leave pipeline total unavailable |
| `rejected.csv`; `rejected.json` on duplicate/pairing failure | Rejected existing summaries; inspect original attempt records too for failures that never produced a summary |
| `dashboard.json` | Rollouts, scenario/seed comparisons, curves, maps, families, worst cases and costs; local HTML import format |
| `training_<index>.png/.svg` | Saved progress queue vs controller-learning steps; includes frozen WCE stages if selected, where controller steps stay constant |
| `<network>_<controller>_<scenario>_<pilot\|publication>_<method>.png/.svg` | Absolute peak maps; difference names use `online_minus_<comparator>` |

`wall_seconds` is the monotonic elapsed time for the attempt; it is the stage total. Current `component_seconds` keys include `reset`, `demand_selection` (including WCE inference), `demand_generation_insertion`, `controller_inference`, `simulation_measurement`, `controller_learning`, `wce_learning`, and `checkpoint` when exercised. They are measured portions, not an exhaustive partition: final flush/logging and other overhead are not all separately timed. Do not claim a separate complete logging or WCE-inference ledger, or replace wall time with the component sum. Report failed/interrupted attempts and shared offline-WCE cost separately as specified above.

### Read an existing rollout without training

This read-only example uses a recorded Grid/IQL pilot. NPZ times mark the end of each one-second interval; selecting `time > 1200` and `time <= 2400` corresponds to the physical window `[1200,2400)`. No new experiment is required.
```python
import json
from pathlib import Path
import numpy as np

attempt = Path("runs_eval/revised/verification/integrated_20260914/grid_iqll/eval_baseline")
with np.load(str(attempt / "rollout.npz"), allow_pickle=False) as data:
    lane_ids = data["lanes"]
    time = data["time"]
    queue = data["queue"]
    assert len(time) == 3600
    mean_total_queue = float(queue.sum(axis=1).mean())
    peak_window = (time > 1200) & (time <= 2400)
    mean_peak_lane_queue = queue[peak_window].mean(axis=0)
with (attempt / "rollout.jsonl").open() as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
summary = json.loads((attempt / "rollout_summary.json").read_text())
assert np.isclose(mean_total_queue, summary["mean_queue"])
vehicle_seconds = sum(row["active"] for row in rows)
mean_speed = (sum(row["speed_sum"] for row in rows) / vehicle_seconds
              if vehicle_seconds else None)
print(queue.shape, mean_total_queue, mean_speed)
```
