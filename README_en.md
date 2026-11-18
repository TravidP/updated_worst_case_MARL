# CB-WCE Standalone: Directory Guide and Operations Manual

Current branch: [`refactor/standalone-local-training`](https://github.com/TravidP/updated_worst_case_MARL/tree/refactor/standalone-local-training). This directory is the branch root. Local CPU execution is the current target; cluster templates are optional. See the [repository layout](docs/repository_structure_zh.md).

[中文 README](README.md) · [English experiment plan](docs/revision_plan_v8_en.md) · [Chinese plan](docs/revision_plan_v8_zh.md) · [Implementation evidence](docs/implementation_status.md)

This standalone project compares MA2C, DQL and PPO, and five continuation methods, on a **5×5 Grid** and the **Monaco subnet used by the revised runner**. Runtime source, necessary networks and original demand profiles were extracted through a whitelist. The directory has independent Git history and can be relocated without the original workspace. See [extraction provenance](provenance/extraction.json) and [LICENSE](LICENSE).

This README describes the files and executable interfaces that actually exist. The bilingual plans describe the experimental design; verification records describe what has been checked. These are different artifacts.

## 1. Version and readiness

| Item | Current state |
|---|---|
| Runtime/manifest protocol | v8.1, defined in `config/protocol.json` |
| Latest experiment plan | v8.2; filenames remain `docs/revision_plan_v8_en.md` and `_zh.md` |
| Implemented controller validation | 3600 seconds per scenario; three routine realizations, ten for confirmation |
| Planned v8.2 controller validation | Eleven pregenerated DR mixtures, provisionally 600 seconds each, one continuous 6600-second sequence with three complete realizations; **not implemented yet** |
| Duration awaiting confirmation | The request also mentioned 60 seconds/group: 11×60=660, whereas 11×600=6600; the plan provisionally adopts the latter |
| Offline WCE validation | 6600 seconds, ten routine seed combinations, thirty for confirmation |
| Final tests | Provisionally eleven seen and twelve generalization scenarios, ten realizations each, 3600 seconds each |
| Existing evidence | Eleven unit tests, six network/controller pilots, thirty continuation branches, recovery, audit and relocation checks |
| Outstanding | Publication validation manifests, SIF build/container acceptance, Hábrók measurements, publication training and publication-model archives |

**Do not pass a 6600-second manifest to the existing v8.1 validator or reuse old selection scores/best/patience under the new design.** Synchronize code, protocol, cache identities and tests first; see Section 8 of the plan.

[Saved verification evidence](provenance/verification) uses short 110-second training episodes and small replay buffers; separate evaluation checks used full 3600-second horizons. This is correctness evidence, not convergence or cluster-memory certification. Temporary pilot checkpoints have been removed; rerun the verifier to recreate them.

## 2. Suggested workflow

1. Read the version distinction above and the experiment plan.
2. Inspect directories, inputs and configuration; run local checks and pilots.
3. Understand checkpoint roles, stage branching, recovery and result exports.
4. Implement the new protocol, freeze manifests, build/verify the image and measure resources.
5. Release source/inputs, submit successive Hábrók stages, then verify archives and independent backups.

## 3. Directory and file reference

```text
cbwce_standalone/
├── README.md / README_en.md      # Chinese / English manuals
├── main.py                      # Unified CLI
├── tf_compat.py                 # TensorFlow compat.v1 bridge
├── LICENSE
├── agents/                      # Controllers, WCE and network layers
├── envs/                        # SUMO, observations, phases and measurements
│   └── grid_builder.py          # Grid helper now inside envs/
├── experiments/                 # Demand, validation, selection and persistence
├── config/protocol.json         # Current experiment contract
├── config/revised/              # Controller and WCE INIs
├── assets/grid/                 # network.net.xml, detectors.add.xml, demand/
├── assets/monaco/               # Same layout, minimal Monaco inputs
├── tests/                       # Unit and state-consistency checks
├── scripts/                     # Verification, evaluation, exports, releases
├── scripts/slurm/               # Job matrices, resource template and launcher
├── packaging/                   # Locks, image definition and release whitelist
├── provenance/                  # Extraction and verification records
└── docs/                        # Plans and implementation status
```

### Controllers and environments

| File | Responsibility |
|---|---|
| [agents/controller.py](agents/controller.py) | MA2C/PPO rollouts, masked losses, advantage normalization and state; delegates `dql` construction |
| [agents/dql.py](agents/dql.py) | Independent 128/64 MLPs, online/target networks, Huber loss, epsilon, fixed-array replay and chunked persistence |
| [agents/recurrent.py](agents/recurrent.py) | Compact recurrent graphs and fingerprint encoding |
| [agents/wce.py](agents/wce.py) | Grid CNN-A2C / Monaco GCN-A2C demand-weight policies and frozen inference |
| [agents/policies.py](agents/policies.py), [agents/utils.py](agents/utils.py) | Reused policies, layers, initializers and helpers |
| [envs/experiment_env.py](envs/experiment_env.py) | Actual revised environment: TraCI, traffic injection, per-second measurements, streaming metrics and audit output |
| [envs/env.py](envs/env.py) | Base simulator, phase, observation and node definitions |
| [envs/large_grid_env.py](envs/large_grid_env.py), [envs/real_net_env.py](envs/real_net_env.py) | Network-specific topology and observation interfaces |

Only `ma2c`, `dql` and `ppo` are exposed by the CLI. Retained legacy helper classes do not enable IA2C/IQLL experiments. MA2C/PPO use separate actor/critic networks with LSTM 64. DQL uses wave 128 encoders → LSTM 64 → Q, with identical online/target architectures. DQL uses `sequence_length=16` for training and inference: each batch samples 64 endpoints and reconstructs contiguous histories, stopping at episode boundaries and masking left padding. Full checkpoints preserve inference histories; old feedforward DQL checkpoints are incompatible. Architecture and hyperparameter fairness are explained in plan Section 3.

Both Grid and Monaco use **wave-only traffic observations** for MA2C, DQL, PPO and WCE. Controllers retain local/neighbor wave inputs and MA2C retains neighbor fingerprints. WCE concatenates only padded per-node wave vectors: Grid CNN input is 25 × 6 = 150 (6 channels, previously 300 / 12 channels); Monaco GCN input is 28 × 11 = 308 (each node padded to 11 monitored lanes). Wave retains the existing measurement semantics: detector-area vehicle counts in Grid and lane vehicle counts in Monaco, divided by `norm_wave` and clipped by `clip_wave`. Waiting time is not an input; queue-based rewards are unchanged. The six controller INIs enforce `observation = wave`; unused `num_ft`, `norm_wait`, `clip_wait` and `fast_wait_metric` settings have been removed. New observation signatures reject checkpoints from the previous observation schema; restart training and regenerate dependent WCE checkpoints.

### Experiment modules

| File | Responsibility |
|---|---|
| [experiments/runner.py](experiments/runner.py) | Stage loops, preflight, validation, segmentation and recovery |
| [experiments/protocol.py](experiments/protocol.py) | Protocol, paths, config locations and portable identities |
| [experiments/configuration.py](experiments/configuration.py) | Required INI fields, types and value checks |
| [experiments/demand.py](experiments/demand.py) | Original profiles, normalization, mixtures, vehicle generation and hashes |
| [experiments/scenarios.py](experiments/scenarios.py) | Seen/generalization test definitions and transformations; publication validation requires explicit manifests |
| [experiments/validation.py](experiments/validation.py) | Manifest validation, traffic cache, fixed random streams and inference workers |
| [experiments/selection.py](experiments/selection.py) | Best, meaningful-improvement anchor, patience, plateau and common-budget selection |
| [experiments/checkpoint.py](experiments/checkpoint.py) | Full/snapshot bundles, integrity, atomic publication, references and retention |
| [experiments/metrics.py](experiments/metrics.py) | Streaming queue, minute/window metrics and diagnostic aggregation |
| [experiments/storage.py](experiments/storage.py), [experiments/core.py](experiments/core.py) | JSON/JSONL, RNG streams, hashing and queue-reward helpers |
| [experiments/runtime.py](experiments/runtime.py) | Runtime identity and publication environment lock |
| [experiments/tuning.py](experiments/tuning.py) | Generates 24 candidates; does not train or select winners |
| [experiments/release.py](experiments/release.py) | Inventories, release packages and verified archive copies |

### Scripts, tests and evidence

| File | Purpose |
|---|---|
| [scripts/verify.py](scripts/verify.py) | Six real-SUMO pilots covering parents, WCE, five continuations and fresh-process recovery |
| [scripts/verify_audit.py](scripts/verify_audit.py) | Repeat an existing verification parent with audit output and compare learning/queues |
| [scripts/verify_crash.py](scripts/verify_crash.py) | Kill a Grid-DQL pilot process group, reject duplicate ownership, verify recovery |
| [scripts/measure_metrics.py](scripts/measure_metrics.py) | Synthetic 6600-second statistics-buffer measurement, not total job RSS |
| [scripts/evaluate_suite.py](scripts/evaluate_suite.py) | Selected controller endpoint evaluation, caching and prescribed audits |
| [scripts/report.py](scripts/report.py) | Paired publication-result checks and training-seed CSV statistics |
| [scripts/export_history.py](scripts/export_history.py) | Export durable episodes excluding discarded recovery branches |
| [scripts/package_code.py](scripts/package_code.py) | Whitelisted code releases without Git objects, images or caches |
| [scripts/slurm/prepare_jobs.py](scripts/slurm/prepare_jobs.py) | Check stage prerequisites and prepare tasks without submitting |
| [scripts/slurm/train.sbatch](scripts/slurm/train.sbatch) | CPU array/resource template and signal forwarding |
| [scripts/slurm/launch.py](scripts/slurm/launch.py) | Verify artifacts/image/gate, extract to `$TMPDIR`, launch container |
| [tests/test_learning.py](tests/test_learning.py) | Replay, targets, exact next-update recovery, masking and inference graphs |
| [tests/test_protocol.py](tests/test_protocol.py) | Plateau, metrics, rewards, manifests and releases |
| [tests/test_storage.py](tests/test_storage.py) | Reference-safe retention and failed-write recovery protection |
| [provenance/extraction.json](provenance/extraction.json) | Source commit, original working-tree state and extraction hashes |
| [provenance/verification](provenance/verification) | Local gates, unit logs, resource and acceptance summaries |

## 4. Separate source, inputs, runs and archives

```text
workspace/
├── cbwce_standalone/   # Git: source, necessary static assets, config, docs
├── cbwce_data/         # Frozen manifests and shared generated traffic
├── cbwce_runs/         # Models, summaries and validation history
├── cbwce_cache/        # Reconstructible caches
└── cbwce_archive/      # Verified releases and retained results
```

External directories are created as needed; their appearance here does not imply existing data.

| Variable | Default | Contents |
|---|---|---|
| `CBWCE_ASSETS_ROOT` | Project `assets/` | Networks, detectors and original profiles |
| `CBWCE_DATA_ROOT` | Sibling `cbwce_data/` | Shared immutable traffic |
| `CBWCE_RUN_ROOT` | Sibling `cbwce_runs/` | Default run root |
| `CBWCE_CACHE_ROOT` | Sibling `cbwce_cache/` | Reconstructible evaluation caches |
| `TMPDIR` | System temporary area | SUMO/worker files; node-local job storage on Hábrók |

Explicit `--output` takes precedence; resume requires the original directory. Scientific identities use content/config/protocol, not old absolute paths. Release inputs must not depend on external symlinks. From the project root:

```bash
export CBWCE_PROJECT_ROOT="$PWD"
export CBWCE_ASSETS_ROOT="$CBWCE_PROJECT_ROOT/assets"
export CBWCE_DATA_ROOT="$(dirname "$CBWCE_PROJECT_ROOT")/cbwce_data"
export CBWCE_RUN_ROOT="$(dirname "$CBWCE_PROJECT_ROOT")/cbwce_runs"
export CBWCE_CACHE_ROOT="$(dirname "$CBWCE_PROJECT_ROOT")/cbwce_cache"
mkdir -p "$CBWCE_DATA_ROOT" "$CBWCE_RUN_ROOT" "$CBWCE_CACHE_ROOT"
```

## 5. Inputs, configuration and stages

Each network contains `network.net.xml`, `detectors.add.xml` and eleven demand CSVs: four cardinal directions, four diagonals, `Periphery_to_Center`, `Center_to_Periphery`, and `Uniform`. Demand loading records source hashes, original totals and normalization factors. Keep profile names/order consistent with mixture weights.

| Configuration | Applies to |
|---|---|
| `config/revised/config_{ma2c,dql,ppo}_large.ini` | Grid controllers |
| `config/revised/config_{ma2c,dql,ppo}_real.ini` | Monaco controllers |
| `config/revised/config_wce_large.ini`, `config_wce_real.ini` | Network-specific WCE |
| `config/protocol.json` | Durations, seeds, budgets, validation counts and stopping thresholds |

Default training lasts 6600 seconds/1320 joint control steps, with five-second control and 600-second demand intervals. MA2C/PPO rollouts contain 120 steps; PPO performs four epochs. DQL uses batch 64, warm-up 5000, updates every four steps and capacity 50,000 joint transitions. These quantities have different meanings and need not match numerically.

```text
parent training → best controller
  ├── baseline
  ├── random_group
  ├── domain_randomization
  └── offline wce against frozen parent → best WCE
        ├── fixed_wce: train controller, freeze WCE
        └── online_wce: train controller and WCE
```

All five branches share a best parent; fixed/online share a best WCE. Baseline uses the original profile sequence, random group samples a single profile, and DR samples Dirichlet mixtures. Publication branching requires plateau-confirmed ancestry with matching network, controller, training seed and WCE-parent identity.

## 6. Environment and basic checks

The target is Linux CPU: Python 3.10, TensorFlow CPU 2.15.1 through compat.v1, NumPy 1.26.4, and SUMO/TraCI/sumolib 1.26.0. See [direct requirements](packaging/requirements.in) and the [hash-locked dependency set](packaging/requirements.lock). The following installs a local development environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r packaging/requirements.lock
python -m pip check
sumo --version
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
export MPLBACKEND=Agg
python main.py --help
python main.py check --network grid
python main.py check --network monaco
python -m unittest discover -s tests -v
```

SUMO system libraries are listed in [Apptainer.def](packaging/Apptainer.def). Set `PYTHONPATH` only when using system SUMO tools instead of installed Python packages; never add the original project. TraCI needs local socket communication. `check` validates configuration/profile loading, not a real simulation. Local installation is not container acceptance.

## 7. Executable short workflow

This is a **v8.1 correctness pilot**, not v8.2 validation or publication training. Run in one Bash session from the project root; each output directory must be new.

```bash
CBWCE_DEMO_ROOT="$(mktemp -d /tmp/cbwce-demo.XXXXXX)"
python main.py pilot-manifest --network grid \
  --output "$CBWCE_DEMO_ROOT/validation.json"
pilot_args=(--network grid --controller dql --seed 9001 --pilot
  --validation "$CBWCE_DEMO_ROOT/validation.json"
  --max-episodes 1 --validate-every 1 --validation-repeats 1
  --pilot-block-seconds 10)
python main.py parent "${pilot_args[@]}" --output "$CBWCE_DEMO_ROOT/parent"
CBWCE_PARENT=$(python - "$CBWCE_DEMO_ROOT/parent" <<'PY'
import sys
from experiments.checkpoint import Checkpoints
print(Checkpoints(sys.argv[1]).role('best'))
PY
)
python main.py wce "${pilot_args[@]}" --parent "$CBWCE_PARENT" \
  --output "$CBWCE_DEMO_ROOT/wce"
CBWCE_WCE=$(python - "$CBWCE_DEMO_ROOT/wce" <<'PY'
import sys
from experiments.checkpoint import Checkpoints
print(Checkpoints(sys.argv[1]).role('best'))
PY
)
for method in baseline random_group domain_randomization fixed_wce online_wce; do
  wce_args=()
  if [[ "$method" == fixed_wce || "$method" == online_wce ]]; then
    wce_args=(--wce "$CBWCE_WCE")
  fi
  python main.py continue "${pilot_args[@]}" --method "$method" \
    --parent "$CBWCE_PARENT" "${wce_args[@]}" \
    --output "$CBWCE_DEMO_ROOT/$method"
done
```

Eleven ten-second blocks produce 110 seconds/22 decisions, below DQL's default 5000-step warm-up. This manual example checks orchestration. The verifier reduces replay/warm-up settings to exercise actual updates and recovery. `no_learning` or `max_budget_unconfirmed` can be normal short-budget outcomes.

The following run SUMO and create models, costing more time/space than unit tests:

```bash
python scripts/verify.py --output "$CBWCE_RUN_ROOT/verification-v81"
# A single case produces a partial gate, not full publication evidence.
python scripts/verify.py --case grid_dql \
  --output "$CBWCE_RUN_ROOT/verification-grid-dql"
python scripts/verify_audit.py \
  --verification "$CBWCE_RUN_ROOT/verification-v81" --case grid_dql
python scripts/verify_crash.py --verification "$CBWCE_RUN_ROOT/verification-v81"
python scripts/measure_metrics.py --output "$CBWCE_RUN_ROOT/metrics-memory.json"
```

Audit/crash scripts require the verifier's model directories, not just archived summaries.

## 8. Manifests, publication entry points and resume

### Current manifest schema

This unfrozen development example is accepted by v8.1; it is **not** the proposed DR validation design:

```json
{
  "version": "8.1", "network": "grid", "split": "validation", "frozen": false,
  "scenarios": [{
    "id": "development_example", "horizon": 3600, "generation_seed": 9001,
    "blocks": [{"start": 0, "duration": 3600, "profile": "Uniform"}]
  }]
}
```

Blocks support a profile or eleven normalized `weights`, plus `multiplier`/`sigma`. Current manifests must cover 3600 seconds contiguously, use unique IDs and a boolean `frozen`. Publication training requires frozen `split=validation`; final tests require `split=test`. Simply changing the Uniform fixture to `true` does not establish an appropriate publication design.

The v8.2 design freezes eleven mixtures/order and three realizations, retaining traffic/recurrent state across blocks. Confirm duration and synchronize implementation before generating its manifest. Offline WCE still evaluates its own demand policy, not replacement fixed-DR actions.

### Arguments and gates

| Argument | Meaning |
|---|---|
| `--network`, `--controller` | `grid`/`monaco`; `ma2c`/`dql`/`ppo` |
| `--seed` | Development default 9001; explicitly use 101/202/303 for publication |
| `--validation` | Required manifest |
| `--output` | New run directory; otherwise a unique directory under the run root |
| `--parent`, `--wce` | Full best checkpoint directories, not individual `.index` files |
| `--method` | Continuation method only |
| `--gate` | Required publication verification gate |
| `--resume` | Restore the original directory and scientific identity |
| `--segment-seconds` | Default 82,800 seconds/23 hours, leaving margin under Slurm's 24-hour template |
| `--audit` | Detailed output; normally disabled for routine training |
| `--pilot` | Development overrides; excludes publication training seeds |

Publication rejects `--config`, `--max-episodes`, `--validate-every`, `--validation-repeats` and `--pilot-block-seconds`. It also checks runtime locks and source fingerprints. Successful process exit does not mean plateau completion.

```bash
python main.py parent --help
python main.py wce --help
python main.py continue --help
python main.py check --network grid --publication \
  --validation "$CBWCE_DATA_ROOT/manifests/grid-validation.json"
```

After implementing the intended protocol and supplying manifests/image/gate, the publication command structure is:

```bash
python main.py parent --network grid --controller ma2c --seed 101 \
  --validation "$CBWCE_DATA_ROOT/manifests/grid-validation.json" \
  --gate "$CBWCE_DATA_ROOT/verification/gate.json" \
  --output "$CBWCE_RUN_ROOT/grid/ma2c/seed_101/parent"
```

Resume by repeating the original complete command with `--resume`, preserving seed, manifest, configuration and ancestry. Budgets, RNG, patience, optimizer and replay resume; DQL continuation epsilon resets only on the initial branch. Resume an `interrupted` stage rather than launching its successor.

```bash
python main.py tuning-configs --output "$CBWCE_DATA_ROOT/tuning-candidates"
```

This generates 24 INIs and `candidates.json` only. Selected settings must be versioned in `config/revised/` and reverified because publication does not accept `--config` overrides.

## 9. Run outputs and checkpoint roles

```text
one_run/
├── manifest.json          # Protocol, source, inputs, ancestry, identity
├── result.json            # Application outcome and resource summary
├── references.json        # best/latest/final/common/window/pins
├── attempts.jsonl         # Restarts and rollback boundaries
├── episodes.jsonl.gz      # Complete episodes with attempt identity
├── blocks.jsonl.gz        # Weights, traffic hashes, diagnostics
├── validation.jsonl.gz    # Routine and confirmation results
├── checkpoints/e.../
│   ├── manifest.json      # Kind, signatures, parents, hashes
│   ├── runner.pkl         # Runner/selection state
│   ├── controller/        # Parameters; full state adds state.pkl and DQL replay/
│   └── wce/               # Present when WCE must be persisted
├── audit/                 # Optional NPZ traces
└── scheduler.json         # Slurm launcher receipt
```

Not all files/models appear in every stage. Frozen parents/fixed WCEs are shared references.

| Role/status | Meaning |
|---|---|
| `best` | Routine-validation best full state; branch from this |
| `best_inference` | Corresponding inference snapshot; cannot resume training |
| `latest` | Latest valid full recovery point, used by `--resume` |
| `final` | Terminal training state, not necessarily best |
| `best_at_common_budget` | Best inference endpoint frozen at 1000 continuation episodes |
| `plateau_confirmed` | Confirmed empirical plateau, not global optimality |
| `max_budget_unconfirmed` | Maximum budget reached without confirmation |
| `no_learning` | Parent/offline WCE reaches its cap without the required meaningful improvement |
| `interrupted` | Unfinished stage requiring recovery |
| `failed` | Error requiring inspection |

Recovery points are saved every ten complete episodes; retain the latest two, full current best, five lightweight validation candidates, common-budget/final states and downstream pins. Older recovery dependencies may retain additional objects. Never delete solely by modification time.

Full states include optimizer, RNG, replay/pending data, counters, selection and simultaneous online WCE. Snapshots are inference-only. Only load trusted pickle states; do not edit checkpoint data, hashes or reference indexes manually.

## 10. Evaluation and exports

These commands define the 23 test scenarios without generating traffic or running SUMO:

```bash
python main.py test-manifest --network grid \
  --output "$CBWCE_DATA_ROOT/manifests/grid-test.json"
python main.py test-manifest --network monaco \
  --output "$CBWCE_DATA_ROOT/manifests/monaco-test.json"
```

Evaluate a terminal controller run, not an offline-WCE run. Publication needs a publication model, frozen test manifest, matching runtime and ten realizations/scenario. Development models need `--pilot` and cannot enter publication reports.

```bash
python scripts/evaluate_suite.py \
  --run "$CBWCE_RUN_ROOT/grid/ma2c/seed_101/online_wce" \
  --manifest "$CBWCE_DATA_ROOT/manifests/grid-test.json" \
  --selection best --output "$CBWCE_RUN_ROOT/evaluation/grid-ma2c-101-best"
# Use best_at_common_budget and a new output directory when that role exists.
python scripts/export_history.py \
  --run "$CBWCE_RUN_ROOT/grid/ma2c/seed_101/online_wce" \
  --output "$CBWCE_RUN_ROOT/exports/grid-ma2c-101-episodes.csv"
# All three evaluation directories must already be complete.
python scripts/report.py \
  --input "$CBWCE_RUN_ROOT/evaluation/grid-ma2c-101-best" \
  --input "$CBWCE_RUN_ROOT/evaluation/grid-ma2c-202-best" \
  --input "$CBWCE_RUN_ROOT/evaluation/grid-ma2c-303-best" \
  --output "$CBWCE_RUN_ROOT/reports/grid-ma2c-online-best"
```

Evaluation produces manifests/results, `rollouts.jsonl.gz` and prescribed audits. Cache identity includes checkpoint, test inputs, source and runtime. The first Uniform/peak_1.5 realization is audited; routine outputs include scalars, minute curves and lane windows.

Reports validate 23×10 rollouts, hashes, traffic/SUMO pairing and three training seeds per cell. Outputs are `seeds.csv`, `comparison.csv` and, when comparators exist, `paired_differences.csv`. Standard deviations are across training seeds. Use canonical exported history rather than counting duplicate pre/post-recovery journals.

## 11. Bounded memory and storage

Per-second measurements continue, but routine statistics are accumulated online. Retain a current lane vector and a 600-second network-total ring buffer; aggregate diagnostics by demand block. Generate training traffic blockwise and retain weights/hashes; share compressed content-addressed evaluation traffic and route tables. DQL replay is fixed-capacity and saved in chunks. Validation constructs inference graphs without optimizer/replay. Enable detailed audit deliberately and derive CSVs/figures offline.

The synthetic buffer benchmark is not total memory. Historical trainer `ru_maxrss` also excludes aggregate simultaneous SUMO/validator memory. Measure total job peak in the target image with filled replay and apply at least a 1.3 margin.

## 12. Optional cluster workflow (not the current execution target)

### Build and verify

No built SIF or generated `packaging/runtime.lock.json` is currently included. On an Apptainer-capable development machine, replace the placeholder with a real Python 3.10 base-image digest:

```bash
export CBWCE_BASE_IMAGE='python:3.10-slim-bookworm@sha256:REPLACE_WITH_REAL_DIGEST'
bash packaging/build.sh
```

The placeholder cannot build an image. The script consumes the committed dependency lock and produces `cbwce-v8.1.sif`, plus `runtime.actual.json` and `runtime.lock.json` under packaging/. Set `CBWCE_UPDATE_LOCK=1` and provide pip-tools only when intentionally refreshing dependencies. Image names/protocol identifiers must be updated when implementing v8.2.

```bash
apptainer exec --cleanenv \
  --bind "$PWD,$CBWCE_DATA_ROOT,$CBWCE_RUN_ROOT,$CBWCE_CACHE_ROOT" \
  cbwce-v8.1.sif python scripts/verify.py \
  --output "$CBWCE_RUN_ROOT/container-verification-v81"
```

The verifier stores shared inputs inside its output directory. Explicitly forward custom environment variables under `--cleanenv`; do not assume host `CBWCE_*` inheritance. Cluster jobs need the full six-case gate from the locked image, not historical host evidence.

### Storage and resources

Keep source/history in local Git; active releases, images, inputs and durable runs under `/scratch/<account>/cbwce/`; extracted source/SUMO temporaries under job `$TMPDIR`; verified archives under `/projects/<account>/cbwce/`; important long-term results in RDMS or an independent backup. Scratch has no backup and projects is not directly available to compute nodes. Archive from a login node and inspect actual quotas with `hbquota`. Official references: [storage](https://wiki.hpc.rug.nl/habrok/migration/migration_habrok), [quota](https://wiki.hpc.rug.nl/habrok/data_management/quota), [Apptainer](https://wiki.hpc.rug.nl/habrok/examples/apptainer), [Slurm](https://wiki.hpc.rug.nl/habrok/job_management/scheduling_system).

The template starts with regular, one task, two CPUs, 8 GiB, 24 hours, concurrency four and a signal 600 seconds before termination. The memory setting is a pilot request, not a certified publication limit.

### Prepare and submit

First prepare the intended protocol, SIF, matching gate, frozen manifests and code release. Replace ACCOUNT, versions and paths. Keep gate/validation files inside the bound data root.

```bash
CBWCE_HPC_ROOT=/scratch/ACCOUNT/cbwce
python scripts/slurm/prepare_jobs.py --stage parent \
  --image "$CBWCE_HPC_ROOT/cbwce-v8.1.sif" \
  --code-release "$CBWCE_HPC_ROOT/releases/code-release.tar.gz" \
  --data-root "$CBWCE_HPC_ROOT/data" --run-root "$CBWCE_HPC_ROOT/runs" \
  --cache-root "$CBWCE_HPC_ROOT/cache" \
  --validation-grid "$CBWCE_HPC_ROOT/data/manifests/grid-validation.json" \
  --validation-monaco "$CBWCE_HPC_ROOT/data/manifests/monaco-validation.json" \
  --gate "$CBWCE_HPC_ROOT/data/verification/gate.json" \
  --output "$CBWCE_HPC_ROOT/parents.json"
# Only use this range when the generator reports 18 tasks.
sbatch --array=0-17%4 scripts/slurm/train.sbatch "$CBWCE_HPC_ROOT/parents.json"
```

Submit from the project root because the template uses a relative launcher path. Upload the code archive and its `.tar.gz.manifest.json`. Matrix preparation needs project dependencies and, for downstream checks, access to `sacct`.

Generation does not submit or automatically schedule successors. Full counts are 18 parents, 18 WCEs and 90 continuations; completed items can reduce the actual matrix. After parents confirm plateaus and Slurm reports `COMPLETED/0:0`, generate a new `--stage wce` JSON; after WCEs pass, generate `--stage continue`. Interrupted jobs acquire `--resume`; failed/active directories require inspection. The launcher verifies artifacts and forwards signals; the trainer periodically writes durable checkpoints directly to the run root, not just at job exit.

## 13. Releases, archives and independent preservation

Code packaging uses [release_allowlist.json](packaging/release_allowlist.json). Outputs must be new and outside the packaged source directory.

```bash
CBWCE_RELEASE_ROOT="$(dirname "$PWD")/cbwce_archive/releases"
mkdir -p "$CBWCE_RELEASE_ROOT"
python scripts/package_code.py \
  --output "$CBWCE_RELEASE_ROOT/cbwce-code-docs-update.tar.gz"
python main.py package --kind inputs --source assets \
  --output "$CBWCE_RELEASE_ROOT/cbwce-assets-new.tar.gz"
python main.py package --kind run \
  --source "$CBWCE_RUN_ROOT/grid/ma2c/seed_101/online_wce" \
  --output "$CBWCE_RELEASE_ROOT/grid-ma2c-101-online-run.tar.gz"
```

An assets-only package omits manifests/generated traffic; stage these explicitly when needed. Each archive has per-file and whole-archive hashes. Code packages exclude Git history, SIF and caches: back up Git separately, retain the SIF separately with its hash. Run packages capture their source directory; externally referenced ancestors also need preservation.

Archive a directory after it stops changing, from the login side:

```bash
python main.py archive \
  --source /scratch/ACCOUNT/cbwce/runs/completed-run \
  --destination /projects/ACCOUNT/cbwce/completed-run
```

This copies to destination-local temporary storage, verifies, publishes and writes `archive_inventory.json` without deleting the source. Only remove inactive scratch copies after archive/backup verification; protect referenced parent/WCE objects and original-project history.

Existing sibling `cbwce_archive/releases/*v8.1*` packages are snapshots of initial commit `a367a66`, **excluding subsequent v8.2 plan changes and these READMEs**. Repackage under a new filename for current documentation; do not overwrite historical hash records.

## 14. Troubleshooting and handover

| Symptom | Check |
|---|---|
| Rejected 6600-second manifest | Implement the protocol migration; current validation is v8.1/3600 seconds |
| Unfrozen-manifest error | Confirm approved design, split and frozen status; never substitute the pilot fixture |
| Missing/stale gate | Reverify scientific source/config/input changes in the matching image |
| Missing runtime lock | Complete an actual image build and runtime recording |
| Signature/identity mismatch | Check config, inputs, protocol, seed and ancestry; do not mix incompatible states |
| Snapshot cannot resume | Branch from full best; resume from full latest |
| Another process owns run | Inspect the existing process/job instead of concurrent writes |
| Output already exists | Use a new directory or correctly resume the original |
| Short-pilot no_learning | Inspect budget/warm-up; it need not indicate a crash |
| Downstream preparation fails | Check plateau, scheduler receipt and successful accounting |
| Missing traces/plots | Compact output is default; export offline or enable deliberate audit |
| Missing relocated inputs | Check path variables, binds and manifests rather than reinstating old absolute paths |

Algorithm changes require relevant configuration/tests. Validation-design changes require synchronized bilingual plans, protocol, manifests, cache identities, gates and documentation. Scientific-identity changes cannot silently resume an old run; Markdown changes alone do not alter runtime behavior.

For handover retain independent Git, code releases, input inventories/generator versions, image/software locks, frozen manifests, selected models and ancestry, canonical results and resource records. Version source, research plans, acceptance evidence and actual training results separately.
