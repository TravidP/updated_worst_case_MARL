# Compatibility commands and historical verification

The corrected implementation now lives in `agents/`, `envs/`, and `experiments/`. This package provides compatibility wrappers, so existing `python -m revision.runner` commands use the integrated implementation. Start new work with [the project README](../README.md) and `python main.py experiment ...`. The working protocol is [the English guide](../reviewer_revision_plan.md), with a [Chinese translation](../reviewer_revision_plan_zh.md).

Publication-scale experiments have not been launched. A passing verification gate is required before the runner permits publication training. Pilot results are correctness evidence, not performance results for the paper.

The [historical implementation report](IMPLEMENTATION.md) and [old gate](verification/corrections_20260914_gate.json) describe the pre-integration source. That gate does not certify the moved code. New verification records are written under `runs_eval/revised/verification/`. The following old command forms remain supported; prefer the new README's output locations.

## Runtime and verification

Run from the repository root:

```bash
conda activate deeprlsc
export PYTHONDONTWRITEBYTECODE=1
export PYTHONWARNINGS=ignore
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
export MPLCONFIGDIR=/tmp/revision-matplotlib
python -m revision.tests
python -m revision.verify --output revision/verification/my_check --workers 4
```

Use a new output directory for every verification attempt. SUMO needs permission to open local TraCI sockets. TensorFlow sessions use one CPU thread for inter-op and intra-op work; the environment variables above also constrain BLAS/OpenMP threads. The verifier records the number of concurrent pilot workers.

The ten-test deterministic suite checks queue arithmetic, separate random streams, IQL cadence and frozen mode, partial batches, checkpoint recovery in fresh processes, checkpoint retention, invalid checkpoints, incomplete rollout handling, exclusive manifests, batched inference, and recurrent-network equivalence. Floating-point recovery comparisons use `rtol=1e-6`, `atol=1e-7`; discrete state is checked exactly.

The integration matrix covers both networks and all four controller families. For each combination it runs:

1. A 160-step parent, including partial-batch handling where needed.
2. Two offline WCE episodes against that frozen parent.
3. Two complete continuation episodes for each of the five methods.
4. A resumed online-WCE episode, compared with the uninterrupted run.
5. Eight complete 3,600-second evaluations: all five methods, an exact repeat, a different SUMO seed, and a different policy seed.
6. One intentionally interrupted evaluation that must remain a failed attempt.

This produces eight pilot cases, 40 continuations, 64 complete evaluations, and eight deliberate evaluation failures. Pilot seed `9001` is outside the publication seed set. A case checks actual learner reward logs against saved one-second lane measurements. Frozen controller/WCE tensors must remain identical; online WCE tensors must change. Exact repeats and resumed episodes must reproduce their recorded results. `gate.json` is written only after every check passes and the source hashes still match.

## Training stages

The stage runner has one controller loop. Its five method IDs are `baseline`, `random_group`, `domain_randomization`, `fixed_wce`, and `online_wce`.

The following publication examples require a passing gate. Repeat them for the prescribed seeds `101, 202, 303, 404, 505`, networks `grid` and `monaco`, and controllers `ia2c`, `ma2c`, `iqll`, and `ppo`.

```bash
python -m revision.runner --stage parent --network grid --controller ia2c \
  --seed 101 --gate revision/verification/my_check/gate.json \
  --output revision/runs/grid/ia2c/101/parent

python -m revision.runner --stage wce --network grid --controller ia2c \
  --seed 101 --gate revision/verification/my_check/gate.json \
  --parent revision/runs/grid/ia2c/101/parent/checkpoint_001000000 \
  --output revision/runs/grid/ia2c/101/offline_wce

python -m revision.runner --stage continue --network grid --controller ia2c \
  --seed 101 --method online_wce --gate revision/verification/my_check/gate.json \
  --parent revision/runs/grid/ia2c/101/parent/checkpoint_001000000 \
  --wce revision/runs/grid/ia2c/101/offline_wce/checkpoint_000660000 \
  --output revision/runs/grid/ia2c/101/online_wce
```

The parent receives exactly 1,000,000 controller-learning steps. Offline WCE receives 500 episodes, or 660,000 frozen-controller simulation steps. Each continuation receives 1,320,000 additional learning steps and finishes at 2,320,000 cumulative learning steps. Fixed and online WCE require the same pretrained WCE with a matching parent identity. The other three methods omit `--wce`.

For interactive short checks, pass `--pilot --visualization --seed 9001`. Automated `verify` checks stay headless; use `--no-visualization` for an explicitly headless manual pilot. Parent pilots accept `--steps 160`; continuation pilots use a multiple of 1,320, such as `--steps 2640`; offline WCE pilots accept `--episodes 2`. Do not use publication seeds in pilots.

Checkpoint directory suffixes count **simulation steps within the current stage**. They are not cumulative controller-learning counters. Read actual learning counters and stage identity from the checkpoint state and run result.

## Resume and random state

Use `--resume` with an exact checkpoint from the same stage and method, keep the original parent/WCE arguments and seed, and choose a new output directory. The total stage budget remains the original budget. The checkpoint restores all TensorFlow variables, including optimizer slots, plus replay/rollout data, schedules, counters, recurrent state, and the independent random streams.

Normal checkpoints are saved every ten complete episodes by default; `--checkpoint-every 1` saves every episode. The final checkpoint is always saved. A partial parent cutoff is a declared stage boundary: its real samples are updated with masked padding and correct bootstrapping, then continuation begins a fresh episode. Padding never creates extra environment transitions or evaluation observations.

Automatic TensorFlow checkpoint deletion is disabled so older immutable bundles retain their referenced files. The checkpoint interval controls how many bundles accumulate.

Only load trusted revision-generated checkpoints: their Python state uses pickle. Hashes detect accidental corruption but do not make untrusted serialized content safe. Legacy weight-only checkpoints are rejected as continuation states. Missing, damaged, and incompatible checkpoints fail explicitly.

Demand generation, controller sampling, WCE sampling, replay sampling, demand selection, initialization, and SUMO seed generation have independent streams. The WCE samples its Gaussian noise outside TensorFlow; the legacy stateful sampling operation is not executed. Legacy NumPy initialization is scoped and restored during graph construction.

## Materialized evaluation demand

Generate the traffic once, then reuse the resulting artifact across controllers and methods:

```bash
python -m revision.runner --stage demand --network grid --controller ia2c \
  --arrival-seed 51001 --output revision/evaluation/grid_uniform_51001

python -m revision.runner --stage evaluate --network grid --controller ia2c \
  --seed 101 --sumo-seed 61001 --policy-seed 71001 \
  --parent revision/runs/grid/ia2c/101/online_wce/checkpoint_001320000 \
  --artifact revision/evaluation/grid_uniform_51001/demand.json \
  --output revision/evaluation/grid_ia2c_101_online_uniform_51001
```

The default demand artifact uses Uniform demand over 3,600 seconds. `--schedule` accepts a JSON list of consecutive blocks with `start`, `duration`, and eleven normalized `weights`. Blocks must cover 3,600 seconds without gaps or overlaps. This supports seen profiles and mixture/switching checks. The complete twelve-scenario manuscript generator and final statistical/heatmap reporting campaign are specified in the working guide; this correction package does not automatically launch that campaign.

Artifacts contain vehicle identifiers, departure times, OD pairs, route edge sequences, speed factors, network identity, and an integrity hash. Routes are checked on an empty network before controller interaction. Route or insertion errors are never silently dropped. Evaluation sets `train_mode=False`, records the SUMO startup command and effective seed, and disables learning. Use `--pilot` when evaluating short pilot checkpoints.

The evaluator requires exactly 3,600 consecutive one-second observations. Short, empty, or duplicated timestamp sequences cannot enter valid summaries. Congested full-horizon rollouts remain valid. For a deliberate pilot fault check, use `--pilot --fail-after-steps 40`.

## Records and implementation map

| Component | Responsibility |
|---|---|
| `core.py` | Queue definitions, separate RNG streams, strict rollout validation, immutable run records |
| `environment.py` | Isolated SUMO runtime files, seeded resets, one-second lane measurements, state encoding |
| `learning.py`, `recurrent.py` | Shared controller learning, fingerprints, IQL cadence, masked batches, WCE inference/updates |
| `checkpoint.py` | Complete versioned checkpoints, integrity and compatibility checks |
| `runner.py` | Common stage execution, evaluation, demand artifacts, computation accounting |
| `tests.py`, `verify.py` | Deterministic checks, integration cases, publication gate |

Each run stores an immutable input manifest, environment identity, exact checkpoint parents, effective SUMO startup records, lane time series, control rewards, demand decisions, timing, and a complete/failed result. Training manifests embed the explicit eleven normalized profiles and their original CSV hashes. Original CSVs are unchanged. Failed attempts keep their partial measurements and error status in their own directory.

`episode_*.npz` and `rollout.npz` contain the common ordered lane set, timestamps, and uncapped lane queues. The matching JSONL files record traffic counts and vehicle-time speed totals. `*.controls.jsonl` records each controller reward vector. SUMO trip XML files are retained under the run's `runtime/` directory. Checkpoint and simulation output directories are ignored by Git because they can be large; verification evidence remains available on disk.

Stage wall time includes initialization and output work. Named timing components cover the controller loop and checkpoint operations; unclassified startup/output overhead is the remainder. Standalone fixed/online WCE cost must include parent training plus offline WCE training plus continuation, even when the experiment shares a pretrained WCE.

## Optional live SUMO window

The compatibility runner accepts `--visualization` (on) and `--no-visualization` (off, default), as does `python main.py experiment <stage>`. See [the visualization guide](../docs/VISUALIZATION.md). The GUI requires a working desktop display and opens on the training machine. Existing commands stay headless. This source change requires a new publication verification gate; historical gates remain preserved.
