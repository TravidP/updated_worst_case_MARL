# Stage-by-stage training launchers
PASS all ten corrections: /home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/verification/four_workers_20260917T140900/gate.json
## Start now: tune the eight revised configurations first

Protocol 6 moves reward scaling to the learner boundary and makes all learner
parameters strict INI inputs. The old protocol-5 gate and checkpoints are
intentionally incompatible. Do not start publication parents until all eight
configuration campaigns finish and a new verification gate passes.

```bash
conda activate deeprlsc
cd /home/sdc_joran/Journal/deeprl_signal_control

python scripts/training/16_tune_all_revised.py \
  --output output_result/revised/tuning_matrix_YYYYMMDD
```

Use `--dry-run` to preview all commands or `--through offline` to stop after
fixed-batch screening. The matrix runs one SUMO/TensorFlow process at a time,
keeps independent artifacts for every combination, and promotes only a 66k
winner that passes all gates. See [the tuning guide](../../docs/REVISED_TUNING.md).

After all eligible configurations are promoted, run `python -m revision.verify`
to create a new gate. Only then use scripts `08`–`15` for publication training.

View [TensorBoard](http://127.0.0.1:6007/): start with `train/reward_by_learning_step` (negative rewards closer to zero mean lower cost) and `monitor/mean_total_queue` (lower is better). Sections 0–3 below describe optional pilots; Sections 4–6 cover full training and subsequent stages.

## Learner-boundary reward scaling and TensorBoard (protocol 6)

Controller rewards use negative queue at the five-second endpoint, without division by 100; MA2C retains 0.9 neighborhood weighting. Monaco IA2C/MA2C batches are now 120. Training episodes remain 6,600 seconds. `train/reward_by_learning_step` records every actual learner reward; `train/episode/mean_total_queue` summarizes a full episode. Before learning, every 50 complete episodes and at final budget, three paired 600-second Uniform tests save per-second CSV, NPZ and TensorBoard curves.

See [settings, commands, metric definitions and output locations](../../docs/TRAINING_MONITORING.md). Defaults are `--monitor-every 50 --monitor-rollouts 3`; pilots may shorten the interval. Monitoring records fixed rollouts, TensorBoard metrics and checkpoints but no longer terminates training for degradation relative to the initial monitor; final model quality is decided by the complete paired evaluation. Start fresh parents using a gate matching the current source; historical checkpoints remain preserved. Exact originals and checksums are archived under `docs/history/endpoint_monitoring_20260916T095506Z/`.

### Open TensorBoard while training continues

Open a **second terminal** on the training computer; leave the training terminal running. To view all parent and baseline runs under `runs/revised/`:

```bash
conda activate deeprlsc
cd /home/sdc_joran/Journal/deeprl_signal_control

tensorboard --logdir "$PWD/runs/revised" \
  --host 127.0.0.1 --port 6007 --reload_interval 15
```

Keep this terminal open and visit **[http://127.0.0.1:6007/](http://127.0.0.1:6007/)**. If TensorBoard is already running there, simply open that address; do not launch another copy on the same port. If the port is occupied by another service, change it to `6008` and open that port instead. Ctrl+C in the TensorBoard terminal stops only that viewer, not training in the other terminal.

To view **only the main training curves** of the run discussed here, use a separate viewer on port 6008:

```bash
CBWCE_RUN="$PWD/runs/revised/grid/ia2c/seed_101/parent/publication_20260916T112410_bca267b5"
tensorboard --logdir "$CBWCE_RUN/tensorboard" \
  --host 127.0.0.1 --port 6008 --reload_interval 15
```

Open [http://127.0.0.1:6008/](http://127.0.0.1:6008/). Replace `CBWCE_RUN` with the exact run directory printed by your launcher when viewing another run. To include that run's per-round detail curves too, use `--logdir "$CBWCE_RUN"` instead. The viewer recursively discovers event files; it does not start or resume training.

| Run / tag | What to read |
|---|---|
| `<run>/tensorboard` | Ongoing training and monitoring summaries; select this run for current learning progress. |
| `<run>/monitoring/round_000000/tensorboard` | Initial 600-second monitoring curves before learning, with simulation seconds on the x-axis. This saved reference does not keep updating. |
| `train/reward_by_learning_step` | Actual learner reward vs cumulative learning steps; less negative means lower queue cost. MA2C displays the agent-mean neighborhood reward. |
| `train/episode/mean_total_queue` | Mean queue over one complete 6,600-second training episode; lower is better. Final partial episodes have separate `train/partial_episode/*` tags. |
| `monitor/mean_total_queue` | Mean queue across the three fixed Uniform tests vs learning steps; lower is better. |
| `monitor/mean_current_wait_seconds` | Mean current waiting time in those monitoring tests; lower is better. |

In **SCALARS**, select the main run and use **STEP** as the horizontal axis. Use the refresh control if the browser appears stale. Event writes are buffered (about 30 seconds), and this command reloads files every 15 seconds, so updates are not instantaneous. Episode metrics appear only after an episode completes; fixed monitoring runs before learning, every 50 complete episodes, and at the final budget. During initial monitoring there are no training-reward points yet. Smoothing only changes the display; it does not change the recorded rewards.

`runs/revised/` covers parents and baseline continuations. For comparison continuations use `output_coevolution/revised/` (grid) or `output_coevolution_real/revised/` (Monaco); for offline WCE use `output_adversary/revised/` or `output_adversary_monaco/revised/`. Point `--logdir` at the relevant root or exact run's `tensorboard/` directory. WCE plots use macro steps, not controller-learning reward points.


[简体中文](README_zh.md) · [Full protocol](../../reviewer_revision_plan.md)

Scripts `01`–`07` launch **one stage for one network/controller/seed** through the existing `python main.py experiment` interface. Matrix scripts `08`–`15` expand this with up to four concurrent workers; see Sections 5–6. Each script prints a bilingual phase banner, method, real budget, selected checkpoints, output location and exact command. Python output is unbuffered so the runner's episode progress appears as it is emitted.

## 0. Prepare the environment and choose the experiment

Run from the repository root:

```bash
conda activate deeprlsc
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

export CBWCE_NETWORK=grid
export CBWCE_CONTROLLER=ia2c
export CBWCE_MODE=pilot
export CBWCE_SEED=9001
unset CBWCE_PARENT CBWCE_WCE CBWCE_RESUME CBWCE_GATE
unset CBWCE_STEPS CBWCE_EPISODES CBWCE_VISUALIZATION

python main.py experiment prepare
python main.py experiment check
```

Networks: `grid`, `monaco`. Controllers: `ia2c`, `ma2c`, `iqll`, `ppo`. Pilot seeds such as `9001` must be separate from publication seeds `101`.

Pilot scripts enable SUMO visualization by default; publication scripts disable it. A pilot needs `sumo-gui` and an accessible desktop display. Use `--no-visualization` when running without a display. The raw Python CLI and automated verification retain their existing headless default.

## 1. Train the common parent

Preview first; this does not start SUMO or create output files:

```bash
bash scripts/training/01_parent.sh --dry-run
```

Start the selected parent run:

```bash
bash scripts/training/01_parent.sh
```

With the pilot defaults, this is equivalent to:

```bash
python -u main.py experiment parent --network grid --controller ia2c \
  --seed 9001 --pilot --visualization --steps 160 --checkpoint-every 1 \
  --monitor-every 50 --monitor-rollouts 3
```

The script additionally supplies the unique output directory printed in its banner. Read the completed run's `result.json`: require `status=complete` and copy its exact `checkpoint` value. Set it for subsequent stages:

```bash
export CBWCE_PARENT='/replace/with/completed/parent/checkpoint_000000160'
```

The replacement must identify your actual run, not a guessed/latest filename. Keep the originating run's `manifest.json` alongside its checkpoint directories. For publication, the parent suffix is `checkpoint_001000000`.

## 2. Train WCE against that frozen parent

```bash
bash scripts/training/02_wce.sh --dry-run
bash scripts/training/02_wce.sh
```

The controller stays frozen. Pilot defaults are two WCE episodes / 2,640 frozen-controller simulation steps. Publication uses 500 episodes / 660,000 frozen-controller simulation steps. These do not add controller-learning steps.

After successful completion, copy the WCE run's `result.json.checkpoint` into:

```bash
export CBWCE_WCE='/replace/with/completed/wce/checkpoint_000002640'
```

For publication, the suffix is `checkpoint_000660000`. The WCE must have been trained against the exact parent in `CBWCE_PARENT`. Keep both variables unchanged throughout the five comparisons.

## 3. Retrain the five continuation methods

Choose and run these one at a time. Each command reloads the **same common parent**, not the preceding continuation's final controller.

```bash
# III-1: original sequential demand baseline
bash scripts/training/03_baseline.sh

# III-2: one uniformly selected demand group per block
bash scripts/training/04_random_group.sh

# III-3: a Dirichlet demand mixture per block
bash scripts/training/05_domain_randomization.sh

# III-4: pretrained WCE, with model parameters frozen
bash scripts/training/06_fixed_wce.sh

# III-5: the same pretrained WCE, updated after each full episode
bash scripts/training/07_online_wce.sh
```

Append `--dry-run` to any command to inspect it first. The first three methods use only `CBWCE_PARENT`; the last two also use the same `CBWCE_WCE`. Having `CBWCE_WCE` exported does not load it into the first three methods. No continuation changes the parent checkpoint or your shell variables.

| Phase / script | Pilot default | Publication budget | Output root |
|---|---:|---:|---|
| I / [01_parent.sh](01_parent.sh) | 160 learning steps | 1,000,000 learning steps | `runs/revised/` |
| II / [02_wce.sh](02_wce.sh) | 2 WCE episodes | 500 WCE episodes | Grid: `output_adversary/revised/`; Monaco: `output_adversary_monaco/revised/` |
| III-1 / [03_baseline.sh](03_baseline.sh) | +2,640 learning steps | +1,320,000 learning steps | `runs/revised/` |
| III-2 / [04_random_group.sh](04_random_group.sh) | +2,640 learning steps | +1,320,000 learning steps | Grid: `output_coevolution/revised/`; Monaco: `output_coevolution_real/revised/` |
| III-3 / [05_domain_randomization.sh](05_domain_randomization.sh) | +2,640 learning steps | +1,320,000 learning steps | Same continuation roots |
| III-4 / [06_fixed_wce.sh](06_fixed_wce.sh) | +2,640 learning steps | +1,320,000 learning steps | Same continuation roots |
| III-5 / [07_online_wce.sh](07_online_wce.sh) | +2,640 learning steps | +1,320,000 learning steps | Same continuation roots |

With a default pilot parent, all five final controllers have **2,800 learning steps**. Publication controllers have **2,320,000**. WCE computation is counted separately. Output directories retain `<root>/<network>/<controller>/seed_<seed>/<stage-or-method>/<run_id>/`.

## 4. Switch to publication mode

The current protocol already passed all eight verification cases. Use the actual gate path at the top of this guide and the read-only `check --gate` command. Older gates do not certify this protocol. Only rerun full verification if the gate becomes stale or the implementation/configuration changes:

```bash
CBWCE_VERIFY_DIR="$PWD/runs_eval/revised/verification/manual_$(date +%Y%m%d_%H%M%S)"
python main.py experiment verify --workers 4 --output "$CBWCE_VERIFY_DIR" \
  && export CBWCE_GATE="$CBWCE_VERIFY_DIR/gate.json"
```

That command launches headless tests and pilots; it is not read-only. After a fresh verification, keep its new gate instead of switching back to an earlier path. The example below uses the currently passing gate for **one parent**, as an alternative to the eight-parent launcher. Do not run both simultaneously. Clear pilot checkpoint selections before creating a fresh publication parent:

```bash
export CBWCE_MODE=publication
export CBWCE_SEED=101
export CBWCE_GATE="$PWD/runs_eval/revised/verification/endpoint_monitoring_20260916_v1/gate.json"
unset CBWCE_PARENT CBWCE_WCE CBWCE_RESUME
unset CBWCE_STEPS CBWCE_EPISODES CBWCE_VISUALIZATION

bash scripts/training/01_parent.sh --dry-run
bash scripts/training/01_parent.sh
```

Set the publication checkpoints as each stage completes:

```bash
export CBWCE_PARENT='/replace/with/publication/parent/checkpoint_001000000'
bash scripts/training/02_wce.sh

export CBWCE_WCE='/replace/with/publication/wce/checkpoint_000660000'
bash scripts/training/03_baseline.sh
bash scripts/training/04_random_group.sh
bash scripts/training/05_domain_randomization.sh
bash scripts/training/06_fixed_wce.sh
bash scripts/training/07_online_wce.sh
```

These are sequential user-run commands, not an automatically launched campaign. Repeat the procedure for both networks, four controllers and the single publication seed `101` to obtain 8 parents, 8 offline WCE runs and 40 continuations. Publication scripts read budgets from `config/revised/protocol.json` and omit pilot-only budget overrides. A pilot checkpoint cannot replace a publication parent.

## 5. Train all networks and controllers with up to eight workers

[08_all_parents.sh](08_all_parents.sh) runs **initial parent training**. With the default `--workers 4`, Grid's four jobs still finish before Monaco's four jobs start; `--workers 8` places both networks in one batch. Each job owns its model, SUMO process, output directory, and random streams. The accepted range is `--workers 1..8`.

| Order with one seed | Network | Controller |
|---|---|---|
| 1 | `grid` | `ia2c` |
| 2 | `grid` | `ma2c` |
| 3 | `grid` | `iqll` |
| 4 | `grid` | `ppo` |
| 5 | `monaco` | `ia2c` |
| 6 | `monaco` | `ma2c` |
| 7 | `monaco` | `iqll` |
| 8 | `monaco` | `ppo` |

**Preview eight pilot runs** (no training, SUMO or output files):

```bash
conda activate deeprlsc
unset CBWCE_RESUME CBWCE_STEPS CBWCE_EPISODES
bash scripts/training/08_all_parents.sh \
  --mode pilot --seed 9001 --visualization --dry-run
```

Remove `--dry-run` to execute the eight pilots, each with 160 learning steps. Use `--no-visualization` on a machine without a desktop. These short parent pilots do not replace the complete verification gate.

**Normal/full training: eight parents**, one publication seed, 1,000,000 learning steps per parent:

```bash
export CBWCE_GATE="$PWD/runs_eval/revised/verification/endpoint_monitoring_20260916_v1/gate.json"
unset CBWCE_RESUME CBWCE_STEPS CBWCE_EPISODES
bash scripts/training/08_all_parents.sh \
  --mode publication --seed 101 --gate "$CBWCE_GATE" \
  --monitor-every 50 --monitor-rollouts 3 --no-visualization
```

This is the complete parent matrix: **8 parents, one independently trained parent per network/controller combination, seed `101`**. Append `--dry-run` to preview or use `--visualization` to show SUMO. Publication training requires a current passing gate. The multi-seed option has been removed; `--all-seeds` now fails explicitly. Explicit matrix selections override `CBWCE_NETWORK` and `CBWCE_CONTROLLER`. A previous exported publication seed other than `101` must be replaced with `--seed 101`.

Outputs remain separate:

```text
runs/revised/<network>/<controller>/seed_<seed>/parent/<run_id>/
  manifest.json
  progress.jsonl
  result.json
  episode_metrics.jsonl
  learner_metrics.jsonl
  tensorboard/
  monitoring/round_000000/  # before learning; later rounds identify full episodes
  checkpoint_001000000/    # successful publication parent
```

If one job fails, the other jobs in that batch receive SIGINT so they can save failure state and clean up SUMO; no later batch starts. Ctrl+C behaves the same way. Rerunning the matrix starts new attempts, including combinations already completed; it does not skip or automatically resume them. Recover an interrupted combination with `01_parent.sh --resume` and the exact same-stage checkpoint, then run remaining combinations individually. Do not export one resume checkpoint for the matrix.

After parent training, follow Sections 2–3 for each network/controller/seed: select that parent's completed `result.json.checkpoint`, train its frozen-controller WCE, then start the five continuations from the same parent and matching WCE. `08_all_parents.sh` does not automatically run these later stages or evaluation. This preserves explicit checkpoint selection; it never guesses inputs from the newest filename.

## 6. Run WCE and all continuation methods with four workers

These launchers cover the same network/controller matrix as `08_all_parents.sh`. They call the existing single-run scripts, preserving budgets, visualization options, output roots and learning behavior.

| Matrix launcher | Existing single-run script | Runs with seed `101` |
|---|---|---:|
| [09_all_wce.sh](09_all_wce.sh) | `02_wce.sh` | 8 |
| [10_all_baseline.sh](10_all_baseline.sh) | `03_baseline.sh` | 8 |
| [11_all_random_group.sh](11_all_random_group.sh) | `04_random_group.sh` | 8 |
| [12_all_domain_randomization.sh](12_all_domain_randomization.sh) | `05_domain_randomization.sh` | 8 |
| [13_all_fixed_wce.sh](13_all_fixed_wce.sh) | `06_fixed_wce.sh` | 8 |
| [14_all_online_wce.sh](14_all_online_wce.sh) | `07_online_wce.sh` | 8 |
| [15_all_continuations.sh](15_all_continuations.sh) | `03`–`07`, in method order | 40 |

`15_all_continuations.sh` completes each method before starting the next: baseline, random groups, domain randomization, fixed WCE, then online WCE. The default four workers produce separate Grid and Monaco batches; `--workers 8` runs both networks together for that method. The accepted range is `1..8`. All methods reload their selected common parent; they do not continue from each other's results. Use either scripts `10`–`14` or script `15`; running both repeats the continuations as new attempts.

### 6.1 Select completed parents explicitly

A single `CBWCE_PARENT` cannot represent eight different parents. Create a selection template, then fill in the actual completed result paths. For a one-seed publication matrix:

```bash
conda activate deeprlsc
unset CBWCE_RESUME CBWCE_STEPS CBWCE_EPISODES
export CBWCE_GATE="$PWD/runs_eval/revised/verification/endpoint_monitoring_20260916_v1/gate.json"
export CBWCE_CHECKPOINTS='runs_eval/revised/selections/publication_seed101.json'

bash scripts/training/09_all_wce.sh --mode publication --seed 101 \
  --write-template "$CBWCE_CHECKPOINTS"
```

Template creation only writes the requested JSON file; it does not train, require a gate, or inspect placeholder paths. It refuses to overwrite an existing file. Open that file in your editor. Each of its eight rows has this format:

```json
{
  "network": "grid",
  "controller": "ia2c",
  "seed": 101,
  "parent_result": "/replace/with/exact/parent/run_id/result.json",
  "wce_result": "/replace/with/exact/wce/run_id/result.json"
}
```

The top-level object contains `"version": 1` and `"runs": [...]`. Initially replace every `parent_result` with the corresponding successful parent run's exact `result.json` path. Leave `wce_result` placeholders until WCE training completes. The helper reads the checkpoint from `result.json.checkpoint`; it never scans for the latest filename. Absolute paths and paths relative to the repository root are accepted, even when the launcher is called from another directory.

Before launching any jobs, the shared [_matrix.py](_matrix.py) checks the full requested matrix for duplicate/missing identities, completed run status, originating manifest integrity, network/controller/seed/stage/mode agreement and checkpoint file hashes. Publication inputs must have the prescribed completed budgets. Fixed/online WCE additionally require the WCE's parent hash to match the selected parent. Runtime availability, the gate and exact model signatures are checked by the existing runner when it launches each job. Dry-run checks the real selection records and launcher arguments, but does not start SUMO or create run outputs.

### 6.2 Train WCE for the complete matrix

Preview, then run eight frozen-parent WCE jobs:

```bash
bash scripts/training/09_all_wce.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" \
  --no-visualization --dry-run

bash scripts/training/09_all_wce.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" --no-visualization
```

Each job trains WCE for 500 episodes / 660,000 frozen-controller simulation steps. The controller remains at 1,000,000 learning steps. Grid outputs go to `output_adversary/revised/`; Monaco outputs go to `output_adversary_monaco/revised/`, followed by the existing network/controller/seed/stage/run hierarchy.

After completion, enter each matching WCE run's `result.json` path in that row's `wce_result`. Baseline, random groups and domain randomization only need `parent_result` and can run before WCE is available. Fixed WCE, online WCE and the combined continuation launcher require both fields for all selected identities.

### 6.3 Run baseline and the four comparison methods

Execute the desired method across all eight combinations:

```bash
# Baseline: original sequential demand schedule
bash scripts/training/10_all_baseline.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" --no-visualization

# Random demand groups
bash scripts/training/11_all_random_group.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" --no-visualization

# Dirichlet demand mixtures
bash scripts/training/12_all_domain_randomization.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" --no-visualization

# Fixed WCE model parameters
bash scripts/training/13_all_fixed_wce.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" --no-visualization

# Online WCE model updates
bash scripts/training/14_all_online_wce.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" --no-visualization
```

Alternatively, **one command executes all five methods sequentially**, stopping on the first failure:

```bash
bash scripts/training/15_all_continuations.sh --mode publication --seed 101 \
  --checkpoints "$CBWCE_CHECKPOINTS" --gate "$CBWCE_GATE" --no-visualization
```

Each continuation adds 1,320,000 controller-learning steps to its common parent, ending at 2,320,000. Baseline outputs remain under `runs/revised/`; comparison outputs remain under `output_coevolution/revised/` (grid) and `output_coevolution_real/revised/` (Monaco). Every run keeps its own manifest, progress, result and checkpoint bundle. Append `--dry-run` to preview; use `--visualization` to show SUMO.

### 6.4 Pilot checks and the complete single-seed study

The full study is already covered by Sections 5–6.3: **8 parents → 8 WCE runs → 40 continuations → 9,200 evaluations**. Use the same eight-row selection file throughout; update its WCE result entries after Stage II. All publication commands use `--seed 101`. There is no five-seed campaign.

For pilots, use a separate selection file and completed pilot parents, not publication checkpoints:

```bash
export CBWCE_CHECKPOINTS='runs_eval/revised/selections/pilot_seed9001.json'
bash scripts/training/09_all_wce.sh --mode pilot --seed 9001 \
  --write-template "$CBWCE_CHECKPOINTS"

# Fill parent_result entries, then preview eight WCE pilots.
bash scripts/training/09_all_wce.sh --mode pilot --seed 9001 \
  --checkpoints "$CBWCE_CHECKPOINTS" --visualization --dry-run

# Remove --dry-run above to train WCE; fill wce_result entries afterwards.
# Preview all 40 continuation pilots using those completed checkpoints.
bash scripts/training/15_all_continuations.sh --mode pilot --seed 9001 \
  --checkpoints "$CBWCE_CHECKPOINTS" --visualization --dry-run
```

Pilot defaults remain two WCE episodes and 2,640 additional learning steps per continuation. `--episodes` overrides pilot WCE only; `--steps` overrides pilot continuations only. Publication budgets cannot be overridden. Clear stale `CBWCE_STEPS`/`CBWCE_EPISODES` before changing stages. `--checkpoints` overrides `CBWCE_CHECKPOINTS`; explicit row selections override any exported `CBWCE_PARENT`/`CBWCE_WCE`. Matrix network/controller selection overrides their single-run environment variables.

**Recovery:** all launchers stop on failure or terminal Ctrl+C. Re-running starts fresh attempts; there is no automatic skip, recovery or checkpoint-map modification. Resume a failed individual job with scripts `02`–`07`, its original parent/WCE selections and its exact same-stage `--resume` checkpoint. Then launch only the remaining combinations individually to avoid repeats. The combined launcher does not launch evaluation or reporting; use the corresponding commands in the full reviewer guide after training.

For the remaining seed-101 continuation campaign, use the state-aware one-command entry point:

```bash
bash scripts/training/16_remaining_continuations_with_cleanup.sh --dry-run
bash scripts/training/16_remaining_continuations_with_cleanup.sh
```

It defaults to a global four-worker queue spanning Grid, Monaco and all remaining methods. Method order sets queue priority; every free slot starts the next task without waiting for slow tasks from the preceding method. It finds or generates a source-matching gate, skips complete jobs, and resumes from the latest valid checkpoint. Failed tasks join the queue tail for at most two retries without cancelling other tasks. Every 15 minutes it deletes non-tenth raw episode/trip files older than 60 minutes while preserving checkpoints, results, metrics, and monitoring. The default is four workers; options include `--workers 1..8`, `--gate PATH`, and `--force-new-gate`; rerunning the same command is safe. See `GLOBAL_QUEUE_zh.md` for details.

## Options, progress and resume

Resume only from a checkpoint using the same new reward protocol. `--resume` cannot convert a historical old-reward model. Keep the original `--monitor-every` and `--monitor-rollouts` settings; completed monitoring rounds are reused rather than repeated.

Command-line options override environment variables; use `--help` on any numbered script. All relative checkpoint/gate paths are resolved from the repository root, even when the script is invoked from elsewhere.

| Environment variable | Default / use |
|---|---|
| `CBWCE_MODE` | `pilot`; choose `publication` explicitly |
| `CBWCE_NETWORK`, `CBWCE_CONTROLLER` | `grid`, `ia2c` |
| `CBWCE_SEED` | If unset: `9001` in pilot mode, `101` in publication mode |
| `CBWCE_PARENT`, `CBWCE_WCE` | Explicit checkpoint directories, according to stage |
| `CBWCE_GATE` | Required for publication training |
| `CBWCE_VISUALIZATION` | If unset: `on` for pilot, `off` for publication |
| `CBWCE_STEPS` | Optional pilot parent/continuation budget; continuation must contain full 1,320-step episodes |
| `CBWCE_EPISODES` | Optional pilot WCE episode count |
| `CBWCE_RESUME` | Same-stage complete checkpoint; unset before starting a different stage/method |
| `CBWCE_MONITOR_EVERY` | Default `50` complete episodes; publication requires `50` |
| `CBWCE_MONITOR_ROLLOUTS` | Default `3` paired realizations; publication requires `3` |
| `CBWCE_WORKERS` | Matrix scripts `08`–`15`: concurrent jobs per network/method batch; default `4`, range `1..4` |
| `CBWCE_PYTHON` | Executable path/name; otherwise `python` from the active environment |

Examples without changing exported selections:

```bash
bash scripts/training/01_parent.sh --mode pilot --network monaco \
  --controller ma2c --seed 9002 --no-visualization --dry-run

bash scripts/training/07_online_wce.sh --resume '/exact/same-stage/checkpoint'
```

For resume, preserve the original stage, method, seed, parent/WCE identities and total stage budget. With nondefault pilots, repeat the original `--steps`/`--episodes`. Resume creates a new output directory; work after the saved checkpoint is discarded. Normal publication checkpoints are every ten complete episodes plus stage completion; these pilot scripts save every episode plus completion.

The banner identifies the phase before training. During execution the existing runner prints completed-episode progress, for example `grid ia2c parent episode=1 simulation_steps=160 learning_steps=160`. It also writes `progress.jsonl` every 120 controller transitions; queue values there describe the recent 600 simulated seconds. Do not interpret a quiet interval between log lines as a warm-up. There is no traffic warm-up: 160 steps equal 800 simulated seconds.

Use Ctrl+C to interrupt the current Python process and its owned SUMO session. These scripts use `exec` rather than a logging pipeline, preserving that signal path and exit status. Console output stays in the terminal; scientific records and checkpoints go to the printed run directory. Trust the final `result.json`, not the banner's expected checkpoint path, as proof of completion.

`--dry-run` calculates a provisional unique path but creates no run. A later actual invocation chooses a different new path. It validates selections and budget syntax but does not certify checkpoint contents, the gate or graphical display; the existing runner performs those checks at launch. No training is started merely by opening these scripts or this guide.
