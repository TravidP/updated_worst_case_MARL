# CB-WCE: multi-agent signal control under changing traffic demand

## Endpoint rewards and native TensorBoard (protocol 5)

Controller rewards use negative queue at the five-second endpoint, without division by 100; MA2C retains 0.9 neighborhood weighting. Monaco IA2C/MA2C batches are now 120. Training episodes remain 6,600 seconds. `train/reward_by_learning_step` records every actual learner reward; `train/episode/mean_total_queue` summarizes a full episode. Before learning, every 50 complete episodes and at final budget, three paired 600-second Uniform tests save per-second CSV, NPZ and TensorBoard curves.

See [settings, commands, metric definitions and output locations](docs/TRAINING_MONITORING.md). Defaults are `--monitor-every 50 --monitor-rollouts 3`; pilots may shorten the interval. Monitoring records fixed rollouts, TensorBoard metrics and checkpoints but no longer terminates training for degradation relative to the initial monitor; final model quality is decided by the complete paired evaluation. Start fresh parents and generate a gate matching the current source; historical checkpoints remain preserved. Exact originals and checksums are archived under `docs/history/endpoint_monitoring_20260916T095506Z/`.

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


[简体中文](README_zh.md) · [Interactive workspace](docs/site/dist/index.html) · [Private online guide](https://cb-wce-training-workspace.loyal-bowl-4834.chatgpt.site) · [Integration report](docs/INTEGRATION_REPORT.md) · [Full experimental protocol](reviewer_revision_plan.md)

This project investigates whether challenging traffic-demand training improves signal-controller robustness. Controllers learn signal actions; CB-WCE observes traffic and learns mixtures of eleven demand profiles. Improved robustness is a research hypothesis to test through paired evaluation and uncertainty reporting.

The study uses the **5×5 grid and Monaco subnet**, with **IA2C, MA2C, IQL-LR (`iqll`), and PPO**. Corrected code is integrated into `agents/`, `envs/`, and `experiments/`. The `revision/` package retains compatibility entrypoints and historical verification evidence. Existing data, checkpoints, and results remain at their original paths.

## Quick start

Run from the repository root:

```bash
conda activate deeprlsc
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
python main.py experiment prepare
python main.py experiment check
python main.py experiment dashboard
```

Open the [local workspace](http://127.0.0.1:8765/). It defaults to pilot mode and launches one selected job at a time. Review settings and checkpoints, preview the command, then start. Progress, logs, stop, and resume controls are provided. Resume creates a new attempt; if no complete checkpoint exists, restart that stage.

The established runtime is Python 3.6.13, TensorFlow 1.12.0, NumPy 1.19.5, and the installed SUMO build. Preserve the existing `deeprlsc` environment. The historical `environment.yml` is a reference; check platform compatibility before recreating it on another machine. The HTML needs no frontend dependency installation.

The private Sites version provides instructions, commands, comics, and exported results. Training runs through the local service. Opening the HTML directly also provides a guide without launching processes.

## SUMO visualization

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

The local workspace has the same **SUMO visualization** On/Off selector. Use a desktop with `sumo-gui`; on Linux it requires an accessible `DISPLAY`. Leave visualization off for timing comparisons. Restart an idle dashboard server and refresh the page after upgrading. See [display behavior, recording and verification](docs/VISUALIZATION.md). This source change invalidates the old publication gate; run verification again before publication training.

## Repository map

| Location | Purpose |
|---|---|
| `agents/` | Existing policies plus shared controller, WCE, and recurrent implementations |
| `envs/` | Existing networks plus corrected one-second SUMO measurements |
| `experiments/` | Protocol, paths, training, evaluation, checkpoints, reports, and local service |
| `config/revised/` | Shared protocol and effective configuration copies |
| `data_traffic/revised/` | Grid train/validation/test inputs |
| `real_net_subnet/demand_groups/revised/` | Monaco train/validation/test inputs |
| `runs/revised/` | Common parents and baseline continuations |
| `output_adversary*/revised/` | Offline WCE outputs for the two networks |
| `output_coevolution*/revised/` | Four comparison continuations for the two networks |
| `runs_eval/revised/` | Evaluation, verification, and local job records |
| `output_result/revised/`, `figs/revised/` | Tables, dashboard exports, scientific figures |
| `docs/history/` | Exact pre-integration snapshots and checksum index |
| `docs/site/dist/` | English/Chinese interactive workspace and comics |

Training uses `<stage-root>/<network>/<controller>/seed_<seed>/<stage-or-method>/<run_id>/`. Evaluation adds split, scenario, rollout, and attempt identifiers. Historical folders remain in place.

## Training procedure

Use the [single-run and sequential shell launchers](scripts/training/README.md) for initial parent training, frozen-parent WCE and all five continuations. Each prints its training phase, settings, exact command and output path. Preview any stage with `--dry-run`; see the [Chinese instructions](scripts/training/README_zh.md).

| Stage | Budget | Full-study output |
|---|---:|---|
| Common parent | 1,000,000 controller-learning steps | 8 independent parents |
| WCE against frozen parent | 500 episodes; 660,000 frozen-controller simulation steps | 8 WCE models |
| Five continuations | 1,320,000 additional learning steps each | 40 final controllers |
| Paired evaluation | 23 scenarios × 10 rollouts | 9,200 rollouts |

The five methods are `baseline`, `random_group`, `domain_randomization`, `fixed_wce`, and `online_wce`. All branch from the same corresponding parent and finish at 2,320,000 controller-learning steps. Fixed/online WCE share their pretrained WCE. Fixed **model parameters** still permit state-dependent **demand-mixture weights**.

Run verification before publication training:

```bash
python main.py experiment verify --workers 4
```

Use the new `gate.json` reported by verification. Code or input changes invalidate an earlier gate. The single publication training seed is `101`; pilots use separate seeds such as `9001`.

A short parent pilot:

```bash
python main.py experiment parent --network grid --controller ia2c \
  --seed 9001 --pilot --visualization --steps 160
```

The runner prints its actual output and checkpoint paths. Select explicit checkpoints for the next stages. Replace the quoted paths below with actual publication checkpoint and gate paths; the short pilot parent is not a publication parent:

```bash
python main.py experiment wce --network grid --controller ia2c \
  --seed 101 --gate '/replace/with/passing/gate.json' --parent '/replace/with/publication/parent/checkpoint_001000000'

python main.py experiment continue --network grid --controller ia2c \
  --seed 101 --method online_wce --gate '/replace/with/passing/gate.json' \
  --parent '/replace/with/publication/parent/checkpoint_001000000' --wce '/replace/with/publication/wce/checkpoint_000660000'
```

Resume with `--resume <same-stage-checkpoint>`, preserving the original stage, method, seed, total budget, and parent/WCE identities. Normal checkpoints are saved every ten complete episodes; `--checkpoint-every 1` saves every episode. Directory suffixes count stage simulation steps, not cumulative controller-learning steps. Full restoration includes model/optimizer variables, buffers, RNG streams, and counters. Old weight-only checkpoints cannot substitute for complete training state.

## Evaluation and reports

Traffic is generated from normalized network-specific OD profiles using redistribution, fixed mixtures, directional switching and peak loads. Each scenario becomes ten complete JSON vehicle schedules with fixed routes/departures/speed factors. See [how validation and test traffic are generated](reviewer_revision_plan.md#traffic-generation), including both networks, seed roles, file locations and the distinction between expected rates and sampled counts.

```bash
python main.py experiment prepare --materialize --network grid
python main.py experiment prepare --materialize --network monaco

python main.py experiment evaluate --network grid --controller ia2c \
  --seed 101 --parent '/replace/with/final/checkpoint_001320000' --suite all

python main.py experiment report --input '/replace/with/selected/run'
```

`--suite all` covers eleven seen profiles and twelve new scenarios; `validation` covers six separate scenarios. One evaluation job processes its selected final controller sequentially. It never automatically starts the full training matrix. Pilot checkpoints require `--pilot`; use `--rollouts 1` for a small evaluation check.

Reports include CSV, PNG, SVG, and `dashboard.json`. Import the JSON through Results & plots. Report the mean and sample SD across ten evaluation rollouts per scenario. With one training seed, training-seed SD and confidence intervals are unavailable; comparison summaries are conditional on this trained model. Peak maps share the `[1200,2400)` window and consistent scales.

The primary metric is mean total queue on controlled approaches, sampled every second and reported in vehicles. Lower is better. Incomplete rollouts cannot be padded into valid observations. Pilot curves do not establish publication performance.

## Historical entrypoints and troubleshooting

`python main.py train/evaluate` and the original standalone scripts retain their historical workflow. Use `python main.py experiment ...` for the corrected study. The original README is preserved in [the history snapshot](docs/history/pre_integration/README.md.txt). Historical `introduction.md` contains outdated environment, path, and Git advice and is not the new training guide.

- Missing TensorFlow: activate `deeprlsc`; do not use the default Python 3.13 runtime.
- SUMO startup failure: check `sumo --version` and local TraCI socket access.
- Incompatible checkpoint: verify network, controller, stage, budget, and parent identity.
- Existing output directory: choose a new attempt and retain the previous one.
- Stale gate: repeat verification after source or input changes.
- Duplicate report observations: select one explicit valid attempt rather than combining reruns.

## Attribution and citation

This project builds on the multi-agent traffic-control implementation by Tianshu Chu and collaborators and retains its [MIT license](LICENSE). Cite the original method and describe the CB-WCE extension separately:

```bibtex
@article{chu2019multi,
  title={Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control},
  author={Chu, Tianshu and Wang, Jie and Codec{\`a}, Lara and Li, Zhaojian},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2019},
  publisher={IEEE}
}
```

[Research comic](docs/site/dist/assets/cbwce-comic-en-v1.png) · [Full protocol](reviewer_revision_plan.md) · [Historical audit](reviewer_revision_audit_2026-09-14.md)

## Repository and portable results

Source, inputs, configuration and compact Grid/Monaco results are versioned. Training output, checkpoints, generated frozen demand and full series remain local. See [data policy](docs/REPOSITORY_DATA_POLICY.md) and [restore instructions](docs/evaluation_workbook/grid_results_site/RESTORE.md). Viewing requires Python 3.10+; training uses the separate legacy environment. No public model/data Release exists yet.
