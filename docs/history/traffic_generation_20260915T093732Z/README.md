# CB-WCE: multi-agent signal control under changing traffic demand

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

| Stage | Budget | Full-study output |
|---|---:|---|
| Common parent | 1,000,000 controller-learning steps | 40 independent parents |
| WCE against frozen parent | 500 episodes; 660,000 frozen-controller simulation steps | 40 WCE models |
| Five continuations | 1,320,000 additional learning steps each | 200 final controllers |
| Paired evaluation | 23 scenarios × 10 rollouts | 46,000 rollouts |

The five methods are `baseline`, `random_group`, `domain_randomization`, `fixed_wce`, and `online_wce`. All branch from the same corresponding parent and finish at 2,320,000 controller-learning steps. Fixed/online WCE share their pretrained WCE. Fixed **model parameters** still permit state-dependent **demand-mixture weights**.

Run verification before publication training:

```bash
python main.py experiment verify --workers 4
```

Use the new `gate.json` reported by verification. Code or input changes invalidate an earlier gate. Publication training seeds are `101,202,303,404,505`; pilots use separate seeds such as `9001`.

A short parent pilot:

```bash
python main.py experiment parent --network grid --controller ia2c \
  --seed 9001 --pilot --steps 160
```

The runner prints its actual output and checkpoint paths. Select explicit checkpoints for the next stages. Angle-bracket values below must be replaced with those paths:

```bash
python main.py experiment wce --network grid --controller ia2c \
  --seed 101 --gate <gate.json> --parent <parent-checkpoint>

python main.py experiment continue --network grid --controller ia2c \
  --seed 101 --method online_wce --gate <gate.json> \
  --parent <parent-checkpoint> --wce <wce-checkpoint>
```

Resume with `--resume <same-stage-checkpoint>`, preserving the original stage, method, seed, total budget, and parent/WCE identities. Normal checkpoints are saved every ten complete episodes; `--checkpoint-every 1` saves every episode. Directory suffixes count stage simulation steps, not cumulative controller-learning steps. Full restoration includes model/optimizer variables, buffers, RNG streams, and counters. Old weight-only checkpoints cannot substitute for complete training state.

## Evaluation and reports

```bash
python main.py experiment prepare --materialize --network grid
python main.py experiment prepare --materialize --network monaco

python main.py experiment evaluate --network grid --controller ia2c \
  --seed 101 --parent <final-checkpoint> --suite all

python main.py experiment report --input <explicit-evaluation-or-training-directory>
```

`--suite all` covers eleven seen profiles and twelve new scenarios; `validation` covers six separate scenarios. One evaluation job processes its selected final controller sequentially. It never automatically starts the full training matrix. Pilot checkpoints require `--pilot`; use `--rollouts 1` for a small evaluation check.

Reports include CSV, PNG, SVG, and `dashboard.json`. Import the JSON through Results & plots. Rollout SD and independent-training uncertainty remain separate; publication intervals require five complete training seeds. Peak maps share the `[1200,2400)` window and consistent scales.

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
