#!/usr/bin/env bash
# Shared command builder. Public entrypoints are the numbered scripts.
set -euo pipefail

if (( $# < 3 )); then
  printf 'Use one of the numbered scripts in scripts/training/.\n' >&2
  exit 2
fi
cbwce_stage=$1
cbwce_method=$2
cbwce_phase=$3
shift 3

cbwce_usage() {
  cat <<'USAGE'
Usage: bash scripts/training/<numbered-script>.sh [options]

  --mode pilot|publication       Default: pilot
  --network grid|monaco          Default: grid
  --controller ia2c|ma2c|iqll|ppo Default: ia2c
  --seed INTEGER                 Default: 9001 for pilot, 101 for publication
  --parent PATH                  Exact common parent checkpoint for WCE/continuation
  --wce PATH                     Exact pretrained WCE for fixed_wce/online_wce
  --gate PATH                    Current passing gate for publication
  --resume PATH                  Exact same-stage checkpoint; creates a new attempt
  --config PATH                  Pilot-only isolated revised candidate INI
  --steps INTEGER                Pilot parent/continuation budget only
  --episodes INTEGER             Pilot WCE budget only
  --visualization                Show SUMO (pilot default)
  --no-visualization             Headless SUMO (publication default)
  --monitor-every INTEGER        Complete episodes between tests (default 50)
  --monitor-rollouts INTEGER     Paired test realizations (default 3)
  --dry-run                      Print phase, budgets, paths and command; launch nothing
  --help                         Show this help

The equivalent environment variables are CBWCE_MODE, CBWCE_NETWORK,
CBWCE_CONTROLLER, CBWCE_SEED, CBWCE_PARENT, CBWCE_WCE, CBWCE_GATE,
CBWCE_RESUME, CBWCE_CONFIG, CBWCE_STEPS, CBWCE_EPISODES and CBWCE_VISUALIZATION (on/off).
Command-line options override them. CBWCE_PYTHON selects a Python executable;
otherwise use python from the activated deeprlsc environment.
USAGE
}

cbwce_fail() { printf 'ERROR / 错误: %s\n' "$*" >&2; exit 2; }
cbwce_mode=${CBWCE_MODE:-pilot}
cbwce_network=${CBWCE_NETWORK:-grid}
cbwce_controller=${CBWCE_CONTROLLER:-ia2c}
cbwce_seed=${CBWCE_SEED:-}
cbwce_parent=${CBWCE_PARENT:-}
cbwce_wce=${CBWCE_WCE:-}
cbwce_gate=${CBWCE_GATE:-}
cbwce_resume=${CBWCE_RESUME:-}
cbwce_config=${CBWCE_CONFIG:-}
cbwce_steps=${CBWCE_STEPS:-}
cbwce_episodes=${CBWCE_EPISODES:-}
cbwce_display=${CBWCE_VISUALIZATION:-}
cbwce_python=${CBWCE_PYTHON:-python}
cbwce_monitor_every=${CBWCE_MONITOR_EVERY:-50}
cbwce_monitor_rollouts=${CBWCE_MONITOR_ROLLOUTS:-3}
cbwce_dry_run=false

while (( $# )); do
  case "$1" in
    --help|-h) cbwce_usage; exit 0 ;;
    --dry-run) cbwce_dry_run=true; shift ;;
    --visualization) cbwce_display=on; shift ;;
    --no-visualization) cbwce_display=off; shift ;;
    --mode|--network|--controller|--seed|--parent|--wce|--gate|--resume|--config|--steps|--episodes|--monitor-every|--monitor-rollouts)
      (( $# >= 2 )) || cbwce_fail "Missing value for $1"
      [[ -n "$2" ]] || cbwce_fail "Empty value for $1"
      case "$1" in
        --mode) cbwce_mode=$2 ;;
        --network) cbwce_network=$2 ;;
        --controller) cbwce_controller=$2 ;;
        --seed) cbwce_seed=$2 ;;
        --parent) cbwce_parent=$2 ;;
        --wce) cbwce_wce=$2 ;;
        --gate) cbwce_gate=$2 ;;
        --resume) cbwce_resume=$2 ;;
        --config) cbwce_config=$2 ;;
        --steps) cbwce_steps=$2 ;;
        --episodes) cbwce_episodes=$2 ;;
        --monitor-every) cbwce_monitor_every=$2 ;;
        --monitor-rollouts) cbwce_monitor_rollouts=$2 ;;
      esac
      shift 2 ;;
    *) cbwce_fail "Unknown option: $1 (use --help)" ;;
  esac
done

case "$cbwce_stage:$cbwce_method" in
  parent:baseline|wce:baseline|continue:baseline|continue:random_group|continue:domain_randomization|continue:fixed_wce|continue:online_wce) ;;
  *) cbwce_fail 'Unknown stage/method' ;;
esac
case "$cbwce_mode" in pilot|publication) ;; *) cbwce_fail 'Mode must be pilot or publication' ;; esac
case "$cbwce_network" in grid|monaco) ;; *) cbwce_fail 'Network must be grid or monaco' ;; esac
case "$cbwce_controller" in ia2c|ma2c|iqll|ppo) ;; *) cbwce_fail 'Invalid controller' ;; esac
if [[ -z "$cbwce_seed" ]]; then
  if [[ "$cbwce_mode" == pilot ]]; then cbwce_seed=9001; else cbwce_seed=101; fi
fi
[[ "$cbwce_seed" =~ ^[1-9][0-9]*$ && ${#cbwce_seed} -le 10 ]] || cbwce_fail 'Seed must be a positive integer'
(( cbwce_seed < 4294967296 )) || cbwce_fail 'Seed must fit NumPy RandomState (less than 4294967296)'
if [[ "$cbwce_mode" == publication ]]; then
  case "$cbwce_seed" in 101) ;; *) cbwce_fail 'Publication uses one seed: 101' ;; esac
  [[ -n "$cbwce_gate" ]] || cbwce_fail 'Publication requires --gate or CBWCE_GATE'
  [[ -z "$cbwce_steps" && -z "$cbwce_episodes" ]] || cbwce_fail 'Publication budgets cannot be overridden; unset CBWCE_STEPS and CBWCE_EPISODES'
  [[ -z "$cbwce_config" ]] || cbwce_fail 'Publication forbids --config; update the tracked revised INI only through the tuning gate'
else
  case "$cbwce_seed" in 101) cbwce_fail 'Pilot seed must not overlap publication seeds' ;; esac
fi
if [[ "$cbwce_stage" != parent && -z "$cbwce_parent" ]]; then
  cbwce_fail 'Select the exact common parent with --parent or CBWCE_PARENT'
fi
if [[ "$cbwce_method" == fixed_wce || "$cbwce_method" == online_wce ]]; then
  [[ -n "$cbwce_wce" ]] || cbwce_fail 'Select pretrained WCE with --wce or CBWCE_WCE'
fi
if [[ -z "$cbwce_display" ]]; then
  if [[ "$cbwce_mode" == pilot ]]; then cbwce_display=on; else cbwce_display=off; fi
fi
case "$cbwce_display" in on|off) ;; *) cbwce_fail 'CBWCE_VISUALIZATION must be on or off' ;; esac

cbwce_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cbwce_root="$(cd -- "$cbwce_script_dir/../.." && pwd)"
cd -- "$cbwce_root"
command -v "$cbwce_python" >/dev/null || cbwce_fail 'Python not found; activate deeprlsc or set CBWCE_PYTHON'
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}

# Pure path/budget calculation: no directory creation or simulator startup.
cbwce_metadata="$("$cbwce_python" - "$cbwce_stage" "$cbwce_method" "$cbwce_network" "$cbwce_controller" "$cbwce_seed" "$cbwce_mode" "$cbwce_steps" "$cbwce_episodes" <<'PY'
import sys
from experiments.protocol import settings, output_path
stage, method, network, controller, seed, mode, steps, episodes = sys.argv[1:]
p = settings()
pilot = mode == 'pilot'
transitions_per_episode = p['training_seconds'] // p['control_seconds']
if stage == 'wce':
    if steps:
        raise SystemExit('WCE uses --episodes, not --steps')
    budget = int(episodes or 2) if pilot else p['offline_episodes']
    if budget <= 0:
        raise SystemExit('WCE episodes must be positive')
    count = budget * transitions_per_episode
    description = '{} WCE episodes / {} frozen-controller simulation steps; controller learning disabled'.format(budget, count)
else:
    if episodes:
        raise SystemExit('Parent/continuation uses --steps, not --episodes')
    default = 160 if stage == 'parent' else 2640
    budget = int(steps or default) if pilot else p['parent_steps' if stage == 'parent' else 'continuation_steps']
    if budget <= 0 or (stage == 'continue' and budget % transitions_per_episode):
        raise SystemExit('Budget must be positive; continuation requires whole training episodes')
    count = budget
    description = '{} {}controller-learning steps'.format(budget, 'additional ' if stage == 'continue' else '')
print(output_path(stage, network, controller, int(seed), method, pilot))
print(budget)
print(description)
print('checkpoint_{:09d}'.format(count))
PY
)"
mapfile -t cbwce_info <<< "$cbwce_metadata"
cbwce_output=${cbwce_info[0]}
cbwce_budget=${cbwce_info[1]}
cbwce_cmd=("$cbwce_python" -u main.py experiment "$cbwce_stage" --network "$cbwce_network"
  --controller "$cbwce_controller" --seed "$cbwce_seed" --output "$cbwce_output")
cbwce_cmd+=(--monitor-every "$cbwce_monitor_every" --monitor-rollouts "$cbwce_monitor_rollouts")
if [[ "$cbwce_stage" == continue ]]; then cbwce_cmd+=(--method "$cbwce_method"); fi
if [[ "$cbwce_mode" == pilot ]]; then
  cbwce_cmd+=(--pilot --checkpoint-every 1)
  if [[ "$cbwce_stage" == wce ]]; then cbwce_cmd+=(--episodes "$cbwce_budget"); else cbwce_cmd+=(--steps "$cbwce_budget"); fi
else
  cbwce_cmd+=(--gate "$cbwce_gate")
fi
if [[ "$cbwce_display" == on ]]; then cbwce_cmd+=(--visualization); else cbwce_cmd+=(--no-visualization); fi
if [[ "$cbwce_stage" != parent ]]; then cbwce_cmd+=(--parent "$cbwce_parent"); fi
if [[ "$cbwce_method" == fixed_wce || "$cbwce_method" == online_wce ]]; then cbwce_cmd+=(--wce "$cbwce_wce"); fi
if [[ -n "$cbwce_resume" ]]; then cbwce_cmd+=(--resume "$cbwce_resume"); fi
if [[ -n "$cbwce_config" ]]; then cbwce_cmd+=(--config "$cbwce_config"); fi

printf '\n============================================================\n'
printf 'TRAINING PHASE / 训练阶段: %s\n' "$cbwce_phase"
printf 'Mode / 模式: %s    Network / 路网: %s    Controller / 控制器: %s\n' "$cbwce_mode" "$cbwce_network" "$cbwce_controller"
printf 'Seed / 种子: %s    Method / 方法: %s    SUMO GUI: %s\n' "$cbwce_seed" "$cbwce_method" "$cbwce_display"
printf 'Budget / 预算: %s\n' "${cbwce_info[2]}"
if [[ "$cbwce_stage" != parent ]]; then printf 'Common parent / 共同父模型: %s\n' "$cbwce_parent"; fi
if [[ "$cbwce_method" == fixed_wce || "$cbwce_method" == online_wce ]]; then printf 'Pretrained WCE / 预训练 WCE: %s\n' "$cbwce_wce"; fi
if [[ -n "$cbwce_resume" ]]; then printf 'Resume / 恢复: %s (same total stage budget / 原阶段总预算)\n' "$cbwce_resume"; fi
printf 'Output / 输出: %s\n' "$cbwce_output"
printf 'Final checkpoint after successful completion / 完成后最终检查点: %s/%s\n' "$cbwce_output" "${cbwce_info[3]}"
printf 'COMMAND:'
printf ' %q' "${cbwce_cmd[@]}"
printf '\n============================================================\n'
if "$cbwce_dry_run"; then
  printf 'DRY RUN / 仅预览: no training or files created. Checkpoint, gate and display availability are checked at actual launch.\n'
  exit 0
fi
printf 'START / 开始: live episode progress follows; Ctrl+C interrupts this attempt.\n\n'
# Replace this shell so interrupts reach the existing Python/SUMO cleanup path.
exec "${cbwce_cmd[@]}"
