#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/sdc_joran/Journal/deeprl_signal_control
GATE=${CBWCE_GATE:-}
MODE=${1:-}

usage() {
  echo 'Usage: evaluate_publication_seed101.sh --preflight | --list | --execute'
  echo 'This script will not execute formal evaluation unless protocol v7 and storage gates pass.'
}

preflight() {
  cd "$ROOT"
  version=$(python3 -c "import json; print(json.load(open('config/revised/protocol.json'))['version'])")
  if [ "$version" != 7 ]; then
    echo "BLOCKED: protocol v7 is required; current version is $version." >&2
    return 2
  fi
  if [ -z "$GATE" ]; then echo "BLOCKED: export CBWCE_GATE to a current protocol-v7 gate.json." >&2; return 4; fi
  available_kb=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
  if [ "$available_kb" -lt 62914560 ]; then
    echo "BLOCKED: at least 60 GiB free is required." >&2
    return 3
  fi
  conda run -n deeprlsc python -c "from experiments.runner import require_gate; require_gate(r'''$GATE'''); print('PASS current verification gate')"
}

run_all() {
  cd "$ROOT"
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ia2c/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ia2c/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/grid/ia2c/seed_101/baseline/publication_20260919T110516_bffa1fe4/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ia2c/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ia2c/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ia2c/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ia2c/seed_101/random_group/publication_20260922T114628_37d306e3/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ia2c/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ia2c/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ia2c/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ia2c/seed_101/domain_randomization/publication_20260923T145705_7a6b9270/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ia2c/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ia2c/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ia2c/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ia2c/seed_101/fixed_wce/publication_20260924T054007_773e599b/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ia2c/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ia2c/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ia2c/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ia2c/seed_101/online_wce/publication_20260925T092820_79e065be/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ia2c/online_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ma2c/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ma2c/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/grid/ma2c/seed_101/baseline/publication_20260919T110516_de174e9f/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ma2c/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ma2c/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ma2c/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ma2c/seed_101/random_group/publication_20260922T114628_4242218e/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ma2c/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ma2c/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ma2c/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ma2c/seed_101/domain_randomization/publication_20260923T145705_080f7af1/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ma2c/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ma2c/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ma2c/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ma2c/seed_101/fixed_wce/publication_20260924T082634_aa85666b/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ma2c/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ma2c/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ma2c/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ma2c/seed_101/online_wce/publication_20260925T102236_52e76186/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ma2c/online_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/iqll/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/iqll/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/grid/iqll/seed_101/baseline/publication_20260921T101956_966174e3/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/iqll/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/iqll/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/iqll/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/iqll/seed_101/random_group/publication_20260922T114628_05ccf7be/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/iqll/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/iqll/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/iqll/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/iqll/seed_101/domain_randomization/publication_20260923T145705_6c1d3fb4/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/iqll/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/iqll/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/iqll/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/iqll/seed_101/fixed_wce/publication_20260924T102114_f27b5c6f/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/iqll/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/iqll/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/iqll/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/iqll/seed_101/online_wce/publication_20260925T103928_e2318819/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/iqll/online_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ppo/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ppo/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/grid/ppo/seed_101/baseline/publication_20260921T091601_2d3778ff/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ppo/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ppo/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ppo/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ppo/seed_101/random_group/publication_20260922T114628_8937c084/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ppo/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ppo/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ppo/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ppo/seed_101/domain_randomization/publication_20260923T145705_6a9f9cf7/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ppo/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ppo/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ppo/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ppo/seed_101/fixed_wce/publication_20260924T112233_d23fa5d8/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ppo/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/grid/ppo/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/grid/ppo/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network grid --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution/revised/grid/ppo/seed_101/online_wce/publication_20260925T111448_26956f37/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/grid/ppo/online_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ia2c/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ia2c/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/monaco/ia2c/seed_101/baseline/publication_20260921T101956_d364f6a8/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ia2c/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ia2c/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ia2c/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ia2c/seed_101/random_group/publication_20260922T114628_0aad1a36/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ia2c/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ia2c/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ia2c/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ia2c/seed_101/domain_randomization/publication_20260923T145705_25ac16f3/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ia2c/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ia2c/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ia2c/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ia2c/seed_101/fixed_wce/publication_20260924T132346_8a25e1dd/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ia2c/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ia2c/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ia2c/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ia2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ia2c/seed_101/online_wce/publication_20260925T133729_5eb7e27b/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ia2c/online_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ma2c/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ma2c/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/monaco/ma2c/seed_101/baseline/publication_20260921T232411_e7c37342/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ma2c/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ma2c/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ma2c/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ma2c/seed_101/random_group/publication_20260922T114628_a9f21707/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ma2c/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ma2c/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ma2c/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ma2c/seed_101/domain_randomization/publication_20260923T145705_0e8b380e/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ma2c/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ma2c/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ma2c/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ma2c/seed_101/fixed_wce/publication_20260924T152812_121189a7/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ma2c/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ma2c/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ma2c/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ma2c --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ma2c/seed_101/online_wce/publication_20260925T182406_2c5ff93d/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ma2c/online_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/iqll/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/iqll/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/monaco/iqll/seed_101/baseline/publication_20260921T101956_50fdfb9e/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/iqll/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/iqll/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/iqll/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/iqll/seed_101/random_group/publication_20260922T114628_9ead06e3/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/iqll/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/iqll/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/iqll/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/iqll/seed_101/domain_randomization/publication_20260923T145705_a1c98737/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/iqll/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/iqll/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/iqll/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/iqll/seed_101/fixed_wce/publication_20260925T010612_973d5910/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/iqll/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/iqll/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/iqll/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller iqll --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/iqll/seed_101/online_wce/publication_20260926T015715_cac25e47/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/iqll/online_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ppo/baseline' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ppo/baseline' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/runs/revised/monaco/ppo/seed_101/baseline/publication_20260921T101956_21fa0919/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ppo/baseline'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ppo/random_group' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ppo/random_group' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ppo/seed_101/random_group/publication_20260922T114628_0881e4fb/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ppo/random_group'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ppo/domain_randomization' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ppo/domain_randomization' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ppo/seed_101/domain_randomization/publication_20260923T145705_294f5ccd/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ppo/domain_randomization'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ppo/fixed_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ppo/fixed_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ppo/seed_101/fixed_wce/publication_20260925T052439_8e7a81ea/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ppo/fixed_wce'
  if [ -e '/home/sdc_joran/Journal/deeprl_signal_control/runs_eval/revised/publication_seed101_v1/monaco/ppo/online_wce' ]; then echo 'BLOCKED: output already exists: runs_eval/revised/publication_seed101_v1/monaco/ppo/online_wce' >&2; return 4; fi
  conda run -n deeprlsc python main.py experiment evaluate \
    --network monaco --controller ppo --seed 101 \
    --parent '/home/sdc_joran/Journal/deeprl_signal_control/output_coevolution_real/revised/monaco/ppo/seed_101/online_wce/publication_20260926T094349_1a6ecd92/checkpoint_001320000' --gate "$GATE" \
    --suite all --rollouts 10 --no-visualization \
    --output 'runs_eval/revised/publication_seed101_v1/monaco/ppo/online_wce'
}

case "$MODE" in
  --preflight) preflight ;;
  --list) sed -n '/^run_all()/,/^}/p' "$0" ;;
  --execute) preflight; run_all ;;
  *) usage; exit 1 ;;
esac
