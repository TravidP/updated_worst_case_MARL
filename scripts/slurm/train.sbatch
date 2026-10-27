#!/bin/bash
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --array=0-0%4
#SBATCH --output=cbwce-%A_%a.log
set -euo pipefail
# Submit: sbatch train.sbatch /scratch/ACCOUNT/cbwce/jobs.json
# Each task entry contains image, code_release, data_root, run_root, cache_root,
# and an argv array. No shell fragments or eval are accepted.
export TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
: "${TMPDIR:?Hábrók must provide node-local TMPDIR}"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
module load Apptainer
worker_pid=''
forward_signal() { if [[ -n "$worker_pid" ]]; then kill -USR1 "$worker_pid" 2>/dev/null || true; fi; }
trap forward_signal USR1 TERM INT
python3 scripts/slurm/launch.py "$1" "${SLURM_ARRAY_TASK_ID:-0}" &
worker_pid=$!
set +e
wait "$worker_pid"
code=$?
while kill -0 "$worker_pid" 2>/dev/null; do wait "$worker_pid"; code=$?; done
exit "$code"
