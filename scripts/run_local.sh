#!/usr/bin/env bash
# Start one local command; run separate experiments in separate processes.
set -euo pipefail
CBWCE_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export CUDA_VISIBLE_DEVICES=-1
export CBWCE_ASSETS_ROOT="${CBWCE_ASSETS_ROOT:-$CBWCE_PROJECT_DIR/assets}"
export CBWCE_DATA_ROOT="${CBWCE_DATA_ROOT:-$(dirname -- "$CBWCE_PROJECT_DIR")/cbwce_data}"
export CBWCE_RUN_ROOT="${CBWCE_RUN_ROOT:-$(dirname -- "$CBWCE_PROJECT_DIR")/cbwce_runs}"
export CBWCE_CACHE_ROOT="${CBWCE_CACHE_ROOT:-$(dirname -- "$CBWCE_PROJECT_DIR")/cbwce_cache}"
cd -- "$CBWCE_PROJECT_DIR"
exec "${CBWCE_PYTHON:-python3}" main.py "$@"
