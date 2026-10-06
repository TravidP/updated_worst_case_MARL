#!/usr/bin/env bash
# Global four-worker queue by default: refill across methods, resume and clean up.
set -eo pipefail

cbwce_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cbwce_root="$(cd -- "$cbwce_script_dir/../.." && pwd)"
cd -- "$cbwce_root"

if [[ "${CONDA_DEFAULT_ENV:-}" != "deeprlsc" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate deeprlsc
  else
    printf 'ERROR: activate the deeprlsc conda environment first.\n' >&2
    exit 2
  fi
fi

set -u
unset CBWCE_PARENT CBWCE_WCE CBWCE_RESUME CBWCE_STEPS CBWCE_EPISODES
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}

exec python -u "$cbwce_script_dir/_remaining_campaign.py" "$@"
