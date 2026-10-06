#!/usr/bin/env bash
set -euo pipefail
dashboard_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$dashboard_dir/dashboard.py" "${@:-start}"
