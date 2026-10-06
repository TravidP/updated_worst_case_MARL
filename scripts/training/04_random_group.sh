#!/usr/bin/env bash
set -euo pipefail
cbwce_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$cbwce_script_dir/_launch.sh" continue random_group 'III-2 — Random-group continuation / 随机需求组续训' "$@"
