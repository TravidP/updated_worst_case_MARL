#!/usr/bin/env bash
set -euo pipefail
cbwce_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$cbwce_script_dir/_launch.sh" continue fixed_wce 'III-4 — Fixed-WCE continuation / 固定 WCE 续训' "$@"
