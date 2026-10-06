#!/usr/bin/env bash
set -euo pipefail
cbwce_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$cbwce_script_dir/_launch.sh" continue online_wce 'III-5 — Online-WCE continuation / 在线 WCE 续训' "$@"
