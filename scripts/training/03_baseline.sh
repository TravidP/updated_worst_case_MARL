#!/usr/bin/env bash
set -euo pipefail
cbwce_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$cbwce_script_dir/_launch.sh" continue baseline 'III-1 — Baseline continuation / 基线续训' "$@"
