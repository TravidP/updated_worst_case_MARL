#!/usr/bin/env bash
set -euo pipefail
cbwce_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$cbwce_script_dir/_launch.sh" wce baseline 'II — WCE against frozen parent / 冻结父模型训练 WCE' "$@"
