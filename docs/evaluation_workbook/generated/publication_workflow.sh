#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
LAUNCHER="$ROOT/docs/evaluation_workbook/generated/evaluate_publication_seed101.sh"
AUDITOR="$ROOT/docs/evaluation_workbook/validate_publication_release.py"
SCHEDULER="$ROOT/docs/evaluation_workbook/run_publication_parallel.py"
SELECTION="$ROOT/runs_eval/revised/selections/final_evaluation_seed101.json"
RAW_ROOT="$ROOT/runs_eval/revised/publication_seed101_v1"
REPORT_ROOT="$ROOT/output_result/revised/publication_seed101_v1"
GATE="${CBWCE_GATE:-}"
MODE="${1:---plan}"
EVALUATION_WORKERS="${CBWCE_EVALUATION_WORKERS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

usage() {
  printf '%s\n' \
    'Usage: publication_workflow.sh MODE [arguments]' \
    '' \
    'Safe/read-only modes:' \
    '  --plan        Print the recommended sequence (default).' \
    '  --status      Inspect protocol, gate, disk, selection, and outputs.' \
    '  --preflight   Run scenario, storage, selection, and environment checks.' \
    '  --list        Print the exact 40-suite evaluation commands.' \
    '  --audit       Read-only audit of raw results and optional report.' \
    '' \
    'Explicit execution modes:' \
    '  --verify [OUTPUT]       Run tests and eight SUMO pilot cases.' \
    '  --materialize           Materialize v7 demand artifacts.' \
    '  --canary N C M          Run validation-only canary.' \
    '  --execute               Launch 40 formal suites with four workers by default.' \
    '  --report                Build report and run read-only audit.' \
    '  --dashboard             Start local dashboard on port 8765.'
}

plan() {
  printf '%s\n' \
    '1. Implement protocol v7 random switching across all 11 seen profiles.' \
    '2. Run --verify and export CBWCE_GATE to the new gate.json.' \
    '3. Run --materialize and inspect scenario/artifact hashes.' \
    '4. Run --preflight and --list; independently review all 40 suites.' \
    '5. Run a validation-only --canary; do not consume test early.' \
    '6. Run --execute with four suite workers after review and storage approval.' \
    '7. Run --report and --audit; require 9,200 valid and zero rejected.' \
    '' \
    'Confirmation tokens:' \
    '  CONFIRM_VERIFICATION=RUN_8_PILOTS' \
    '  CONFIRM_MATERIALIZE=BUILD_V7_ARTIFACTS' \
    '  CONFIRM_CANARY=RUN_VALIDATION_ONLY' \
    '  CONFIRM_PUBLICATION=RUN_9200' \
    '  CONFIRM_REPORT=BUILD_REPORT' \
    '' \
    'Worker defaults:' \
    '  VERIFY_WORKERS=4' \
    '  CBWCE_EVALUATION_WORKERS=4'
}

require_token() {
  local name="$1"
  local expected="$2"
  local actual="${!name:-}"
  if [[ "$actual" != "$expected" ]]; then
    echo "BLOCKED: set $name=$expected to authorize this mode." >&2
    return 2
  fi
}

scenario_contract() {
  cd "$ROOT"
  python3 - <<'PY'
from experiments.protocol import settings
from experiments.scenarios import definitions
from experiments.demand import original_profiles

protocol = settings()
assert protocol["version"] == 7, "protocol version 7 is required"
for network in protocol["networks"]:
    seen = {row["name"] for row in original_profiles(network)}
    temporal = [row for row in definitions(network, "test")
                if row["family"] == "temporal"]
    assert [row["id"] for row in temporal] == [
        "switch_300", "switch_900", "switch_1200"
    ]
    for scenario, expected_blocks in zip(temporal, (12, 4, 3)):
        assert scenario.get("protocol_version") == 7
        assert scenario.get("temporal_policy") == "seeded_random_seen_profiles"
        profiles = [block["profile"] for block in scenario["blocks"]]
        assert len(profiles) == expected_blocks
        assert set(profiles) <= seen
        assert all(a != b for a, b in zip(profiles, profiles[1:]))
print("PASS protocol-v7 temporal scenario contract")
PY
}

selection_contract() {
  python3 - "$SELECTION" "$RAW_ROOT" <<'PY'
import json
import sys
from pathlib import Path

selection = json.loads(Path(sys.argv[1]).read_text())
assert selection["counts"] == {
    "parents": 8, "offline_wce": 8, "continuations": 40,
    "evaluation_suites": 40, "expected_rollouts": 9200,
}
assert len({x["id"] for x in selection["continuations"]}) == 40
missing = [x["checkpoint_absolute_path"] for x in selection["continuations"]
           if not Path(x["checkpoint_absolute_path"]).is_dir()]
assert not missing, "missing checkpoint(s): " + repr(missing[:3])
assert not Path(sys.argv[2]).exists(), "formal output root already exists"
print("PASS selection and output-root contract")
PY
}

status() {
  cd "$ROOT"
  python3 - "$GATE" "$SELECTION" "$RAW_ROOT" "$REPORT_ROOT" "$EVALUATION_WORKERS" <<'PY'
import json
import shutil
import sys
from pathlib import Path

gate_arg = sys.argv[1]
gate = Path(gate_arg) if gate_arg else None
selection, raw_root, report_root = map(Path, sys.argv[2:5])
evaluation_workers = int(sys.argv[5])
protocol = json.loads(Path("config/revised/protocol.json").read_text())
selected = json.loads(selection.read_text()) if selection.exists() else {}
gate_data = json.loads(gate.read_text()) if gate and gate.is_file() else {}
print(json.dumps({
    "protocol_version": protocol.get("version"),
    "temporal_required": "seeded_random_seen_profiles",
    "gate": str(gate) if gate else None,
    "gate_exists": bool(gate and gate.is_file()),
    "gate_status": gate_data.get("status"),
    "selected_continuations": len(selected.get("continuations", [])),
    "planned_suites": len(selected.get("evaluation_suites", [])),
    "free_gib": round(shutil.disk_usage(Path.cwd()).free / 1024**3, 2),
    "raw_root_exists": raw_root.exists(),
    "report_root_exists": report_root.exists(),
    "evaluation_workers": evaluation_workers,
}, indent=2))
PY
}

preflight() {
  cd "$ROOT"
  scenario_contract
  selection_contract
  local available_kb
  available_kb=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
  if [[ "$available_kb" -lt 62914560 ]]; then
    echo 'BLOCKED: at least 60 GiB free is required.' >&2
    return 3
  fi
  [[ -n "$GATE" && -f "$GATE" ]] || { echo "BLOCKED: export CBWCE_GATE to a current protocol-v7 gate.json." >&2; return 4; }
  CBWCE_GATE="$GATE" "$LAUNCHER" --preflight
  conda run -n deeprlsc python "$SCHEDULER" \
    --selection "$SELECTION" --gate "$GATE" \
    --output-root "$RAW_ROOT" --workers "$EVALUATION_WORKERS" --dry-run \
    >/dev/null
}

verify_gate() {
  require_token CONFIRM_VERIFICATION RUN_8_PILOTS
  local output="${2:-$ROOT/runs_eval/revised/verification/protocol_v7_$(date -u +%Y%m%dT%H%M%SZ)}"
  cd "$ROOT"
  conda run -n deeprlsc python main.py experiment verify \
    --workers "${VERIFY_WORKERS:-4}" --output "$output"
  printf 'Verification complete. Use:\nexport CBWCE_GATE=%q\n' "$output/gate.json"
}

materialize() {
  require_token CONFIRM_MATERIALIZE BUILD_V7_ARTIFACTS
  cd "$ROOT"
  scenario_contract
  [[ -n "$GATE" && -f "$GATE" ]] || { echo "BLOCKED: export CBWCE_GATE to a current protocol-v7 gate.json." >&2; return 4; }
  conda run -n deeprlsc python -c     "from experiments.runner import require_gate; require_gate(r'''$GATE'''); print('PASS current verification gate')"
  conda run -n deeprlsc python main.py experiment prepare --materialize --network grid
  conda run -n deeprlsc python main.py experiment prepare --materialize --network monaco
}

canary() {
  require_token CONFIRM_CANARY RUN_VALIDATION_ONLY
  local network="${2:?network required}"
  local controller="${3:?controller required}"
  local method="${4:?method required}"
  local checkpoint
  checkpoint=$(python3 - "$SELECTION" "$network" "$controller" "$method" <<'PY'
import json
import sys
from pathlib import Path
selection = json.loads(Path(sys.argv[1]).read_text())
matches = [x["checkpoint_absolute_path"] for x in selection["continuations"]
           if (x["network"], x["controller"], x["method"]) == tuple(sys.argv[2:5])]
assert len(matches) == 1
print(matches[0])
PY
)
  local output="$ROOT/runs_eval/revised/validation_canary_v7/$(date -u +%Y%m%dT%H%M%SZ)/$network/$controller/$method"
  cd "$ROOT"
  conda run -n deeprlsc python main.py experiment evaluate \
    --network "$network" --controller "$controller" --seed 101 \
    --parent "$checkpoint" --gate "$GATE" \
    --suite validation --rollouts 10 --no-visualization \
    --output "$output"
}

execute_publication() {
  require_token CONFIRM_PUBLICATION RUN_9200
  preflight
  conda run -n deeprlsc python "$SCHEDULER" \
    --selection "$SELECTION" --gate "$GATE" \
    --output-root "$RAW_ROOT" --workers "$EVALUATION_WORKERS" --execute
}

build_report() {
  require_token CONFIRM_REPORT BUILD_REPORT
  cd "$ROOT"
  [[ -d "$RAW_ROOT" ]] || { echo "BLOCKED: missing $RAW_ROOT" >&2; return 5; }
  [[ ! -e "$REPORT_ROOT" ]] || { echo "BLOCKED: existing $REPORT_ROOT" >&2; return 6; }
  conda run -n deeprlsc python main.py experiment report \
    --input "$RAW_ROOT" --output "$REPORT_ROOT"
  conda run -n deeprlsc python "$AUDITOR" \
    --raw-root "$RAW_ROOT" --report "$REPORT_ROOT"
}

audit_release() {
  cd "$ROOT"
  if [[ -e "$REPORT_ROOT/dashboard.json" ]]; then
    conda run -n deeprlsc python "$AUDITOR" --raw-root "$RAW_ROOT" --report "$REPORT_ROOT"
  else
    conda run -n deeprlsc python "$AUDITOR" --raw-root "$RAW_ROOT"
  fi
}

case "$MODE" in
  --plan) plan ;;
  --status) status ;;
  --preflight) preflight ;;
  --list) "$LAUNCHER" --list ;;
  --verify) verify_gate "$@" ;;
  --materialize) materialize ;;
  --canary) canary "$@" ;;
  --execute) execute_publication ;;
  --report) build_report ;;
  --audit) audit_release ;;
  --dashboard)
    cd "$ROOT"
    conda run -n deeprlsc python main.py experiment dashboard --port 8765
    ;;
  -h|--help) usage ;;
  *) usage; exit 1 ;;
esac
