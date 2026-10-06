#!/usr/bin/env python3
"""Generate the immutable workbook registry, schemas, tables, and command catalogue.

This generator is read-only with respect to experiments.  It discovers only completed,
budget-valid checkpoints and writes documentation artifacts; it never launches SUMO.
"""
from __future__ import print_function

import datetime
import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"
SCHEMAS = HERE / "schemas"
EXAMPLES = HERE / "examples"
SELECTION_PATH = ROOT / "runs_eval/revised/selections/final_evaluation_seed101.json"
BASE_SELECTION = ROOT / "runs_eval/revised/selections/publication_seed101.json"
PROTOCOL_PATH = ROOT / "config/revised/protocol.json"
NETWORKS = ("grid", "monaco")
CONTROLLERS = ("ia2c", "ma2c", "iqll", "ppo")
METHODS = ("baseline", "random_group", "domain_randomization", "fixed_wce", "online_wce")
ARRIVAL_SEEDS = list(range(51001, 51011))
SUMO_SEEDS = list(range(61001, 61011))
TEST_SCENARIOS = (
    "redistribution_0.25", "redistribution_0.5", "redistribution_0.75",
    "mixture_1", "mixture_2", "mixture_3",
    "switch_300", "switch_900", "switch_1200",
    "peak_1.1", "peak_1.25", "peak_1.5",
)
PROFILE_ORDER = {
    "grid": ("Center_to_Periphery", "E_to_W", "NE_to_SW", "NW_to_SE", "N_to_S",
             "Periphery_to_Center", "SE_to_NW", "SW_to_NE", "S_to_N", "Uniform", "W_to_E"),
    "monaco": ("N_to_S", "S_to_N", "W_to_E", "E_to_W", "NW_to_SE", "SE_to_NW",
               "SW_to_NE", "NE_to_SW", "Periphery_to_Center", "Center_to_Periphery", "Uniform"),
}
MIXTURE_SEEDS = (41004, 41005, 41006)


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path):
    path = Path(path)
    digest = hashlib.sha256()
    files = [p for p in path.rglob("*") if p.is_file()]
    for item in sorted(files):
        rel = str(item.relative_to(path)).replace(os.sep, "/")
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(sha256_file(item).encode("ascii") + b"\0")
    return digest.hexdigest(), len(files)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT)).replace(os.sep, "/")


def find_final(network, controller, method):
    if method == "baseline":
        pattern = ROOT / "runs/revised" / network / controller / "seed_101" / method / "publication_*" / "result.json"
    else:
        base = "output_coevolution" if network == "grid" else "output_coevolution_real"
        pattern = ROOT / base / "revised" / network / controller / "seed_101" / method / "publication_*" / "result.json"
    valid = []
    for filename in sorted(glob.glob(str(pattern))):
        result = read_json(filename)
        checkpoint = Path(result.get("checkpoint", ""))
        if (result.get("status") == "complete" and result.get("learning_steps") == 2320000
                and result.get("stage_simulation_steps") == 1320000 and checkpoint.is_dir()):
            valid.append(Path(filename))
    if len(valid) != 1:
        raise RuntimeError("Expected one valid final checkpoint for {}/{}/{}, found {}: {}".format(
            network, controller, method, len(valid), valid))
    return valid[0]


def entry(kind, result_path, network, controller, method):
    result_path = Path(result_path).resolve()
    result = read_json(result_path)
    manifest_path = result_path.parent / "manifest.json"
    manifest = read_json(manifest_path)
    checkpoint = Path(result["checkpoint"]).resolve()
    checkpoint_hash, checkpoint_files = sha256_tree(checkpoint)
    runtime = manifest.get("runtime", {})
    compatibility = "known_baseline_runtime_mismatch" if method == "baseline" and kind == "continuation" else "reference_runtime_group"
    return {
        "id": "{}-{}-{}-{}".format(kind, network, controller, method),
        "kind": kind,
        "network": network,
        "controller": controller,
        "method": method,
        "training_seed": 101,
        "status": result.get("status"),
        "learning_steps": result.get("learning_steps"),
        "stage_simulation_steps": result.get("stage_simulation_steps"),
        "result_path": relative(result_path),
        "manifest_path": relative(manifest_path),
        "checkpoint_path": relative(checkpoint),
        "checkpoint_absolute_path": str(checkpoint),
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_file_count": checkpoint_files,
        "manifest_hash": result.get("manifest_hash"),
        "manifest_file_sha256": sha256_file(manifest_path),
        "source_commit": manifest.get("source_commit"),
        "experiment_env_source_hash": manifest.get("source_hashes", {}).get("envs/experiment_env.py"),
        "runtime": {
            "python": runtime.get("python"),
            "tensorflow": runtime.get("tensorflow"),
            "numpy": runtime.get("numpy"),
            "platform": runtime.get("platform"),
            "logical_cpus": runtime.get("logical_cpus"),
            "tensorflow_threads": runtime.get("tensorflow_threads"),
            "blas_threads": runtime.get("blas_threads"),
            "omp_threads": runtime.get("omp_threads"),
            "concurrent_workers": runtime.get("concurrent_workers"),
        },
        "runtime_compatibility": compatibility,
    }


def build_selection():
    base = read_json(BASE_SELECTION)
    upstream = {(r["network"], r["controller"]): r for r in base["runs"]}
    parents, wces, finals, suites = [], [], [], []
    for network in NETWORKS:
        for controller in CONTROLLERS:
            selected = upstream[(network, controller)]
            parents.append(entry("parent", ROOT / selected["parent_result"], network, controller, "parent"))
            wces.append(entry("offline_wce", ROOT / selected["wce_result"], network, controller, "offline_wce"))
            for method in METHODS:
                final = entry("continuation", find_final(network, controller, method), network, controller, method)
                finals.append(final)
                output = "runs_eval/revised/publication_seed101_v1/{}/{}/{}".format(network, controller, method)
                suites.append({
                    "id": "evaluation-{}-{}-{}".format(network, controller, method),
                    "network": network,
                    "controller": controller,
                    "method": method,
                    "training_seed": 101,
                    "checkpoint_id": final["id"],
                    "checkpoint_hash": final["checkpoint_hash"],
                    "checkpoint_path": final["checkpoint_path"],
                    "output_path": output,
                    "status": "planned",
                    "expected_rollouts": 230,
                })
    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    selection = {
        "$schema": "../../../docs/evaluation_workbook/schemas/selection.schema.json",
        "schema_version": 1,
        "selection_id": "final_evaluation_seed101",
        "generated_at": now,
        "repository_root": str(ROOT),
        "training_seed": 101,
        "implemented_protocol_version": read_json(PROTOCOL_PATH)["version"],
        "required_evaluation_protocol_version": 7,
        "protocol_file": relative(PROTOCOL_PATH),
        "protocol_file_sha256": sha256_file(PROTOCOL_PATH),
        "formal_evaluation_status": {"accepted": 0, "expected": 9200, "publication_complete": False},
        "counts": {"parents": 8, "offline_wce": 8, "continuations": 40, "evaluation_suites": 40,
                   "expected_rollouts": 9200},
        "known_limitations": [
            "Baseline continuation runtimes and experiment_env source hashes differ from most other methods.",
            "Protocol v7 artifacts must be materialized and audited before execution.",
            "Formal publication evaluation has not started; peak_validation records are pilots only.",
        ],
        "parents": parents,
        "offline_wce": wces,
        "continuations": finals,
        "evaluation_suites": suites,
    }
    write_json(SELECTION_PATH, selection)
    return selection


def tex_escape(value):
    value = str(value)
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                    "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in value)


def build_protocol_tex():
    text = r"""\begin{center}
\begin{tabular}{@{}ll@{}}
\toprule
Item / 项目 & Frozen value / 冻结值 \\
\midrule
Networks & Grid, Monaco \\
Controllers & IA2C, MA2C, IQLL, PPO \\
Methods & baseline, random group, domain randomization, fixed WCE, online WCE \\
Training seed & 101 \\
Final controller steps & 2,320,000 (1,000,000 parent + 1,320,000 continuation) \\
Evaluation horizon & 3,600 s; empty start; no warm-up; no drain \\
SUMO / control step & 1 s / 5 s (2 s yellow + 3 s green) \\
Scenarios & 11 seen + 12 test = 23 \\
Paired seeds & arrival 51001--51010 / SUMO 61001--61010 \\
Formal rollouts & 9,200 (current accepted: 0) \\
Implemented / required protocol & v7 / v7 \\
\bottomrule
\end{tabular}
\end{center}
"""
    (GENERATED / "protocol_snapshot.tex").write_text(text)


def build_mixture_tex():
    weights = [np.random.RandomState(seed).dirichlet(np.ones(11)).tolist() for seed in MIXTURE_SEEDS]
    lines = [r"\section*{Named mixture weights / 按名称列出的 mixture 权重}",
             r"The same numerical vector is interpreted against each network's explicit profile order. Values are generated from NumPy RandomState with seeds 41004--41006."]
    for network in NETWORKS:
        lines.extend([r"\subsection*{" + network.title() + "}", r"\begin{longtable}{@{}lrrr@{}}", r"\toprule",
                      r"Profile & Mixture 1 & Mixture 2 & Mixture 3 \\", r"\midrule"])
        for index, name in enumerate(PROFILE_ORDER[network]):
            lines.append("{} & {:.9f} & {:.9f} & {:.9f} \\\\".format(
                tex_escape(name), weights[0][index], weights[1][index], weights[2][index]))
        lines.extend([r"\bottomrule", r"\end{longtable}"])
    (GENERATED / "mixture_weights.tex").write_text("\n".join(lines) + "\n")


def artifact_directory(network):
    base = "data_traffic/revised" if network == "grid" else "real_net_subnet/demand_groups/revised"
    return ROOT / base / "protocol_v7/test/artifacts"


def build_scenario_counts_tex():
    lines = [r"\section*{Materialized test vehicle counts / 已物化测试车辆数}",
             r"Protocol-v7 artifact counts remain pending until the versioned demand suite is materialized; no protocol-v6 artifact is included here.",
             r"\begin{longtable}{@{}llrrr@{}}", r"\toprule",
             r"Network & Scenario & Mean & Min & Max \\", r"\midrule"]
    for network in NETWORKS:
        directory = artifact_directory(network)
        for scenario in TEST_SCENARIOS:
            counts = []
            for seed in ARRIVAL_SEEDS:
                path = directory / ("{}_{}.json".format(scenario, seed))
                if path.exists():
                    counts.append(len(read_json(path)["vehicles"]))
            if len(counts) == 10:
                lines.append("{} & {} & {:.1f} & {} & {} \\\\".format(
                    network.title(), tex_escape(scenario), float(np.mean(counts)), min(counts), max(counts)))
            else:
                lines.append("{} & {} & pending & -- & -- \\\\".format(network.title(), tex_escape(scenario)))
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    (GENERATED / "scenario_counts.tex").write_text("\n".join(lines) + "\n")


def build_checkpoint_tex(selection):
    lines = [r"\scriptsize", r"\begin{longtable}{@{}llllp{0.18\linewidth}p{0.43\linewidth}@{}}", r"\toprule",
             r"No. & Network & Controller & Method & Runtime & Exact checkpoint (repo-relative) \\", r"\midrule", r"\endfirsthead",
             r"\toprule No. & Network & Controller & Method & Runtime & Exact checkpoint (repo-relative) \\", r"\midrule", r"\endhead"]
    for number, item in enumerate(selection["continuations"], 1):
        runtime = "Py {}/TF {}".format(item["runtime"].get("python"), item["runtime"].get("tensorflow"))
        lines.append("{} & {} & {} & {} & {} & \\path{{{}}} \\\\".format(
            number, item["network"], item["controller"], tex_escape(item["method"]), tex_escape(runtime), item["checkpoint_path"]))
    lines.extend([r"\bottomrule", r"\end{longtable}",
                  r"Full absolute paths, aggregate checkpoint hashes, manifest hashes, source hashes, and planned suite outputs are authoritative in \path{runs_eval/revised/selections/final_evaluation_seed101.json}."])
    (GENERATED / "checkpoint_manifest.tex").write_text("\n".join(lines) + "\n")


def shell_quote(value):
    return "'" + str(value).replace("'", "'\\''") + "'"


def build_commands(selection):
    script = GENERATED / "evaluate_publication_seed101.sh"
    lines = [
        "#!/usr/bin/env bash", "set -euo pipefail", "",
        "ROOT=/home/sdc_joran/Journal/deeprl_signal_control",
        "GATE=${CBWCE_GATE:-}",
        "MODE=${1:-}", "",
        "usage() {",
        "  echo 'Usage: evaluate_publication_seed101.sh --preflight | --list | --execute'",
        "  echo 'This script will not execute formal evaluation unless protocol v7 and storage gates pass.'",
        "}", "",
        "preflight() {",
        "  cd \"$ROOT\"",
        "  version=$(python3 -c \"import json; print(json.load(open('config/revised/protocol.json'))['version'])\")",
        "  if [ \"$version\" != 7 ]; then",
        "    echo \"BLOCKED: protocol v7 is required; current version is $version.\" >&2",
        "    return 2",
        "  fi",
        "  if [ -z \"$GATE\" ]; then echo \"BLOCKED: export CBWCE_GATE to a current protocol-v7 gate.json.\" >&2; return 4; fi",
        "  available_kb=$(df -Pk \"$ROOT\" | awk 'NR==2 {print $4}')",
        "  if [ \"$available_kb\" -lt 62914560 ]; then",
        "    echo \"BLOCKED: at least 60 GiB free is required.\" >&2",
        "    return 3",
        "  fi",
        "  conda run -n deeprlsc python -c \"from experiments.runner import require_gate; require_gate(r'''$GATE'''); print('PASS current verification gate')\"",
        "}", "",
        "run_all() {", "  cd \"$ROOT\"",
    ]
    for item in selection["continuations"]:
        output = "runs_eval/revised/publication_seed101_v1/{}/{}/{}".format(
            item["network"], item["controller"], item["method"])
        lines.extend([
            "  if [ -e {} ]; then echo {} >&2; return 4; fi".format(
                shell_quote(str(ROOT / output)), shell_quote("BLOCKED: output already exists: " + output)),
            "  conda run -n deeprlsc python main.py experiment evaluate \\",
            "    --network {} --controller {} --seed 101 \\".format(item["network"], item["controller"]),
            "    --parent {} --gate \"$GATE\" \\".format(shell_quote(item["checkpoint_absolute_path"])),
            "    --suite all --rollouts 10 --no-visualization \\",
            "    --output {}".format(shell_quote(output)),
        ])
    lines.extend(["}", "", "case \"$MODE\" in",
                  "  --preflight) preflight ;;",
                  "  --list) sed -n '/^run_all()/,/^}/p' \"$0\" ;;",
                  "  --execute) preflight; run_all ;;",
                  "  *) usage; exit 1 ;;", "esac", ""])
    script.write_text("\n".join(lines))
    script.chmod(0o755)

    tex = r"""\section*{Command status / 命令状态}
The environment check, evaluation, current report, and dashboard commands exist now. The proposed selection-driven report, validate-report, and export-site interfaces are specifications only and are not represented as current capabilities.

The exact command catalogue is \path{docs/evaluation_workbook/generated/evaluate_publication_seed101.sh}; it retains forty explicit invocations and a sequential fallback. The primary entry point is \path{docs/evaluation_workbook/generated/publication_workflow.sh}. It is safe by default and uses \path{docs/evaluation_workbook/run_publication_parallel.py} to run four suites concurrently only after protocol, gate, storage, selection, artifact, and output-root checks pass. Rollouts remain sequential within each suite, and failures stop new dispatch without automatic retry.

\begin{Verbatim}[fontsize=\small,breaklines=true]
./docs/evaluation_workbook/generated/evaluate_publication_seed101.sh --preflight
./docs/evaluation_workbook/generated/evaluate_publication_seed101.sh --list
./docs/evaluation_workbook/generated/evaluate_publication_seed101.sh --execute
\end{Verbatim}
"""
    (GENERATED / "command_catalog.tex").write_text(tex)


def build_schemas():
    common_hash = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    checkpoint = {
        "type": "object", "additionalProperties": True,
        "required": ["id", "kind", "network", "controller", "method", "training_seed", "status",
                     "result_path", "manifest_path", "checkpoint_path", "checkpoint_absolute_path",
                     "checkpoint_hash", "manifest_hash", "runtime", "runtime_compatibility"],
        "properties": {
            "id": {"type": "string"}, "kind": {"enum": ["parent", "offline_wce", "continuation"]},
            "network": {"enum": list(NETWORKS)}, "controller": {"enum": list(CONTROLLERS)},
            "method": {"type": "string"}, "training_seed": {"const": 101}, "status": {"const": "complete"},
            "learning_steps": {"type": ["integer", "null"]}, "stage_simulation_steps": {"type": ["integer", "null"]},
            "result_path": {"type": "string"}, "manifest_path": {"type": "string"},
            "checkpoint_path": {"type": "string"}, "checkpoint_absolute_path": {"type": "string", "pattern": "^/"},
            "checkpoint_hash": common_hash, "manifest_hash": common_hash, "manifest_file_sha256": common_hash,
            "runtime": {"type": "object"}, "runtime_compatibility": {"type": "string"},
        }}
    selection = {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "selection.schema.json",
        "title": "CB-WCE frozen model selection", "type": "object", "additionalProperties": False,
        "required": ["schema_version", "selection_id", "training_seed", "implemented_protocol_version",
                     "required_evaluation_protocol_version", "formal_evaluation_status", "counts", "parents",
                     "offline_wce", "continuations", "evaluation_suites"],
        "properties": {
            "$schema": {"type": "string"}, "schema_version": {"const": 1}, "selection_id": {"const": "final_evaluation_seed101"},
            "generated_at": {"type": "string"}, "repository_root": {"type": "string"}, "training_seed": {"const": 101},
            "implemented_protocol_version": {"type": "integer"}, "required_evaluation_protocol_version": {"const": 7},
            "protocol_file": {"type": "string"}, "protocol_file_sha256": common_hash,
            "formal_evaluation_status": {"type": "object"}, "counts": {"type": "object"},
            "known_limitations": {"type": "array", "items": {"type": "string"}},
            "parents": {"type": "array", "minItems": 8, "maxItems": 8, "items": checkpoint},
            "offline_wce": {"type": "array", "minItems": 8, "maxItems": 8, "items": checkpoint},
            "continuations": {"type": "array", "minItems": 40, "maxItems": 40, "items": checkpoint},
            "evaluation_suites": {"type": "array", "minItems": 40, "maxItems": 40, "items": {"type": "object"}},
        }}
    rollout_metrics = {
        "mean_total_queue": {"type": "number", "minimum": 0}, "integrated_queue_vehicle_seconds": {"type": "number", "minimum": 0},
        "peak_total_queue": {"type": "number", "minimum": 0}, "p95_total_queue": {"type": ["number", "null"], "minimum": 0},
        "mean_speed_vehicle_time_weighted": {"type": ["number", "null"], "minimum": 0},
        "speed_denominator_vehicle_seconds": {"type": "integer", "minimum": 0},
        "scheduled": {"type": "integer", "minimum": 0}, "inserted": {"type": "integer", "minimum": 0},
        "completed": {"type": "integer", "minimum": 0}, "remaining": {"type": "integer", "minimum": 0},
        "pending": {"type": "integer", "minimum": 0}, "completion_rate": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "mean_completed_travel_time_seconds": {"type": ["number", "null"], "minimum": 0},
        "mean_completed_waiting_time_seconds": {"type": ["number", "null"], "minimum": 0},
        "mean_completed_time_loss_seconds": {"type": ["number", "null"], "minimum": 0},
        "mean_completed_departure_delay_seconds": {"type": ["number", "null"], "minimum": 0},
        "completed_trip_denominator": {"type": "integer", "minimum": 0}, "teleports": {"type": "integer", "minimum": 0},
        "collisions": {"type": "integer", "minimum": 0}, "wall_seconds": {"type": "number", "minimum": 0},
    }
    rollout = {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "rollout.schema.json", "title": "Validated CB-WCE rollout",
        "type": "object", "additionalProperties": False,
        "required": ["schema_version", "rollout_id", "network", "controller", "method", "training_seed", "split", "family",
                     "scenario_id", "rollout_index", "arrival_seed", "sumo_seed", "policy_seed", "protocol_hash", "scenario_hash",
                     "demand_hash", "checkpoint_hash", "manifest_hash", "pilot", "status", "horizon_seconds", "sample_count", "metrics"],
        "properties": {
            "schema_version": {"const": 3}, "rollout_id": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "network": {"enum": list(NETWORKS)}, "controller": {"enum": list(CONTROLLERS)}, "method": {"enum": list(METHODS)},
            "training_seed": {"const": 101}, "split": {"enum": ["seen", "test"]}, "family": {"type": "string"},
            "scenario_id": {"type": "string"}, "rollout_index": {"type": "integer", "minimum": 1, "maximum": 10},
            "generation_seed": {"type": ["integer", "null"]}, "arrival_seed": {"enum": ARRIVAL_SEEDS}, "sumo_seed": {"enum": SUMO_SEEDS},
            "policy_seed": {"type": "integer"}, "protocol_hash": common_hash, "scenario_hash": common_hash, "demand_hash": common_hash,
            "checkpoint_hash": common_hash, "manifest_hash": common_hash, "source_commit": {"type": ["string", "null"]},
            "pilot": {"const": False}, "status": {"const": "complete"}, "horizon_seconds": {"const": 3600}, "sample_count": {"const": 3600},
            "metrics": {"type": "object", "additionalProperties": True,
                        "required": ["mean_total_queue", "integrated_queue_vehicle_seconds", "peak_total_queue", "scheduled", "inserted",
                                     "completed", "remaining", "pending", "completed_trip_denominator", "teleports", "collisions", "wall_seconds"],
                        "properties": rollout_metrics},
        }}
    release = {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "release_manifest.schema.json", "title": "Validated publication release",
        "type": "object", "additionalProperties": False,
        "required": ["schema_version", "release_id", "generated_at", "protocol_version", "protocol_hash", "source_commit", "counts",
                     "publication_complete", "selection_hash", "selected_checkpoint_hashes", "outputs"],
        "properties": {
            "schema_version": {"const": 3}, "release_id": {"type": "string"}, "generated_at": {"type": "string"},
            "protocol_version": {"const": 7}, "protocol_hash": common_hash, "source_commit": {"type": "string"},
            "counts": {"type": "object", "required": ["expected", "accepted", "rejected", "missing", "duplicate", "extra"],
                       "properties": {key: {"type": "integer", "minimum": 0} for key in ("expected", "accepted", "rejected", "missing", "duplicate", "extra")}},
            "publication_complete": {"type": "boolean"}, "strict_runtime_comparability": {"type": "boolean"},
            "selection_hash": common_hash, "selected_checkpoint_hashes": {"type": "array", "minItems": 40, "maxItems": 40, "items": common_hash},
            "outputs": {"type": "array", "items": {"type": "object", "required": ["path", "sha256"],
                                                             "properties": {"path": {"type": "string"}, "sha256": common_hash}}},
        }}
    site = {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "site_publication.schema.json", "title": "Sanitized static-site publication manifest",
        "type": "object", "additionalProperties": False,
        "required": ["schema_version", "release_id", "generated_at", "publication_complete", "counts", "metric_registry", "shards", "checksums"],
        "properties": {
            "schema_version": {"const": 3}, "release_id": {"type": "string"}, "generated_at": {"type": "string"},
            "publication_complete": {"const": True}, "counts": {"type": "object"},
            "metric_registry": {"type": "array", "items": {"type": "object", "required": ["id", "unit", "direction", "denominator", "null_semantics"]}},
            "shards": {"type": "array", "items": {"type": "object", "required": ["kind", "path", "records", "sha256"],
                                                     "properties": {"kind": {"enum": ["overview", "rollouts", "heatmap", "curve"]},
                                                                    "path": {"type": "string", "pattern": "^(?!/)(?!.*\\.\\.).+$"},
                                                                    "records": {"type": "integer", "minimum": 0}, "sha256": common_hash}}},
            "checksums": {"type": "string", "pattern": "^(?!/)(?!.*\\.\\.).+$"},
        }}
    for name, schema in (("selection.schema.json", selection), ("rollout.schema.json", rollout),
                         ("release_manifest.schema.json", release), ("site_publication.schema.json", site)):
        write_json(SCHEMAS / name, schema)


def build_examples():
    digest = "0" * 64
    metrics = {
        "mean_total_queue": 120.5, "integrated_queue_vehicle_seconds": 433800.0,
        "peak_total_queue": 250.0, "p95_total_queue": 210.0,
        "mean_speed_vehicle_time_weighted": 8.4, "speed_denominator_vehicle_seconds": 540000,
        "scheduled": 3000, "inserted": 2990, "completed": 2700, "remaining": 290,
        "pending": 10, "completion_rate": 2700.0 / 2990.0,
        "mean_completed_travel_time_seconds": 410.0, "mean_completed_waiting_time_seconds": 95.0,
        "mean_completed_time_loss_seconds": 140.0, "mean_completed_departure_delay_seconds": 2.0,
        "completed_trip_denominator": 2700, "teleports": 0, "collisions": 0, "wall_seconds": 180.0,
    }
    rollout = {
        "schema_version": 3, "rollout_id": digest, "network": "grid", "controller": "ia2c",
        "method": "online_wce", "training_seed": 101, "split": "test", "family": "peak",
        "scenario_id": "peak_1.5", "rollout_index": 1, "generation_seed": 41012,
        "arrival_seed": 51001, "sumo_seed": 61001, "policy_seed": 123456,
        "protocol_hash": digest, "scenario_hash": digest, "demand_hash": digest,
        "checkpoint_hash": digest, "manifest_hash": digest, "source_commit": None,
        "pilot": False, "status": "complete", "horizon_seconds": 3600, "sample_count": 3600,
        "metrics": metrics,
    }
    release = {
        "schema_version": 3, "release_id": "publication_seed101_v1", "generated_at": "2026-09-29T12:00:00+00:00",
        "protocol_version": 7, "protocol_hash": digest, "source_commit": "8fd0e625b0535502fada77020e6db418939e2225",
        "counts": {"expected": 9200, "accepted": 9200, "rejected": 0, "missing": 0, "duplicate": 0, "extra": 0},
        "publication_complete": True, "strict_runtime_comparability": False, "selection_hash": digest,
        "selected_checkpoint_hashes": [digest] * 40,
        "outputs": [{"path": "overview.json", "sha256": digest}],
    }
    site = {
        "schema_version": 3, "release_id": "publication_seed101_v1", "generated_at": "2026-09-29T12:00:00+00:00",
        "publication_complete": True, "counts": {"expected": 9200, "accepted": 9200, "rejected": 0},
        "metric_registry": [{"id": "mean_total_queue", "unit": "vehicles", "direction": "lower",
                             "denominator": "3600 per-second samples", "null_semantics": "never null for complete rollout"}],
        "shards": [{"kind": "overview", "path": "overview.json", "records": 1, "sha256": digest}],
        "checksums": "checksums.json",
    }
    write_json(EXAMPLES / "rollout.example.json", rollout)
    write_json(EXAMPLES / "release_manifest.example.json", release)
    write_json(EXAMPLES / "site_publication.example.json", site)


def main():
    GENERATED.mkdir(parents=True, exist_ok=True)
    SCHEMAS.mkdir(parents=True, exist_ok=True)
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    selection = build_selection()
    build_protocol_tex()
    build_mixture_tex()
    build_scenario_counts_tex()
    build_checkpoint_tex(selection)
    build_commands(selection)
    build_schemas()
    build_examples()
    print("generated parents={} wce={} continuations={} suites={}".format(
        len(selection["parents"]), len(selection["offline_wce"]),
        len(selection["continuations"]), len(selection["evaluation_suites"])))


if __name__ == "__main__":
    main()
