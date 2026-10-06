#!/usr/bin/env python3
"""Run the forty frozen publication-evaluation suites with a bounded worker pool."""

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.demand import check_artifact
from experiments.protocol import scenario_root, settings
from experiments.runner import require_gate
from experiments.scenarios import definitions


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_artifacts() -> List[Path]:
    protocol = settings()
    paths = []
    for network in protocol["networks"]:
        for split in ("seen", "test"):
            directory = scenario_root(network, split) / "artifacts"
            for scenario in definitions(network, split):
                for arrival_seed in protocol["arrival_seeds"]:
                    paths.append(directory / "{}_{}.json".format(
                        scenario["id"], arrival_seed))
    return paths


def validate_artifacts() -> None:
    missing = [path for path in expected_artifacts() if not path.is_file()]
    if missing:
        raise ValueError(
            "All demand artifacts must be materialized before parallel evaluation; "
            "missing {} (first: {})".format(len(missing), missing[0])
        )
    for path in expected_artifacts():
        check_artifact(read_json(path))


def build_jobs(selection: dict, gate: Path, output_root: Path) -> List[dict]:
    checkpoints = {
        row["id"]: row for row in selection["continuations"]
    }
    jobs = []
    for suite in selection["evaluation_suites"]:
        checkpoint = checkpoints[suite["checkpoint_id"]]
        output = ROOT / suite["output_path"]
        if output_root not in output.parents:
            raise ValueError("Suite output is outside the requested output root")
        command = [
            "conda", "run", "-n", "deeprlsc", "python", "main.py",
            "experiment", "evaluate",
            "--network", suite["network"],
            "--controller", suite["controller"],
            "--seed", "101",
            "--parent", checkpoint["checkpoint_absolute_path"],
            "--gate", str(gate),
            "--suite", "all",
            "--rollouts", "10",
            "--no-visualization",
            "--output", str(output),
        ]
        jobs.append({
            "id": suite["id"],
            "output": output,
            "command": command,
        })
    if len(jobs) != 40 or len({job["id"] for job in jobs}) != 40:
        raise ValueError("Selection must contain forty unique evaluation suites")
    return jobs


def preflight(selection_path: Path, gate: Path, output_root: Path, workers: int) -> List[dict]:
    protocol = settings()
    if protocol["version"] != 7:
        raise ValueError("Protocol v7 is required")
    if not 1 <= workers <= 4:
        raise ValueError("Publication evaluation workers must be in 1..4")
    if (os.cpu_count() or 1) < workers:
        raise ValueError("Requested workers exceed available logical CPUs")
    if shutil.disk_usage(str(ROOT)).free < 60 * 1024 ** 3:
        raise ValueError("At least 60 GiB free disk is required")
    require_gate(str(gate))
    gate_data = read_json(gate)
    if gate_data.get("concurrent_workers") != workers:
        raise ValueError(
            "Gate concurrent_workers={} does not match evaluation workers={}".format(
                gate_data.get("concurrent_workers"), workers
            )
        )

    selection = read_json(selection_path)
    if selection.get("implemented_protocol_version") != 7:
        raise ValueError("Selection registry must be regenerated under protocol v7")
    if selection.get("required_evaluation_protocol_version") != 7:
        raise ValueError("Selection registry protocol requirement mismatch")
    if selection.get("counts", {}).get("expected_rollouts") != 9200:
        raise ValueError("Selection registry must expect 9,200 rollouts")
    for checkpoint in selection.get("continuations", []):
        if not Path(checkpoint["checkpoint_absolute_path"]).is_dir():
            raise ValueError("Missing checkpoint: " + checkpoint["checkpoint_absolute_path"])

    if output_root.exists():
        raise ValueError("Formal evaluation output root already exists: " + str(output_root))
    validate_artifacts()
    return build_jobs(selection, gate, output_root)


def terminate_all(running: Dict[str, dict]) -> None:
    for state in running.values():
        process = state["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def run(jobs: List[dict], output_root: Path, workers: int) -> int:
    output_root.mkdir(parents=True, exist_ok=False)
    log_root = output_root / "_scheduler_logs"
    log_root.mkdir()
    plan_path = output_root / "_scheduler_plan.json"
    plan_path.write_text(
        json.dumps({
            "workers": workers,
            "suites": [
                {
                    "id": job["id"],
                    "output": str(job["output"].relative_to(ROOT)),
                    "command": job["command"],
                }
                for job in jobs
            ],
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    environment = dict(os.environ)
    environment.update({
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "PYTHONWARNINGS": "ignore",
        "CBWCE_CONCURRENT_WORKERS": str(workers),
    })

    pending = deque(jobs)
    running: Dict[str, dict] = {}
    results = []
    stop_requested = False
    launch_blocked = False

    def request_stop(*_args):
        nonlocal stop_requested
        stop_requested = True
        terminate_all(running)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    while pending or running:
        while pending and len(running) < workers and not stop_requested and not launch_blocked:
            job = pending.popleft()
            log_path = log_root / (job["id"] + ".log")
            stream = log_path.open("x")
            process = subprocess.Popen(
                job["command"],
                cwd=str(ROOT),
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[job["id"]] = {
                "job": job,
                "process": process,
                "stream": stream,
                "log": log_path,
                "started": time.time(),
            }
            print("START {} active={}/{}".format(job["id"], len(running), workers),
                  flush=True)

        completed = []
        for job_id, state in running.items():
            returncode = state["process"].poll()
            if returncode is None:
                continue
            state["stream"].close()
            elapsed = time.time() - state["started"]
            result = {
                "id": job_id,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
                "log": str(state["log"].relative_to(ROOT)),
            }
            results.append(result)
            completed.append(job_id)
            print("DONE {} exit={} elapsed={:.1f}s".format(
                job_id, returncode, elapsed), flush=True)
            if returncode != 0:
                launch_blocked = True

        for job_id in completed:
            del running[job_id]

        if stop_requested and not running:
            break
        if launch_blocked and not running:
            break
        if running:
            time.sleep(2)

    result_path = output_root / "_scheduler_result.json"
    result_path.write_text(
        json.dumps({
            "workers": workers,
            "completed_suites": sum(x["returncode"] == 0 for x in results),
            "failed_suites": sum(x["returncode"] != 0 for x in results),
            "not_started_suites": len(pending),
            "interrupted": stop_requested,
            "results": results,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    if stop_requested or launch_blocked or pending:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection",
        default=str(ROOT / "runs_eval/revised/selections/final_evaluation_seed101.json"),
    )
    parser.add_argument("--gate", required=True)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "runs_eval/revised/publication_seed101_v1"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("CBWCE_EVALUATION_WORKERS", "4")),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    selection_path = Path(args.selection).resolve()
    gate = Path(args.gate).resolve()
    output_root = Path(args.output_root).resolve()
    jobs = preflight(selection_path, gate, output_root, args.workers)

    if args.dry_run:
        for job in jobs:
            print("{}\t{}".format(
                job["id"],
                " ".join(shlex.quote(part) for part in job["command"]),
            ))
        return 0
    return run(jobs, output_root, args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
