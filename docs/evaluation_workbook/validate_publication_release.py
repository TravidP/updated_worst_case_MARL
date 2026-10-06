#!/usr/bin/env python3
"""Read-only completeness audit for a CB-WCE publication evaluation release."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.protocol import settings
from experiments.reporting import collect
from experiments.scenarios import definitions


def audit(raw_root: Path, report_path: Path | None) -> dict:
    protocol = settings()
    networks = tuple(protocol["networks"])
    controllers = tuple(protocol["controllers"])
    methods = tuple(protocol["methods"])
    arrivals = tuple(protocol["arrival_seeds"])
    seed_map = dict(zip(arrivals, protocol["sumo_seeds"]))
    training_seed = protocol["training_seeds"][0]
    scenarios = {
        (network, split): tuple(x["id"] for x in definitions(network, split))
        for network in networks for split in ("seen", "test")
    }

    rows, rejected = collect([raw_root])
    errors = [
        "Rejected rollout: {}: {}".format(x["path"], x["reason"])
        for x in rejected
    ]
    actual = set()
    combo_counts = Counter()
    scenario_counts = Counter()

    def error(message: str) -> None:
        if len(errors) < 200:
            errors.append(message)

    for row in rows:
        key = (
            row["network"], row["controller"], row["method"], row["seed"],
            row["split"], row["scenario"], row["arrival_seed"], row["pilot"],
        )
        if key in actual:
            error("Duplicate accepted key: {!r}".format(key))
        actual.add(key)
        combo_counts[key[:3]] += 1
        scenario_counts[(row["network"], row["controller"], row["method"],
                         row["split"], row["scenario"])] += 1
        label = "/".join(str(x) for x in key[:3] + key[4:6])

        if row.get("pilot") is not False:
            error(label + ": pilot must be false")
        if row.get("seed") != training_seed:
            error(label + ": unexpected training seed")
        if row.get("status") != "complete":
            error(label + ": status is not complete")
        if row.get("horizon") != 3600 or row.get("sample_count") != 3600:
            error(label + ": horizon/sample_count must both be 3600")
        if row.get("effective_sumo_seed") != seed_map.get(row.get("arrival_seed")):
            error(label + ": arrival/SUMO seed pairing mismatch")

        mean_queue = row.get("mean_queue")
        integrated = row.get("integrated_queue")
        peak = row.get("peak_queue")
        if not all(isinstance(x, (int, float)) and math.isfinite(x)
                   for x in (mean_queue, integrated, peak)):
            error(label + ": invalid queue metric")
        else:
            tolerance = max(1e-6, abs(integrated) * 1e-9)
            if abs(integrated - 3600.0 * mean_queue) > tolerance:
                error(label + ": integrated_queue != 3600 * mean_queue")
            if peak < mean_queue:
                error(label + ": peak_queue < mean_queue")

        vehicle_fields = tuple(row.get(x) for x in
                               ("scheduled", "inserted", "pending", "completed",
                                "completed_trip_denominator"))
        if not all(isinstance(x, int) for x in vehicle_fields):
            error(label + ": vehicle accounting fields are missing")
        else:
            scheduled, inserted, pending, completed, denominator = vehicle_fields
            if scheduled != inserted + pending:
                error(label + ": scheduled != inserted + pending")
            if completed > inserted:
                error(label + ": completed > inserted")
            if denominator != completed:
                error(label + ": completed-trip denominator mismatch")

    expected = {
        (network, controller, method, training_seed, split, scenario, arrival, False)
        for network in networks
        for controller in controllers
        for method in methods
        for split in ("seen", "test")
        for scenario in scenarios[(network, split)]
        for arrival in arrivals
    }
    missing = expected - actual
    extra = actual - expected
    if missing:
        error("Missing keys: {} (first: {!r})".format(len(missing), sorted(missing)[0]))
    if extra:
        error("Unexpected keys: {} (first: {!r})".format(len(extra), sorted(extra)[0]))

    for network in networks:
        for controller in controllers:
            for method in methods:
                if combo_counts[(network, controller, method)] != 230:
                    error("{}/{}/{} does not have 230 rollouts".format(
                        network, controller, method))
                for split in ("seen", "test"):
                    for scenario in scenarios[(network, split)]:
                        key = (network, controller, method, split, scenario)
                        if scenario_counts[key] != 10:
                            error("/".join(key) + " does not have 10 rollouts")

    report_checks = None
    if report_path:
        dashboard = report_path / "dashboard.json" if report_path.is_dir() else report_path
        if not dashboard.exists():
            error("Report dashboard does not exist: " + str(dashboard))
        else:
            report = json.loads(dashboard.read_text(encoding="utf-8"))
            report_checks = {
                "path": str(dashboard),
                "rollouts": len(report.get("rollouts", [])),
                "rejected_count": report.get("rejected_count"),
                "publication_complete": report.get("publication_complete"),
            }
            if report_checks != {
                "path": str(dashboard),
                "rollouts": 9200,
                "rejected_count": 0,
                "publication_complete": True,
            }:
                error("Report completeness fields do not equal 9200/0/true")

    return {
        "raw_root": str(raw_root),
        "expected": len(expected),
        "accepted": len(rows),
        "rejected": len(rejected),
        "missing": len(missing),
        "extra": len(extra),
        "duplicate": max(0, len(rows) - len(actual)),
        "report": report_checks,
        "publication_complete": not errors and len(rows) == 9200,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default=str(ROOT / "runs_eval/revised/publication_seed101_v1"),
    )
    parser.add_argument("--report")
    args = parser.parse_args()
    result = audit(
        Path(args.raw_root).resolve(),
        Path(args.report).resolve() if args.report else None,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["publication_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
