"""
Recompute per-group horizon averages from saved raw rollout CSV files.

Each raw CSV contains one time column (`t`) and one column per rollout.
For every file, this script computes:
1. the mean across rollout columns at each second
2. the mean of those per-second means across the full horizon

It then writes one summary CSV row per controller and demand group, with both
queue and speed horizon averages.
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, Tuple


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "runs_eval", "signal_controller_benchmark_real")
DEFAULT_OUTPUT_NAME = "horizon_averages_from_raw.csv"
CONTROLLER_ORDER = [
    "ia2c_real",
    "ia2c_retrained",
    "ma2c_real",
    "ma2c_retrained",
    "iqll_real",
    "iqll_retrained",
    "ppo_real",
    "ppo_retrained"

]
RAW_FILE_RE = re.compile(
    r"^(?P<controller>(?:ia2c|ma2c|iqll|ppo)_(?:real|retrained))_"
    r"group(?P<group_index>\d{2})_"
    r"(?P<group_slug>.+)_"
    r"(?P<metric>queue|speed)_raw\.csv$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute horizon averages from saved raw benchmark CSV files."
    )
    parser.add_argument(
        "--benchmark-dir",
        default=DEFAULT_BENCHMARK_DIR,
        help="Directory containing ia2c/ma2c/iqll benchmark subdirectories.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. Defaults to <benchmark-dir>/%s."
            % DEFAULT_OUTPUT_NAME
        ),
    )
    return parser.parse_args()


def load_group_names(benchmark_dir: str) -> Dict[int, str]:
    manifest_path = os.path.join(benchmark_dir, "demand_manifest.csv")
    group_names: Dict[int, str] = {}
    if not os.path.exists(manifest_path):
        return group_names

    with open(manifest_path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            group_names[int(row["group_index"])] = row["group_name"]
    return group_names


def compute_horizon_average(csv_path: str) -> Tuple[float, int, int]:
    with open(csv_path, "r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            raise ValueError("CSV is empty: %s" % csv_path)

        rollout_indices = [index for index, name in enumerate(header) if name != "t"]
        if not rollout_indices:
            raise ValueError("No rollout columns found in %s" % csv_path)

        per_second_means = []
        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) <= max(rollout_indices):
                raise ValueError(
                    "Row %d in %s does not contain all rollout columns"
                    % (row_number, csv_path)
                )
            rollout_values = [float(row[index]) for index in rollout_indices]
            per_second_means.append(sum(rollout_values) / float(len(rollout_values)))

    if not per_second_means:
        raise ValueError("No data rows found in %s" % csv_path)

    horizon_average = sum(per_second_means) / float(len(per_second_means))
    return horizon_average, len(per_second_means), len(rollout_indices)


def infer_group_name(group_names: Dict[int, str], group_index: int, group_slug: str) -> str:
    if group_index in group_names:
        return group_names[group_index]
    return "demand_%s" % group_slug


def collect_horizon_averages(benchmark_dir: str):
    group_names = load_group_names(benchmark_dir)
    summaries = defaultdict(dict)

    for family in ("ia2c", "ma2c", "iqll", "ppo"):
        family_dir = os.path.join(benchmark_dir, family)
        if not os.path.isdir(family_dir):
            continue

        for filename in sorted(os.listdir(family_dir)):
            match = RAW_FILE_RE.match(filename)
            if not match:
                continue

            controller = match.group("controller")
            group_index = int(match.group("group_index"))
            group_slug = match.group("group_slug")
            metric = match.group("metric")
            csv_path = os.path.join(family_dir, filename)
            horizon_average, horizon_seconds, num_rollouts = compute_horizon_average(csv_path)

            key = (controller, group_index)
            summary = summaries[key]
            summary["controller"] = controller
            summary["family"] = controller.split("_", 1)[0]
            summary["variant"] = controller.split("_", 1)[1]
            summary["group_index"] = group_index
            summary["group_name"] = infer_group_name(group_names, group_index, group_slug)
            summary["group_slug"] = group_slug

            existing_horizon = summary.get("horizon_seconds")
            if existing_horizon is not None and existing_horizon != horizon_seconds:
                raise ValueError(
                    "Mismatched horizon length for %s group %02d"
                    % (controller, group_index)
                )
            summary["horizon_seconds"] = horizon_seconds

            existing_rollouts = summary.get("num_rollouts")
            if existing_rollouts is not None and existing_rollouts != num_rollouts:
                raise ValueError(
                    "Mismatched rollout count for %s group %02d"
                    % (controller, group_index)
                )
            summary["num_rollouts"] = num_rollouts

            if metric == "queue":
                summary["queue_horizon_average"] = horizon_average
            else:
                summary["speed_horizon_average"] = horizon_average

    rows = []
    order_lookup = {name: index for index, name in enumerate(CONTROLLER_ORDER)}
    for key in sorted(
        summaries,
        key=lambda item: (
            order_lookup.get(item[0], len(CONTROLLER_ORDER)),
            item[1],
        ),
    ):
        row = summaries[key]
        if "queue_horizon_average" not in row or "speed_horizon_average" not in row:
            raise ValueError(
                "Missing queue/speed pair for %s group %02d"
                % (row["controller"], row["group_index"])
            )
        rows.append(row)
    return rows


def write_summary_csv(output_path: str, rows) -> None:
    fieldnames = [
        "controller",
        "family",
        "variant",
        "group_index",
        "group_name",
        "group_slug",
        "num_rollouts",
        "horizon_seconds",
        "queue_horizon_average",
        "speed_horizon_average",
    ]
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    benchmark_dir = os.path.abspath(args.benchmark_dir)
    output_path = os.path.abspath(
        args.output if args.output else os.path.join(benchmark_dir, DEFAULT_OUTPUT_NAME)
    )

    rows = collect_horizon_averages(benchmark_dir)
    if not rows:
        raise RuntimeError("No raw benchmark CSV files found under %s" % benchmark_dir)
    write_summary_csv(output_path, rows)

    print("Wrote %d rows to %s" % (len(rows), output_path))


if __name__ == "__main__":
    main()
