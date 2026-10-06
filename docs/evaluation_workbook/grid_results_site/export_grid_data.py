#!/usr/bin/env python3
"""Export the completed Grid publication rollouts into a read-only web dataset.

This script never launches SUMO or evaluation jobs. It validates and reads existing
rollout_summary.json, manifest.json, artifact JSON, and rollout.npz files only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "runs_eval/revised/publication_seed101_v1/grid"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "dist/data"
CONTROLLERS = ("ia2c", "ma2c", "iqll", "ppo")
METHODS = ("baseline", "random_group", "domain_randomization", "fixed_wce", "online_wce")
ARRIVAL_SEEDS = tuple(range(51001, 51011))
EXPECTED_STEPS = 3600

FAMILY_META = {
    "seen": {"zh": "训练分布内需求", "en": "Seen demand profiles"},
    "redistribution": {"zh": "OD 重分配", "en": "OD redistribution"},
    "mixture": {"zh": "混合分布", "en": "Mixture distribution"},
    "temporal": {"zh": "时序切换", "en": "Temporal switching"},
    "peak": {"zh": "峰值缩放", "en": "Peak scaling"},
}

PROFILE_LABELS = {
    "N_to_S": ("北向南", "North to south"),
    "S_to_N": ("南向北", "South to north"),
    "W_to_E": ("西向东", "West to east"),
    "E_to_W": ("东向西", "East to west"),
    "NW_to_SE": ("西北向东南", "Northwest to southeast"),
    "SE_to_NW": ("东南向西北", "Southeast to northwest"),
    "SW_to_NE": ("西南向东北", "Southwest to northeast"),
    "NE_to_SW": ("东北向西南", "Northeast to southwest"),
    "Periphery_to_Center": ("外围到中心", "Periphery to center"),
    "Center_to_Periphery": ("中心到外围", "Center to periphery"),
    "Uniform": ("均匀需求", "Uniform demand"),
}

METHOD_META = {
    "baseline": {"zh": "基线", "en": "Baseline", "color": "#244a75"},
    "random_group": {"zh": "随机分组", "en": "Random grouping", "color": "#d6781f"},
    "domain_randomization": {"zh": "域随机化", "en": "Domain randomization", "color": "#2d876e"},
    "fixed_wce": {"zh": "固定 WCE", "en": "Fixed WCE", "color": "#985da5"},
    "online_wce": {"zh": "在线 WCE", "en": "Online WCE", "color": "#c3423f"},
}

METRICS = (
    ("mean_queue", "mean_queue", "vehicles", "lower", "平均排队", "Mean queue", "每秒全网受监控车道排队车辆总数的平均值。", "Mean of the network-wide queued-vehicle total at each second."),
    ("peak_queue", "peak_queue", "vehicles", "lower", "峰值排队", "Peak queue", "10 次运行内的全网排队峰值。", "Peak network-wide queued-vehicle total across a run."),
    ("integrated_queue", "integrated_queue", "vehicle_seconds", "lower", "累计排队", "Integrated queue", "逐秒全网排队总量的时间积分。", "Time integral of the second-by-second network queue."),
    ("mean_speed", "mean_speed", "m_per_s", "higher", "平均速度", "Mean speed", "仿真期间车辆平均速度。", "Average vehicle speed during the simulation."),
    ("completed", "completed", "vehicles", "higher", "已完成车辆", "Completed vehicles", "在仿真结束前完成行程的车辆数。", "Vehicles that completed their trips before simulation end."),
    ("completion_rate", None, "percent", "higher", "完成率", "Completion rate", "已完成车辆数除以已插入车辆数。", "Completed vehicles divided by inserted vehicles."),
    ("pending", "pending", "vehicles", "lower", "待出发车辆", "Pending vehicles", "仿真结束时尚未插入路网的计划车辆。", "Scheduled vehicles not yet inserted at simulation end."),
    ("remaining", "remaining", "vehicles", "lower", "在网车辆", "Remaining vehicles", "仿真结束时仍在路网中的车辆。", "Vehicles still present in the network at simulation end."),
    ("mean_completed_departure_delay", "mean_completed_departure_delay", "seconds", "lower", "平均出发延误", "Mean departure delay", "已完成车辆的平均实际出发延迟。", "Mean actual departure delay among completed vehicles."),
    ("mean_completed_time_loss", "mean_completed_time_loss", "seconds", "lower", "平均时间损失", "Mean time loss", "已完成车辆相对自由流的平均时间损失。", "Mean time loss relative to free flow among completed vehicles."),
    ("mean_completed_travel_time", "mean_completed_travel_time", "seconds", "lower", "平均旅行时间", "Mean travel time", "已完成车辆的平均旅行时间。", "Mean travel time among completed vehicles."),
    ("mean_completed_waiting_time", "mean_completed_waiting_time", "seconds", "lower", "平均等待时间", "Mean waiting time", "已完成车辆的平均累计等待时间。", "Mean accumulated waiting time among completed vehicles."),
    ("teleports", "teleports", "count", "lower", "传送次数", "Teleports", "SUMO 为解除堵塞而传送车辆的次数。", "Number of SUMO vehicle teleports used to resolve blocking."),
    ("collisions", "collisions", "count", "lower", "碰撞次数", "Collisions", "仿真记录的碰撞事件数。", "Number of collision events recorded by the simulation."),
    ("scheduled", "scheduled", "vehicles", "check", "计划车辆", "Scheduled vehicles", "需求文件中的计划车辆数，仅用于核对。", "Vehicles scheduled by the demand file; verification only."),
    ("inserted", "inserted", "vehicles", "check", "已插入车辆", "Inserted vehicles", "实际插入路网的车辆数，仅用于核对。", "Vehicles actually inserted into the network; verification only."),
)

UNIT_META = {
    "vehicles": {"zh": "车辆数", "en": "vehicles"},
    "vehicle_seconds": {"zh": "车辆·秒", "en": "vehicle·seconds"},
    "m_per_s": {"zh": "米/秒", "en": "m/s"},
    "percent": {"zh": "%", "en": "%"},
    "seconds": {"zh": "秒", "en": "s"},
    "count": {"zh": "次", "en": "count"},
}


def scenario_label(scenario_id: str, family: str) -> tuple[str, str]:
    if scenario_id.startswith("seen_"):
        raw = scenario_id[len("seen_"):]
        return PROFILE_LABELS.get(raw, (raw, raw))
    if scenario_id.startswith("redistribution_"):
        value = scenario_id.split("_", 1)[1]
        return (f"OD 重分配 σ={value}", f"OD redistribution σ={value}")
    if scenario_id.startswith("mixture_"):
        value = scenario_id.split("_", 1)[1]
        return (f"混合场景 {value}", f"Mixture scenario {value}")
    if scenario_id.startswith("switch_"):
        value = scenario_id.split("_", 1)[1]
        return (f"每 {value} 秒快速切换", f"Rapid switching every {value} s")
    if scenario_id.startswith("peak_"):
        value = scenario_id.split("_", 1)[1]
        return (f"峰值需求 ×{value}", f"Peak demand ×{value}")
    return (scenario_id, scenario_id)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite_number(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise AssertionError(f"{label} is not finite: {value!r}")
    return number


def metric_values(summary: dict) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, source, *_ in METRICS:
        if key == "completion_rate":
            inserted = finite_number(summary["inserted"], "inserted")
            completed = finite_number(summary["completed"], "completed")
            values[key] = 100.0 * completed / inserted if inserted else 0.0
        else:
            values[key] = finite_number(summary[source], source)
    return values


def write_csv(path: Path, header: list[str], rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Replace an existing generated data directory.")
    args = parser.parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"Grid result directory does not exist: {input_root}")
    if output_root.exists() and not args.force:
        raise SystemExit(f"Output already exists: {output_root} (pass --force to replace generated data)")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="grid-data-", dir=str(output_root.parent)))

    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    scenario_meta: dict[tuple[str, str], str] = {}
    pairing: dict[tuple[str, str, int], tuple[str, int]] = {}
    policy_pairing: dict[tuple[str, str, str, int], int] = {}
    raw_rows: list[list[object]] = []
    summary_paths = sorted(input_root.glob("*/*/*/*/rollout_*/attempt_001/rollout_summary.json"))
    if len(summary_paths) != 4600:
        raise AssertionError(f"Expected 4,600 Grid summaries, found {len(summary_paths):,}")

    for index, summary_path in enumerate(summary_paths, 1):
        rel = summary_path.relative_to(input_root)
        controller, method, split, scenario_id, rollout_dir, attempt, filename = rel.parts
        if controller not in CONTROLLERS or method not in METHODS or split not in ("seen", "test"):
            raise AssertionError(f"Unexpected path identifiers: {rel}")
        summary = read_json(summary_path)
        if summary.get("status") != "complete":
            raise AssertionError(f"Incomplete result: {summary_path}")
        if int(summary.get("sample_count", -1)) != EXPECTED_STEPS:
            raise AssertionError(f"Unexpected sample_count in {summary_path}")
        manifest = read_json(summary_path.with_name("manifest.json"))
        result = read_json(summary_path.with_name("result.json"))
        if result.get("status") != "complete":
            raise AssertionError(f"Failed result: {summary_path.with_name('result.json')}")
        artifact_path = Path(manifest["artifact"])
        if not artifact_path.is_absolute():
            artifact_path = REPO_ROOT / artifact_path
        artifact = read_json(artifact_path)
        art_scenario = artifact["scenario"]
        if art_scenario["id"] != scenario_id or art_scenario["split"] != split:
            raise AssertionError(f"Scenario mismatch for {summary_path}")
        family = art_scenario["family"]
        scenario_meta[(split, scenario_id)] = family
        arrival_seed = int(artifact["arrival_seed"])
        effective_sumo_seed = int(summary["effective_sumo_seed"])
        demand_hash = str(summary["demand_hash"])
        if arrival_seed not in ARRIVAL_SEEDS:
            raise AssertionError(f"Unexpected arrival seed {arrival_seed} in {artifact_path}")
        traffic_key = (split, scenario_id, arrival_seed)
        traffic_value = (demand_hash, effective_sumo_seed)
        if traffic_key in pairing and pairing[traffic_key] != traffic_value:
            raise AssertionError(f"Demand/SUMO pairing mismatch at {traffic_key}")
        pairing[traffic_key] = traffic_value
        policy_key = (controller, split, scenario_id, arrival_seed)
        policy_seed = int(manifest["policy_seed"])
        if policy_key in policy_pairing and policy_pairing[policy_key] != policy_seed:
            raise AssertionError(f"Policy-seed pairing mismatch at {policy_key}")
        policy_pairing[policy_key] = policy_seed
        values = metric_values(summary)
        if any(value < 0 for value in values.values()):
            raise AssertionError(f"Negative metric in {summary_path}")
        npz_path = summary_path.with_name("rollout.npz")
        groups[(controller, method, split, scenario_id)].append({
            "summary_path": summary_path,
            "npz_path": npz_path,
            "arrival_seed": arrival_seed,
            "effective_sumo_seed": effective_sumo_seed,
            "demand_hash": demand_hash,
            "family": family,
            "metrics": values,
        })
        raw_rows.append([
            controller, method, split, family, scenario_id, rollout_dir,
            arrival_seed, effective_sumo_seed, demand_hash,
            *[f"{values[key]:.9g}" for key, *_ in METRICS],
        ])
        if index % 500 == 0:
            print(f"validated metadata: {index:,}/4,600", flush=True)

    expected_groups = 4 * 5 * 23
    if len(groups) != expected_groups:
        raise AssertionError(f"Expected {expected_groups} groups, found {len(groups)}")
    if len(scenario_meta) != 23:
        raise AssertionError(f"Expected 23 scenarios, found {len(scenario_meta)}")

    metric_keys = [metric[0] for metric in METRICS]
    summary_header = ["controller", "method", "split", "family", "scenario", "n"]
    for key in metric_keys:
        summary_header.extend((f"{key}_mean", f"{key}_min", f"{key}_max", f"{key}_sd"))
    summary_rows: list[list[object]] = []
    series_count = 0
    group_order = sorted(groups, key=lambda item: (CONTROLLERS.index(item[0]), METHODS.index(item[1]), item[2], item[3]))
    for group_index, key in enumerate(group_order, 1):
        controller, method, split, scenario_id = key
        records = sorted(groups[key], key=lambda record: record["arrival_seed"])
        seeds = tuple(record["arrival_seed"] for record in records)
        if seeds != ARRIVAL_SEEDS:
            raise AssertionError(f"Expected arrival seeds 51001-51010 for {key}, found {seeds}")
        totals: list[np.ndarray] = []
        reference_time = None
        for record in records:
            with np.load(record["npz_path"], allow_pickle=False) as payload:
                queue = np.asarray(payload["queue"])
                time = np.asarray(payload["time"])
            if queue.ndim != 2 or queue.shape[0] != EXPECTED_STEPS or time.shape != (EXPECTED_STEPS,):
                raise AssertionError(f"Unexpected NPZ shape in {record['npz_path']}: queue={queue.shape}, time={time.shape}")
            if not np.isfinite(queue).all() or not np.isfinite(time).all() or (queue < 0).any():
                raise AssertionError(f"Invalid numeric data in {record['npz_path']}")
            total = queue.sum(axis=1, dtype=np.float64)
            if reference_time is None:
                reference_time = time.astype(np.float64)
                expected_time = np.arange(1, EXPECTED_STEPS + 1, dtype=np.float64)
                if not np.array_equal(reference_time, expected_time):
                    raise AssertionError(f"Time axis must be 1..3600 in {record['npz_path']}")
            elif not np.array_equal(time, reference_time):
                raise AssertionError(f"Time axis mismatch in group {key}")
            summary = read_json(record["summary_path"])
            if not math.isclose(float(total.mean()), float(summary["mean_queue"]), rel_tol=1e-10, abs_tol=1e-8):
                raise AssertionError(f"mean_queue mismatch in {record['summary_path']}")
            if not math.isclose(float(total.max()), float(summary["peak_queue"]), rel_tol=0, abs_tol=1e-8):
                raise AssertionError(f"peak_queue mismatch in {record['summary_path']}")
            if not math.isclose(float(total.sum()), float(summary["integrated_queue"]), rel_tol=1e-10, abs_tol=1e-6):
                raise AssertionError(f"integrated_queue mismatch in {record['summary_path']}")
            totals.append(total)
        matrix = np.stack(totals, axis=0)
        queue_min = matrix.min(axis=0)
        queue_mean = matrix.mean(axis=0)
        queue_max = matrix.max(axis=0)
        series_path = temp_root / "series" / controller / method / split / f"{scenario_id}.csv"
        write_csv(
            series_path,
            ["time", "queue_min", "queue_mean", "queue_max", "n"],
            ((int(t), f"{lo:.6f}", f"{mean:.6f}", f"{hi:.6f}", 10)
             for t, lo, mean, hi in zip(reference_time, queue_min, queue_mean, queue_max)),
        )
        series_count += 1
        family = records[0]["family"]
        row: list[object] = [controller, method, split, family, scenario_id, len(records)]
        for metric_key in metric_keys:
            metric_data = [record["metrics"][metric_key] for record in records]
            row.extend((
                f"{statistics.fmean(metric_data):.9g}",
                f"{min(metric_data):.9g}",
                f"{max(metric_data):.9g}",
                f"{statistics.stdev(metric_data):.9g}",
            ))
        summary_rows.append(row)
        if group_index % 25 == 0:
            print(f"exported time series: {group_index:,}/{expected_groups}", flush=True)

    write_csv(temp_root / "metrics_summary.csv", summary_header, summary_rows)
    write_csv(
        temp_root / "rollout_metrics.csv",
        ["controller", "method", "split", "family", "scenario", "rollout", "arrival_seed", "sumo_seed", "demand_hash", *metric_keys],
        raw_rows,
    )

    scenario_entries = []
    for split in ("seen", "test"):
        ids = sorted(scenario_id for (item_split, scenario_id) in scenario_meta if item_split == split)
        for scenario_id in ids:
            family = scenario_meta[(split, scenario_id)]
            zh, en = scenario_label(scenario_id, family)
            scenario_entries.append({"id": scenario_id, "split": split, "family": family, "label": {"zh": zh, "en": en}})
    metrics_catalog = []
    for key, source, unit, direction, zh, en, desc_zh, desc_en in METRICS:
        metrics_catalog.append({
            "key": key,
            "source": source,
            "label": {"zh": zh, "en": en},
            "unit": {"key": unit, **UNIT_META[unit]},
            "direction": direction,
            "directionLabel": {
                "zh": "越低越好" if direction == "lower" else "越高越好" if direction == "higher" else "仅核对",
                "en": "lower is better" if direction == "lower" else "higher is better" if direction == "higher" else "verification only",
            },
            "description": {"zh": desc_zh, "en": desc_en},
        })
    catalog = {
        "schemaVersion": 1,
        "title": {"zh": "Grid 临时结果 · Protocol v7 · 训练种子 101", "en": "Preliminary Grid-only results · Protocol v7 · training seed 101"},
        "generatedFrom": str(input_root.relative_to(REPO_ROOT)),
        "controllers": [{"id": value, "label": value.upper()} for value in CONTROLLERS],
        "methods": [{"id": value, **METHOD_META[value]} for value in METHODS],
        "families": [{"id": key, "label": value} for key, value in FAMILY_META.items()],
        "scenarios": scenario_entries,
        "metrics": metrics_catalog,
        "counts": {"controllers": 4, "methods": 5, "scenarios": 23, "groups": 460, "rollouts": 4600, "stepsPerRollout": 3600, "runsPerGroup": 10},
    }
    with (temp_root / "catalog.json").open("w", encoding="utf-8") as handle:
        json.dump(catalog, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    validation = {
        "status": "passed",
        "checks": {
            "rollouts": 4600,
            "groups": 460,
            "seriesCsvFiles": series_count,
            "summaryRows": len(summary_rows),
            "stepsPerRollout": EXPECTED_STEPS,
            "arrivalSeeds": list(ARRIVAL_SEEDS),
            "pairedDemandAndSumoSeeds": True,
            "finiteNonnegativeQueue": True,
            "meanQueueReconciled": True,
        },
    }
    with (temp_root / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if output_root.exists():
        if not args.force:
            raise AssertionError("Output appeared while exporting; refusing to replace it")
        shutil.rmtree(output_root)
    temp_root.rename(output_root)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"Wrote dataset to {output_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise
