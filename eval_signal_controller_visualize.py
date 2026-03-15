"""
Visualize one traffic-signal controller on one demand group in SUMO GUI.

This script is intentionally scoped to a single rollout so you can inspect model
behavior without rerunning the full benchmark or overwriting benchmark outputs.
"""

import argparse
import csv
import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from eval_signal_controllers import (
    ACTIVE_CONTROLLERS,
    BASE_GROUP_BLOCK_SEC,
    DEFAULT_DEMAND_DIR,
    DEFAULT_GROUP_DURATION_SEC,
    DemandSubsetLargeGridEnv,
    RolloutResult,
    build_group_demand_sequence,
    build_model_for_env,
    build_rollout_seeds,
    init_logging,
    load_config,
    resolve_config_path,
    resolve_demand_groups,
    resolve_latest_checkpoint_step,
    run_single_rollout,
    save_raw_timeseries,
    slugify,
)
from tf_compat import tf


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "runs_eval", "signal_controller_visualize")
DEFAULT_CONTROLLER = "ppo_marl"
DEFAULT_GROUP = "demand_Uniform"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one controller on one demand group for visualization. "
            "GUI is enabled by default."
        )
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=DEFAULT_CONTROLLER,
        help="Controller name to load (e.g. ppo_marl, ppo_retrained).",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=DEFAULT_GROUP,
        help="Demand group name or group index from demand_dir.",
    )
    parser.add_argument(
        "--demand-dir",
        type=str,
        default=DEFAULT_DEMAND_DIR,
        help="Directory containing demand CSV files.",
    )
    parser.add_argument(
        "--group-duration-sec",
        type=int,
        default=DEFAULT_GROUP_DURATION_SEC,
        help=(
            "Simulation duration for the selected group. Must be a multiple of %d."
            % BASE_GROUP_BLOCK_SEC
        ),
    )
    parser.add_argument(
        "--evaluation-seeds",
        type=str,
        default="",
        help="Optional comma-separated seed override (same semantics as eval script).",
    )
    parser.add_argument(
        "--seed-index",
        type=int,
        default=0,
        help="Index into the evaluation seed list. 0 uses the first seed.",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default="default",
        choices=["default", "deterministic", "stochastic"],
        help="Inference policy selection mode, matching eval_signal_controllers.py.",
    )
    parser.add_argument(
        "--port-offset",
        type=int,
        default=0,
        help="SUMO port offset for this run.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where a unique run subdirectory will be created.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional extra label appended to the output run directory name.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable GUI (default is GUI enabled).",
    )
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="Print available controllers and demand groups, then exit.",
    )
    return parser.parse_args()


def build_controller_lookup() -> Dict[str, object]:
    return {spec.name: spec for spec in ACTIVE_CONTROLLERS}


def resolve_controller(controller_name: str):
    lookup = build_controller_lookup()
    if controller_name in lookup:
        return lookup[controller_name]
    raise ValueError(
        "Unknown controller '%s'. Available: %s"
        % (controller_name, ", ".join(sorted(lookup.keys())))
    )


def resolve_group_selection(
    group_arg: str,
    demand_groups: List[Tuple[int, str, str]],
) -> Tuple[int, str, str]:
    if not demand_groups:
        raise ValueError("No demand groups available.")

    by_index = {group_index: item for group_index, *item in demand_groups}
    by_name = {}
    for group_index, group_name, csv_path in demand_groups:
        by_name[group_name.lower()] = (group_index, group_name, csv_path)
        csv_name = os.path.splitext(os.path.basename(csv_path))[0].lower()
        by_name[csv_name] = (group_index, group_name, csv_path)

    token = str(group_arg).strip()
    if token.lstrip("-").isdigit():
        group_index = int(token)
        if group_index in by_index:
            name, path = by_index[group_index]
            return group_index, name, path

    normalized = token.lower().replace(".csv", "")
    if normalized in by_name:
        return by_name[normalized]

    available = ", ".join("%d:%s" % (idx, name) for idx, name, _ in demand_groups)
    raise ValueError("Unknown group '%s'. Available groups: %s" % (group_arg, available))


def ensure_unique_run_dir(
    output_root: str,
    controller: str,
    group_name: str,
    run_name: str,
) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parts = [timestamp, slugify(controller), slugify(group_name)]
    run_name = run_name.strip()
    if run_name:
        parts.append(slugify(run_name))
    base_name = "_".join(filter(None, parts))

    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)
    candidate = os.path.join(output_root, base_name)
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(output_root, "%s_%02d" % (base_name, suffix))
        suffix += 1
    os.makedirs(candidate, exist_ok=False)
    return candidate


def save_single_summary(output_dir: str, row: dict):
    summary_path = os.path.join(output_dir, "single_run_summary.csv")
    fieldnames = [
        "controller",
        "controller_label",
        "family",
        "variant",
        "group_index",
        "group_name",
        "seed_index",
        "seed_value",
        "group_duration_sec",
        "policy_type",
        "gui",
        "checkpoint_step",
        "model_dir",
        "config_path",
        "queue_overall_mean",
        "speed_overall_mean",
    ]
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def print_options():
    controller_specs = sorted(ACTIVE_CONTROLLERS, key=lambda item: item.name)
    print("Available controllers:")
    for spec in controller_specs:
        print("  %-16s %s" % (spec.name, spec.label))

    demand_groups = resolve_demand_groups(DEFAULT_DEMAND_DIR)
    print("")
    print("Available demand groups (default demand dir):")
    for group_index, group_name, _ in demand_groups:
        print("  %02d  %s" % (group_index, group_name))


def main():
    args = parse_args()

    if args.list_options:
        print_options()
        return

    if args.seed_index < 0:
        raise ValueError("seed_index must be >= 0")

    controller_spec = resolve_controller(args.controller)
    demand_groups = resolve_demand_groups(args.demand_dir)
    group_index, group_name, demand_csv = resolve_group_selection(args.group, demand_groups)
    output_dir = ensure_unique_run_dir(
        output_root=args.output_root,
        controller=controller_spec.name,
        group_name=group_name,
        run_name=args.run_name,
    )

    init_logging(output_dir)
    logging.info("Visualization output dir: %s", output_dir)
    logging.info("Controller: %s", controller_spec.name)
    logging.info("Demand group: %d (%s)", group_index, group_name)
    logging.info("Demand CSV: %s", os.path.abspath(demand_csv))
    logging.info("Group duration: %d sec", args.group_duration_sec)

    config_path = resolve_config_path(controller_spec)
    checkpoint_step = resolve_latest_checkpoint_step(controller_spec.model_dir)
    config = load_config(config_path, args.evaluation_seeds)
    seeds = build_rollout_seeds(config, args.seed_index + 1)
    selected_seed = seeds[args.seed_index]
    logging.info("Selected test seed index=%d value=%d", args.seed_index, selected_seed)
    logging.info(
        "Loading checkpoint %d from %s",
        checkpoint_step,
        os.path.abspath(controller_spec.model_dir),
    )

    env = None
    model = None
    graph = tf.Graph()

    try:
        env = DemandSubsetLargeGridEnv(
            config["ENV_CONFIG"],
            build_group_demand_sequence(demand_csv, args.group_duration_sec),
            port=args.port_offset,
            output_path=os.path.join(output_dir, ""),
        )
        env.init_test_seeds([selected_seed])

        with graph.as_default():
            model = build_model_for_env(config.get("ENV_CONFIG", "agent"), env, config["MODEL_CONFIG"])
            if not model.load(controller_spec.model_dir + os.sep, checkpoint=checkpoint_step):
                raise RuntimeError(
                    "Failed to load checkpoint %d from %s"
                    % (checkpoint_step, controller_spec.model_dir)
                )

        queue_series, speed_series = run_single_rollout(
            env=env,
            model=model,
            policy_type=args.policy_type,
            demo=(not args.no_gui),
            test_index=0,
        )
    finally:
        if env is not None:
            env.terminate()
        if model is not None:
            try:
                model.sess.close()
            except Exception:
                pass

    result = RolloutResult(queue_ts=[queue_series], speed_ts=[speed_series])
    save_raw_timeseries(
        output_dir=output_dir,
        controller_name=controller_spec.name,
        group_index=group_index,
        group_name=group_name,
        result=result,
    )

    queue_mean = float(np.mean(queue_series)) if queue_series else float("nan")
    speed_mean = float(np.mean(speed_series)) if speed_series else float("nan")
    summary_row = {
        "controller": controller_spec.name,
        "controller_label": controller_spec.label,
        "family": controller_spec.family,
        "variant": controller_spec.variant,
        "group_index": group_index,
        "group_name": group_name,
        "seed_index": args.seed_index,
        "seed_value": selected_seed,
        "group_duration_sec": args.group_duration_sec,
        "policy_type": args.policy_type,
        "gui": int(not args.no_gui),
        "checkpoint_step": checkpoint_step,
        "model_dir": os.path.abspath(controller_spec.model_dir),
        "config_path": os.path.abspath(config_path),
        "queue_overall_mean": queue_mean,
        "speed_overall_mean": speed_mean,
    }
    save_single_summary(output_dir, summary_row)

    logging.info("Single visualization run complete.")
    logging.info("Queue overall mean: %.6f", queue_mean)
    logging.info("Speed overall mean: %.6f", speed_mean)
    logging.info("Saved raw series + summary under: %s", output_dir)


if __name__ == "__main__":
    main()
