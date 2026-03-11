"""
Evaluate baseline and retrained traffic-signal controllers on fixed demand groups.

This script mirrors the structure of the older `eval_marl_vs_drmarl.py` workflow,
but uses the local TensorFlow checkpoints from this project instead of exported
CSV weights. It evaluates one demand CSV at a time so each controller gets a
separate result for each demand group.
"""

import argparse
import configparser
import csv
import glob
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents.models import A2C, IA2C, IQL, MA2C, PPO
from envs.large_grid_env import LargeGridEnv
from tf_compat import tf


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEMAND_DIR = os.path.join(PROJECT_ROOT, "data_traffic")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runs_eval", "signal_controller_benchmark")
BASE_GROUP_BLOCK_SEC = 600
DEFAULT_NUM_ROLLOUTS = 10
DEFAULT_GROUP_DURATION_SEC = 3600
DEMAND_GROUP_ORDER = [
    "demand_N_to_S",
    "demand_S_to_N",
    "demand_W_to_E",
    "demand_E_to_W",
    "demand_NW_to_SE",
    "demand_SE_to_NW",
    "demand_SW_to_NE",
    "demand_NE_to_SW",
    "demand_Periphery_to_Center",
    "demand_Center_to_Periphery",
    "demand_Uniform",
]


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    label: str
    family: str
    variant: str
    model_dir: str
    config_hint: str


@dataclass
class RolloutResult:
    queue_ts: List[List[float]]
    speed_ts: List[List[float]]


@dataclass
class GroupStats:
    queue_mean: np.ndarray
    queue_std: np.ndarray
    queue_min: np.ndarray
    queue_max: np.ndarray
    speed_mean: np.ndarray
    speed_std: np.ndarray
    speed_min: np.ndarray
    speed_max: np.ndarray
    queue_overall_mean: float
    speed_overall_mean: float


ACTIVE_CONTROLLERS: List[ControllerSpec] = [
    ControllerSpec(
        name="ia2c_marl",
        label="IA2C MARL",
        family="ia2c",
        variant="baseline",
        model_dir=os.path.join(PROJECT_ROOT, "runs", "ia2c_large", "model"),
        config_hint=os.path.join(PROJECT_ROOT, "runs", "ia2c_large", "data", "config_ia2c_large.ini"),
    ),
    ControllerSpec(
        name="ma2c_marl",
        label="MA2C MARL",
        family="ma2c",
        variant="baseline",
        model_dir=os.path.join(PROJECT_ROOT, "runs", "ma2c_large", "model"),
        config_hint=os.path.join(PROJECT_ROOT, "runs", "ma2c_large", "data", "config_ma2c_large.ini"),
    ),
    # ControllerSpec(
    #     name="iqll_marl",
    #     label="IQL-LR MARL",
    #     family="iqll",
    #     variant="baseline",
    #     model_dir=os.path.join(PROJECT_ROOT, "runs", "iqll_large", "model"),
    #     config_hint=os.path.join(PROJECT_ROOT, "runs", "iqll_large", "data", "config_iqll_large.ini"),
    # ),
    ControllerSpec(
        name="ia2c_retrained",
        label="IA2C Retrained",
        family="ia2c",
        variant="retrained",
        model_dir=os.path.join(PROJECT_ROOT, "output_coevolution", "ia2c_large", "model_traffic"),
        config_hint=os.path.join(PROJECT_ROOT, "runs", "ia2c_large", "data", "config_ia2c_large.ini"),
    ),
    ControllerSpec(
        name="ma2c_retrained",
        label="MA2C Retrained",
        family="ma2c",
        variant="retrained",
        model_dir=os.path.join(PROJECT_ROOT, "output_coevolution", "ma2c_large", "model_traffic"),
        config_hint=os.path.join(PROJECT_ROOT, "runs", "ma2c_large", "data", "config_ma2c_large.ini"),
    ),
    # ControllerSpec(
    #     name="iqll_retrained",
    #     label="IQL-LR Retrained",
    #     family="iqll",
    #     variant="retrained",
    #     model_dir=os.path.join(PROJECT_ROOT, "output_coevolution", "iqll_large", "model_traffic"),
    #     config_hint=os.path.join(PROJECT_ROOT, "runs", "iqll_large", "data", "config_iqll_large.ini"),
    # ),
]


COMPARISON_PAIRS: List[Tuple[str, str]] = [
    ("ia2c_marl", "ia2c_retrained"),
    ("ma2c_marl", "ma2c_retrained"),
    # ("iqll_marl", "iqll_retrained"),
]


class DemandSubsetLargeGridEnv(LargeGridEnv):
    """Large-grid env that only loads the provided demand CSVs."""

    def __init__(self, config, demand_csvs: Sequence[str], port: int = 0, output_path: str = ""):
        self._demand_csvs = [os.path.abspath(path) for path in demand_csvs]
        super().__init__(config, port=port, output_path=output_path, is_record=True, record_stat=False)

    def _resolve_traffic_data_dir(self):
        if self._demand_csvs:
            return os.path.dirname(self._demand_csvs[0])
        return super()._resolve_traffic_data_dir()

    def _load_dynamic_scenarios(self):
        if not self._demand_csvs:
            return super()._load_dynamic_scenarios()

        self.csv_paths = list(self._demand_csvs)
        self.scenarios = []
        self.loaded_filenames = []

        for i, path in enumerate(self.csv_paths):
            if not os.path.exists(path):
                raise FileNotFoundError("Demand CSV does not exist: %s" % path)

            df = pd.read_csv(path)
            if "veh_per_hour" not in df.columns:
                raise ValueError("Demand CSV missing 'veh_per_hour' column: %s" % path)

            scenario_data = []
            for _, row in df.iterrows():
                scenario_data.append(
                    {
                        "origin": row["origin_edge"],
                        "dest": row["dest_edge"],
                        "rate": row["veh_per_hour"],
                    }
                )

            self.scenarios.append(scenario_data)
            self.loaded_filenames.append(os.path.basename(path).replace(".csv", ""))
            logging.info("Loaded demand subset %d from %s", i, os.path.basename(path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demand-dir",
        type=str,
        default=DEFAULT_DEMAND_DIR,
        help="Directory containing the demand group CSV files to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used for CSV exports, figures, and the evaluation log.",
    )
    parser.add_argument(
        "--controllers",
        type=str,
        default="",
        help="Optional comma-separated subset of controller names to run.",
    )
    parser.add_argument(
        "--evaluation-seeds",
        type=str,
        default="",
        help="Override ENV_CONFIG.test_seeds with a comma-separated list.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=DEFAULT_NUM_ROLLOUTS,
        help="Number of rollouts to run per controller and demand group.",
    )
    parser.add_argument(
        "--group-duration-sec",
        type=int,
        default=DEFAULT_GROUP_DURATION_SEC,
        help=(
            "Total simulation time per demand group. Must be a multiple of %d seconds."
            % BASE_GROUP_BLOCK_SEC
        ),
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default="default",
        choices=["default", "deterministic", "stochastic"],
        help="Inference policy selection mode, matching main.py evaluate.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use SUMO GUI during rollouts.",
    )
    return parser.parse_args()


def init_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "evaluation.log")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)


def get_family_output_dir(base_output_dir: str, family: str) -> str:
    family_dir = os.path.join(base_output_dir, family)
    os.makedirs(family_dir, exist_ok=True)
    return family_dir


def normalize_config_paths(config: configparser.ConfigParser):
    if config.has_option("ENV_CONFIG", "data_path"):
        data_path = config.get("ENV_CONFIG", "data_path").strip()
        if data_path and not os.path.isabs(data_path):
            config.set(
                "ENV_CONFIG",
                "data_path",
                os.path.abspath(os.path.join(PROJECT_ROOT, data_path)),
            )


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def resolve_demand_groups(demand_dir: str) -> List[Tuple[int, str, str]]:
    search_path = os.path.join(os.path.abspath(demand_dir), "*.csv")
    csv_paths = glob.glob(search_path)
    if not csv_paths:
        raise FileNotFoundError("No demand CSV files found under %s" % os.path.abspath(demand_dir))

    path_by_name = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in csv_paths
    }

    missing = [name for name in DEMAND_GROUP_ORDER if name not in path_by_name]
    if missing:
        raise FileNotFoundError(
            "Missing expected demand CSV files: %s" % ", ".join(missing)
        )

    ordered_names = list(DEMAND_GROUP_ORDER)
    extra_names = sorted(name for name in path_by_name if name not in DEMAND_GROUP_ORDER)
    ordered_names.extend(extra_names)

    groups = []
    for index, name in enumerate(ordered_names):
        groups.append((index, name, path_by_name[name]))
    return groups


def resolve_selected_controllers(controller_arg: str) -> List[ControllerSpec]:
    if not controller_arg.strip():
        return list(ACTIVE_CONTROLLERS)

    requested = [token.strip() for token in controller_arg.split(",") if token.strip()]
    available = {spec.name: spec for spec in ACTIVE_CONTROLLERS}
    selected = []
    for name in requested:
        if name not in available:
            raise ValueError("Unknown controller name '%s'. Available: %s" % (name, ", ".join(sorted(available))))
        selected.append(available[name])
    return selected


def resolve_config_path(spec: ControllerSpec) -> str:
    candidates = [spec.config_hint]
    fallback_run_config = os.path.join(
        PROJECT_ROOT,
        "config",
        "config_%s_large.ini" % spec.family,
    )
    if fallback_run_config not in candidates:
        candidates.append(fallback_run_config)

    model_parent = os.path.dirname(spec.model_dir)
    sibling_data = os.path.abspath(os.path.join(model_parent, "..", "data"))
    if os.path.isdir(sibling_data):
        matches = sorted(glob.glob(os.path.join(sibling_data, "*.ini")))
        candidates.extend(matches)

    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)

    raise FileNotFoundError("Could not resolve config for controller %s" % spec.name)


def resolve_latest_checkpoint_step(model_dir: str) -> int:
    pattern = os.path.join(model_dir, "checkpoint-*.index")
    checkpoint_paths = glob.glob(pattern)
    if not checkpoint_paths:
        raise FileNotFoundError("No checkpoints found under %s" % model_dir)

    steps = []
    for path in checkpoint_paths:
        filename = os.path.basename(path)
        match = re.match(r"checkpoint-(\d+)\.index$", filename)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError("No numbered checkpoint files found under %s" % model_dir)
    return max(steps)


def pad_and_stack(series_list: Sequence[Sequence[float]]) -> np.ndarray:
    if not series_list:
        raise ValueError("No time series available to aggregate.")

    max_len = max(len(series) for series in series_list)
    arr = np.zeros((len(series_list), max_len), dtype=np.float32)
    for i, series in enumerate(series_list):
        if not series:
            continue
        arr[i, : len(series)] = series
        if len(series) < max_len:
            arr[i, len(series) :] = series[-1]
    return arr


def summarize_group(result: RolloutResult) -> GroupStats:
    queue_arr = pad_and_stack(result.queue_ts)
    speed_arr = pad_and_stack(result.speed_ts)
    return GroupStats(
        queue_mean=queue_arr.mean(axis=0),
        queue_std=queue_arr.std(axis=0),
        queue_min=queue_arr.min(axis=0),
        queue_max=queue_arr.max(axis=0),
        speed_mean=speed_arr.mean(axis=0),
        speed_std=speed_arr.std(axis=0),
        speed_min=speed_arr.min(axis=0),
        speed_max=speed_arr.max(axis=0),
        queue_overall_mean=float(queue_arr.mean()),
        speed_overall_mean=float(speed_arr.mean()),
    )


def save_raw_timeseries(
    output_dir: str,
    controller_name: str,
    group_index: int,
    group_name: str,
    result: RolloutResult,
):
    queue_arr = pad_and_stack(result.queue_ts)
    speed_arr = pad_and_stack(result.speed_ts)
    num_rollouts, horizon = queue_arr.shape
    t_axis = np.arange(1, horizon + 1, dtype=int)
    group_slug = slugify(group_name)
    prefix = "%s_group%02d_%s" % (controller_name, group_index, group_slug)

    queue_path = os.path.join(output_dir, prefix + "_queue_raw.csv")
    with open(queue_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t"] + ["rollout_%d" % i for i in range(num_rollouts)])
        for time_index in range(horizon):
            writer.writerow([t_axis[time_index]] + [queue_arr[r, time_index] for r in range(num_rollouts)])

    speed_path = os.path.join(output_dir, prefix + "_speed_raw.csv")
    with open(speed_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t"] + ["rollout_%d" % i for i in range(num_rollouts)])
        for time_index in range(horizon):
            writer.writerow([t_axis[time_index]] + [speed_arr[r, time_index] for r in range(num_rollouts)])


def plot_groups(
    stats_per_group: Dict[int, GroupStats],
    group_labels: Dict[int, str],
    title: str,
    ylabel: str,
    save_path: str,
    metric: str,
):
    plt.figure(figsize=(11, 6))
    for group_index, stats in sorted(stats_per_group.items()):
        label = group_labels.get(group_index, "group %d" % group_index)
        if metric == "queue":
            plt.plot(np.arange(1, len(stats.queue_mean) + 1), stats.queue_mean, label=label)
        else:
            plt.plot(np.arange(1, len(stats.speed_mean) + 1), stats.speed_mean, label=label)
    plt.xlabel("Simulation second", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_worst_group_comparison(
    stats_left: GroupStats,
    stats_right: GroupStats,
    left_label: str,
    right_label: str,
    title: str,
    ylabel: str,
    save_path: str,
    metric: str,
):
    plt.figure(figsize=(11, 6))

    if metric == "queue":
        left_mean = stats_left.queue_mean
        left_min = stats_left.queue_min
        left_max = stats_left.queue_max
        right_mean = stats_right.queue_mean
        right_min = stats_right.queue_min
        right_max = stats_right.queue_max
    else:
        left_mean = stats_left.speed_mean
        left_min = stats_left.speed_min
        left_max = stats_left.speed_max
        right_mean = stats_right.speed_mean
        right_min = stats_right.speed_min
        right_max = stats_right.speed_max

    left_t = np.arange(1, len(left_mean) + 1)
    right_t = np.arange(1, len(right_mean) + 1)

    left_line, = plt.plot(left_t, left_mean, label=left_label)
    plt.fill_between(left_t, left_min, left_max, alpha=0.18, color=left_line.get_color())

    right_line, = plt.plot(right_t, right_mean, label=right_label)
    plt.fill_between(right_t, right_min, right_max, alpha=0.18, color=right_line.get_color())

    plt.xlabel("Simulation second", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_config(config_path: str, evaluation_seeds: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(config_path)
    normalize_config_paths(config)
    if evaluation_seeds.strip():
        config.set("ENV_CONFIG", "test_seeds", evaluation_seeds.strip())
    return config


def parse_seed_list(config: configparser.ConfigParser) -> List[int]:
    seeds_raw = config.get("ENV_CONFIG", "test_seeds")
    return [int(token.strip()) for token in seeds_raw.split(",") if token.strip()]


def build_rollout_seeds(config: configparser.ConfigParser, num_rollouts: int) -> List[int]:
    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive.")

    base_seeds = parse_seed_list(config)
    if not base_seeds:
        raise ValueError("No evaluation seeds configured.")

    seeds = list(base_seeds[:num_rollouts])
    next_seed = seeds[-1] if seeds else base_seeds[-1]
    while len(seeds) < num_rollouts:
        next_seed += 10000
        seeds.append(next_seed)
    return seeds


def build_group_demand_sequence(demand_csv: str, group_duration_sec: int) -> List[str]:
    if group_duration_sec <= 0:
        raise ValueError("group_duration_sec must be positive.")
    if group_duration_sec % BASE_GROUP_BLOCK_SEC != 0:
        raise ValueError(
            "group_duration_sec must be a multiple of %d seconds." % BASE_GROUP_BLOCK_SEC
        )

    repeat_count = group_duration_sec // BASE_GROUP_BLOCK_SEC
    return [demand_csv] * repeat_count


def build_model_for_env(agent: str, env, model_config):
    if agent == "a2c":
        return A2C(env.n_s, env.n_a, 0, model_config)
    if agent == "ia2c":
        return IA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, 0, model_config)
    if agent == "ma2c":
        return MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls, env.n_f_ls, 0, model_config)
    if agent == "ppo":
        return PPO(env.n_s_ls, env.n_a_ls, env.n_w_ls, 0, model_config)
    if agent == "iqld":
        return IQL(env.n_s_ls, env.n_a_ls, env.n_w_ls, 0, model_config, seed=0, model_type="dqn")
    if agent == "iqll":
        return IQL(env.n_s_ls, env.n_a_ls, env.n_w_ls, 0, model_config, seed=0, model_type="lr")
    raise ValueError("Unsupported agent type for this evaluator: %s" % agent)


def is_policy_gradient_agent(agent: str) -> bool:
    return str(agent).strip().lower().endswith("a2c") or str(agent).strip().lower() == "ppo"


def select_action(model, agent: str, env, ob, done: bool, policy_type: str):
    if agent == "greedy":
        return model.forward(ob)

    if is_policy_gradient_agent(agent):
        policy = model.forward(ob, done, "p")
        if agent == "ma2c":
            env.update_fingerprint(policy)
        if agent == "a2c":
            if policy_type != "deterministic":
                return int(np.random.choice(np.arange(len(policy)), p=policy))
            return int(np.argmax(np.asarray(policy)))

        actions = []
        for pi in policy:
            if policy_type != "deterministic":
                actions.append(int(np.random.choice(np.arange(len(pi)), p=pi)))
            else:
                actions.append(int(np.argmax(np.asarray(pi))))
        return actions

    if policy_type != "stochastic":
        action, _ = model.forward(ob)
    else:
        action, _ = model.forward(ob, stochastic=True)
    return action


def extract_episode_rows(rows: Sequence[dict], episode_id: int) -> List[dict]:
    return [row for row in rows if int(row.get("episode", -1)) == int(episode_id)]


def collect_tripinfo_with_retry(env):
    last_error = None
    for delay in (2.0, 1.0, 1.0, 2.0, 3.0):
        if delay:
            time.sleep(delay)
        try:
            env.collect_tripinfo()
            return
        except (FileNotFoundError, ET.ParseError) as exc:
            last_error = exc
            logging.warning("Tripinfo XML not ready yet, retrying: %s", exc)
        except Exception as exc:
            last_error = exc
            break
    raise last_error


def run_single_rollout(env, model, policy_type: str, demo: bool, test_index: int) -> Tuple[List[float], List[float]]:
    ob = env.reset(gui=demo, test_ind=test_index)
    done = True
    model.reset()

    while True:
        action = select_action(model, env.agent, env, ob, done, policy_type)
        next_ob, _, done, _ = env.step(action)
        if done:
            break
        ob = next_ob

    episode_id = env.cur_episode
    env.terminate()
    collect_tripinfo_with_retry(env)

    traffic_rows = extract_episode_rows(env.traffic_data, episode_id)
    if not traffic_rows:
        raise RuntimeError("No traffic data recorded for episode %d" % episode_id)

    sorted_rows = sorted(traffic_rows, key=lambda row: row["time_sec"])
    queue_series = [
        float(row["total_queue"]) if "total_queue" in row else float(row["avg_queue"])
        for row in sorted_rows
    ]
    speed_series = [float(row["avg_speed_mps"]) for row in sorted_rows]
    return queue_series, speed_series


def evaluate_group(
    env,
    model,
    seeds: Sequence[int],
    controller_name: str,
    group_name: str,
    policy_type: str,
    demo: bool,
) -> RolloutResult:
    queue_ts: List[List[float]] = []
    speed_ts: List[List[float]] = []

    for rollout_index, _ in enumerate(seeds):
        logging.info(
            "Running rollout %d/%d for controller=%s group=%s",
            rollout_index + 1,
            len(seeds),
            controller_name,
            group_name,
        )
        queue_series, speed_series = run_single_rollout(
            env=env,
            model=model,
            policy_type=policy_type,
            demo=demo,
            test_index=rollout_index,
        )
        queue_ts.append(queue_series)
        speed_ts.append(speed_series)

    return RolloutResult(queue_ts=queue_ts, speed_ts=speed_ts)


def save_summary_csv(
    output_dir: str,
    summary_rows: Sequence[dict],
):
    summary_path = os.path.join(output_dir, "summary_averages.csv")
    fieldnames = [
        "controller",
        "controller_label",
        "family",
        "variant",
        "group_index",
        "group_name",
        "queue_metric",
        "queue_overall_mean",
        "speed_overall_mean",
        "checkpoint_step",
        "model_dir",
        "config_path",
    ]
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def save_horizon_averages_csv(
    output_dir: str,
    summary_rows: Sequence[dict],
    num_rollouts: int,
    horizon_seconds: int,
):
    horizon_path = os.path.join(output_dir, "horizon_averages_from_raw.csv")
    controller_order = [
        "ia2c_marl",
        "ia2c_retrained",
        "ma2c_marl",
        "ma2c_retrained",
        "iqll_marl",
        "iqll_retrained",
    ]
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

    rows = []
    for row in summary_rows:
        controller = row["controller"]
        controller_variant = controller.split("_", 1)[1] if "_" in controller else row["variant"]
        rows.append(
            {
                "controller": controller,
                "family": row["family"],
                "variant": controller_variant,
                "group_index": row["group_index"],
                "group_name": row["group_name"],
                "group_slug": slugify(row["group_name"]),
                "num_rollouts": num_rollouts,
                "horizon_seconds": horizon_seconds,
                "queue_horizon_average": row["queue_overall_mean"],
                "speed_horizon_average": row["speed_overall_mean"],
            }
        )

    controller_rank = {name: index for index, name in enumerate(controller_order)}
    rows.sort(
        key=lambda item: (
            controller_rank.get(item["controller"], len(controller_order)),
            int(item["group_index"]),
        )
    )

    with open(horizon_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_manifest_csv(output_dir: str, manifest_rows: Sequence[dict]):
    manifest_path = os.path.join(output_dir, "controller_manifest.csv")
    fieldnames = [
        "controller",
        "controller_label",
        "family",
        "variant",
        "checkpoint_step",
        "model_dir",
        "config_path",
    ]
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)


def save_demand_manifest(output_dir: str, demand_groups: Sequence[Tuple[int, str, str]]):
    demand_path = os.path.join(output_dir, "demand_manifest.csv")
    with open(demand_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["group_index", "group_name", "csv_path"])
        for group_index, group_name, csv_path in demand_groups:
            writer.writerow([group_index, group_name, os.path.abspath(csv_path)])


def find_worst_group(stats_per_group: Dict[int, GroupStats]) -> Tuple[int, GroupStats]:
    group_index = max(
        stats_per_group.items(),
        key=lambda item: item[1].queue_overall_mean,
    )[0]
    return group_index, stats_per_group[group_index]


def build_pair_lookup(controller_specs: Sequence[ControllerSpec]) -> Dict[str, ControllerSpec]:
    return {spec.name: spec for spec in controller_specs}


def evaluate_controller(
    spec: ControllerSpec,
    demand_groups: Sequence[Tuple[int, str, str]],
    evaluation_seeds: str,
    num_rollouts: int,
    group_duration_sec: int,
    policy_type: str,
    demo: bool,
    output_dir: str,
    port_offset: int,
) -> Tuple[Dict[int, GroupStats], dict]:
    config_path = resolve_config_path(spec)
    checkpoint_step = resolve_latest_checkpoint_step(spec.model_dir)
    config = load_config(config_path, evaluation_seeds)
    seeds = build_rollout_seeds(config, num_rollouts)
    family_output_dir = get_family_output_dir(output_dir, spec.family)
    if not seeds:
        raise ValueError("No evaluation seeds configured for controller %s" % spec.name)

    family_stats: Dict[int, GroupStats] = {}

    logging.info(
        "Controller %s: loading checkpoint %d from %s",
        spec.name,
        checkpoint_step,
        spec.model_dir,
    )

    model = None
    graph = tf.Graph()

    try:
        for group_offset, (group_index, group_name, demand_csv) in enumerate(demand_groups):
            logging.info(
                "Evaluating controller=%s on group=%d (%s)",
                spec.name,
                group_index,
                group_name,
            )

            env = DemandSubsetLargeGridEnv(
                config["ENV_CONFIG"],
                build_group_demand_sequence(demand_csv, group_duration_sec),
                port=port_offset,
                output_path=os.path.join(family_output_dir, ""),
            )
            env.init_test_seeds(seeds)

            try:
                if model is None:
                    with graph.as_default():
                        model = build_model_for_env(config.get("ENV_CONFIG", "agent"), env, config["MODEL_CONFIG"])
                        if not model.load(spec.model_dir + os.sep, checkpoint=checkpoint_step):
                            raise RuntimeError(
                                "Failed to load checkpoint %d from %s" % (checkpoint_step, spec.model_dir)
                            )

                result = evaluate_group(
                    env=env,
                    model=model,
                    seeds=seeds,
                    controller_name=spec.name,
                    group_name=group_name,
                    policy_type=policy_type,
                    demo=demo,
                )
            finally:
                env.terminate()

            save_raw_timeseries(
                output_dir=family_output_dir,
                controller_name=spec.name,
                group_index=group_index,
                group_name=group_name,
                result=result,
            )
            family_stats[group_index] = summarize_group(result)

    finally:
        if model is not None:
            try:
                model.sess.close()
            except Exception:
                pass

    manifest_row = {
        "controller": spec.name,
        "controller_label": spec.label,
        "family": spec.family,
        "variant": spec.variant,
        "checkpoint_step": checkpoint_step,
        "model_dir": os.path.abspath(spec.model_dir),
        "config_path": os.path.abspath(config_path),
    }
    return family_stats, manifest_row


def main():
    args = parse_args()
    init_logging(args.output_dir)

    logging.info("Output directory: %s", os.path.abspath(args.output_dir))
    logging.info("Demand directory: %s", os.path.abspath(args.demand_dir))
    logging.info("Rollouts per controller/group: %d", args.num_rollouts)
    logging.info("Group duration: %d seconds", args.group_duration_sec)

    controller_specs = resolve_selected_controllers(args.controllers)
    demand_groups = resolve_demand_groups(args.demand_dir)
    group_labels = {group_index: group_name for group_index, group_name, _ in demand_groups}

    save_demand_manifest(args.output_dir, demand_groups)

    all_stats: Dict[str, Dict[int, GroupStats]] = {}
    summary_rows: List[dict] = []
    manifest_rows: List[dict] = []
    spec_lookup = build_pair_lookup(controller_specs)

    for controller_index, spec in enumerate(controller_specs):
        family_output_dir = get_family_output_dir(args.output_dir, spec.family)
        stats_per_group, manifest_row = evaluate_controller(
            spec=spec,
            demand_groups=demand_groups,
            evaluation_seeds=args.evaluation_seeds,
            num_rollouts=args.num_rollouts,
            group_duration_sec=args.group_duration_sec,
            policy_type=args.policy_type,
            demo=args.demo,
            output_dir=args.output_dir,
            port_offset=controller_index,
        )
        all_stats[spec.name] = stats_per_group
        manifest_rows.append(manifest_row)

        for group_index, group_name, _ in demand_groups:
            group_stats = stats_per_group[group_index]
            summary_rows.append(
                {
                    "controller": spec.name,
                    "controller_label": spec.label,
                    "family": spec.family,
                    "variant": spec.variant,
                    "group_index": group_index,
                    "group_name": group_name,
                    "queue_metric": "total_queue",
                    "queue_overall_mean": group_stats.queue_overall_mean,
                    "speed_overall_mean": group_stats.speed_overall_mean,
                    "checkpoint_step": manifest_row["checkpoint_step"],
                    "model_dir": manifest_row["model_dir"],
                    "config_path": manifest_row["config_path"],
                }
            )

        plot_groups(
            stats_per_group,
            group_labels,
            title="%s - Total Queue by Demand Group" % spec.label,
            ylabel="Total queued vehicles (veh)",
            save_path=os.path.join(family_output_dir, "%s_queue_by_group.png" % spec.name),
            metric="queue",
        )
        plot_groups(
            stats_per_group,
            group_labels,
            title="%s - Average Speed by Demand Group" % spec.label,
            ylabel="Average speed (m/s)",
            save_path=os.path.join(family_output_dir, "%s_speed_by_group.png" % spec.name),
            metric="speed",
        )

    for left_name, right_name in COMPARISON_PAIRS:
        if left_name not in all_stats or right_name not in all_stats:
            continue

        left_spec = spec_lookup[left_name]
        right_spec = spec_lookup[right_name]
        family_output_dir = get_family_output_dir(args.output_dir, left_spec.family)
        left_group, left_stats = find_worst_group(all_stats[left_name])
        right_group, right_stats = find_worst_group(all_stats[right_name])

        family_slug = slugify(left_spec.family)
        plot_worst_group_comparison(
            stats_left=left_stats,
            stats_right=right_stats,
            left_label="%s worst (%s)" % (left_spec.label, group_labels[left_group]),
            right_label="%s worst (%s)" % (right_spec.label, group_labels[right_group]),
            title="%s Baseline vs Retrained - Worst Queue Groups" % left_spec.family.upper(),
            ylabel="Total queued vehicles (veh)",
            save_path=os.path.join(family_output_dir, "%s_worst_queue.png" % family_slug),
            metric="queue",
        )
        plot_worst_group_comparison(
            stats_left=left_stats,
            stats_right=right_stats,
            left_label="%s worst (%s)" % (left_spec.label, group_labels[left_group]),
            right_label="%s worst (%s)" % (right_spec.label, group_labels[right_group]),
            title="%s Baseline vs Retrained - Worst Speed Groups" % left_spec.family.upper(),
            ylabel="Average speed (m/s)",
            save_path=os.path.join(family_output_dir, "%s_worst_speed.png" % family_slug),
            metric="speed",
        )

    save_summary_csv(args.output_dir, summary_rows)
    save_horizon_averages_csv(
        args.output_dir,
        summary_rows,
        num_rollouts=args.num_rollouts,
        horizon_seconds=args.group_duration_sec,
    )
    save_manifest_csv(args.output_dir, manifest_rows)
    logging.info("Evaluation complete.")


if __name__ == "__main__":
    main()
