#!/usr/bin/env python3
"""Manually compare four existing raw eval CSVs from runs_eval.
Automatically extracts group names for the legend.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Tuple

_MPLCONFIGDIR = Path("/tmp/matplotlib")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib as mpl

mpl.use("Agg")


def _tex_package_available(package_name: str) -> bool:
    kpsewhich = shutil.which("kpsewhich")
    if not kpsewhich:
        return False
    result = subprocess.run(
        [kpsewhich, package_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def _configure_plot_style() -> None:
    params = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}",
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 11, # 图例字数变多了，稍微调小一点字体
        "figure.titlesize": 20,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    missing_bins = [exe for exe in ("latex", "kpsewhich", "dvipng") if shutil.which(exe) is None]
    required_sty = ("newtxtext.sty", "newtxmath.sty", "type1ec.sty")
    missing_sty = [sty for sty in required_sty if not _tex_package_available(sty)]

    if missing_bins or missing_sty:
        params["text.usetex"] = False
        params.pop("text.latex.preamble", None)
        warnings.warn(
            "Falling back to Matplotlib text rendering because TeX dependencies are missing. "
            "Missing executables: %s. Missing TeX packages: %s."
            % (missing_bins or "none", missing_sty or "none"),
            RuntimeWarning,
        )

    mpl.rcParams.update(params)


_configure_plot_style()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ===========================
# Manual configuration block
# ===========================
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_EVAL_DIR = PROJECT_ROOT / "runs_eval"
# OUTPUT_DIR = RUNS_EVAL_DIR / "manual_comparisons_real"

# OUTPUT_FILENAME = "All_Algorithms_Queue_Comparison_label.png"
# PLOT_TITLE = "Worst-case Queue Comparison (All Algorithms) in the Monaco City dataset"
# Y_LABEL = "Total queued vehicles (veh)"

# DATA_PAIRS = [
#     {
#         "algo_name": "IA2C",
#         "baseline_csv": "signal_controller_benchmark_real/ia2c_real_group10_uniform_queue_raw.csv",
#         "retrained_csv": "signal_controller_benchmark_real/ia2c_retrained_group01_s_to_n_queue_raw.csv",
#         "color": "#1f77b4" # Matplotlib tab:blue
#     },
#     {
#         "algo_name": "MA2C",
#         "baseline_csv": "signal_controller_benchmark_real/ma2c_real_group01_s_to_n_queue_raw.csv",
#         "retrained_csv": "signal_controller_benchmark_real/ma2c_retrained_group01_s_to_n_queue_raw.csv",
#         "color": "#ff7f0e" # Matplotlib tab:orange
#     },
#     {
#         "algo_name": "IQL-LR",
#         "baseline_csv": "signal_controller_benchmark_real/iqll_real_group10_uniform_queue_raw.csv",
#         "retrained_csv": "signal_controller_benchmark_real/iqll_retrained_group05_se_to_nw_queue_raw.csv",
#         "color": "#2ca02c" # Matplotlib tab:green
#     },
#     {
#         "algo_name": "PPO",
#         "baseline_csv": "signal_controller_benchmark_real/ppo_real_group10_uniform_queue_raw.csv",
#         "retrained_csv": "signal_controller_benchmark_real/ppo_retrained_group01_s_to_n_queue_raw.csv",
#         "color": "#d62728" # Matplotlib tab:red
#     }
# ]

# OUTPUT_DIR = RUNS_EVAL_DIR / "manual_comparisons_real"

# OUTPUT_FILENAME = "All_Algorithms_Speed_Comparison_label.png"
# PLOT_TITLE = "Worst-case Speed Comparison (All Algorithms) in the Monaco City Dataset"
# Y_LABEL = "Average vehicle speed (km/h)"

# DATA_PAIRS = [
#     {
#         "algo_name": "IA2C",
#         "baseline_csv": "signal_controller_benchmark_real/ia2c_real_group10_uniform_speed_raw.csv",
#         "retrained_csv": "signal_controller_benchmark_real/ia2c_retrained_group01_s_to_n_speed_raw.csv",
#         "color": "#1f77b4" # Matplotlib tab:blue
#     },
#     {
#         "algo_name": "MA2C",
#         "baseline_csv": "signal_controller_benchmark_real/ma2c_real_group01_s_to_n_speed_raw.csv",
#         "retrained_csv": "signal_controller_benchmark_real/ma2c_retrained_group01_s_to_n_speed_raw.csv",
#         "color": "#ff7f0e" # Matplotlib tab:orange
#     },
#     {
#         "algo_name": "IQL-LR",
#         "baseline_csv": "signal_controller_benchmark_real/iqll_real_group10_uniform_speed_raw.csv",
#         "retrained_csv": "signal_controller_benchmark_real/iqll_retrained_group05_se_to_nw_speed_raw.csv",
#         "color": "#2ca02c" # Matplotlib tab:green
#     },
#     {
#         "algo_name": "PPO",
#         "baseline_csv": "signal_controller_benchmark_real/ppo_real_group10_uniform_speed_raw.csv",
#         "retrained_csv": "signal_controller_benchmark_real/ppo_retrained_group01_s_to_n_speed_raw.csv",
#         "color": "#d62728" # Matplotlib tab:red
#     }
# ]

# OUTPUT_DIR = RUNS_EVAL_DIR / "manual_comparisons"

# OUTPUT_FILENAME = "All_Algorithms_Speed_Comparison.png"
# PLOT_TITLE = "Worst-case Speed Comparison (All Algorithms) in 5x5 Grid"
# Y_LABEL = "Average vehicle speed (km/h)"

# DATA_PAIRS = [
#     {
#         "algo_name": "IA2C",
#         "baseline_csv": "signal_controller_benchmark/ia2c_marl_group02_demand_w_to_e_speed_raw.csv",
#         "retrained_csv": "signal_controller_benchmark/ia2c_retrained_group06_demand_sw_to_ne_speed_raw.csv",
#         "color": "#1f77b4" # Matplotlib tab:blue
#     },
#     {
#         "algo_name": "MA2C",
#         "baseline_csv": "signal_controller_benchmark/ma2c_marl_group06_demand_sw_to_ne_speed_raw.csv",
#         "retrained_csv": "signal_controller_benchmark/ma2c_retrained_group06_demand_sw_to_ne_speed_raw.csv",
#         "color": "#ff7f0e" # Matplotlib tab:orange
#     },
#     {
#         "algo_name": "IQL-LR",
#         "baseline_csv": "signal_controller_benchmark/iqll_marl_group07_demand_ne_to_sw_speed_raw.csv",
#         "retrained_csv": "signal_controller_benchmark/iqll_retrained_group06_demand_sw_to_ne_speed_raw.csv",
#         "color": "#2ca02c" # Matplotlib tab:green
#     },
#     {
#         "algo_name": "PPO",
#         "baseline_csv": "signal_controller_benchmark/ppo_marl_group02_demand_w_to_e_speed_raw.csv",
#         "retrained_csv": "signal_controller_benchmark/ppo_retrained_group05_demand_se_to_nw_speed_raw.csv",
#         "color": "#d62728" # Matplotlib tab:red
#     }
# ]
OUTPUT_DIR = RUNS_EVAL_DIR / "manual_comparisons"

OUTPUT_FILENAME = "All_Algorithms_Queue_Comparison.png"
PLOT_TITLE = "Worst-case Queue Comparison (All Algorithms) in 5x5 Grid"
Y_LABEL = "Average vehicle queue length (vehicles)"

DATA_PAIRS = [
    {
        "algo_name": "IA2C",
        "baseline_csv": "signal_controller_benchmark/ia2c_marl_group02_demand_w_to_e_queue_raw.csv",
        "retrained_csv": "signal_controller_benchmark/ia2c_retrained_group02_demand_w_to_e_queue_raw.csv",
        "color": "#1f77b4" # Matplotlib tab:blue
    },
    {
        "algo_name": "MA2C",
        "baseline_csv": "signal_controller_benchmark/ma2c_marl_group02_demand_w_to_e_queue_raw.csv",
        "retrained_csv": "signal_controller_benchmark/ma2c_retrained_group06_demand_sw_to_ne_queue_raw.csv",
        "color": "#ff7f0e" # Matplotlib tab:orange
    },
    {
        "algo_name": "IQL-LR",
        "baseline_csv": "signal_controller_benchmark/iqll_marl_group02_demand_w_to_e_queue_raw.csv",
        "retrained_csv": "signal_controller_benchmark/iqll_retrained_group02_demand_w_to_e_queue_raw.csv",
        "color": "#2ca02c" # Matplotlib tab:green
    },
    {
        "algo_name": "PPO",
        "baseline_csv": "signal_controller_benchmark/ppo_marl_group02_demand_w_to_e_queue_raw.csv",
        "retrained_csv": "signal_controller_benchmark/ppo_retrained_group05_demand_se_to_nw_queue_raw.csv",
        "color": "#d62728" # Matplotlib tab:red
    }
]


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _resolve_csv(csv_ref: str) -> Path:
    candidate = Path(csv_ref).expanduser()

    if candidate.is_absolute():
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError("CSV path does not exist: %s" % candidate)

    direct_in_runs_eval = RUNS_EVAL_DIR / candidate
    if direct_in_runs_eval.exists():
        return direct_in_runs_eval.resolve()

    direct_in_project = PROJECT_ROOT / candidate
    if direct_in_project.exists():
        return direct_in_project.resolve()

    if candidate.parent != Path("."):
        scoped_root = RUNS_EVAL_DIR / candidate.parent
        if scoped_root.exists() and scoped_root.is_dir():
            scoped_matches = sorted(scoped_root.rglob(candidate.name))
            if len(scoped_matches) == 1:
                return scoped_matches[0].resolve()
            if len(scoped_matches) > 1:
                rel_matches = [str(path.relative_to(RUNS_EVAL_DIR)) for path in scoped_matches]
                raise ValueError(
                    "Ambiguous CSV '%s' inside %s. Matches:\n- %s"
                    % (csv_ref, candidate.parent, "\n- ".join(rel_matches))
                )

    matches = sorted(RUNS_EVAL_DIR.rglob(candidate.name))
    if not matches:
        raise FileNotFoundError(
            "Could not find '%s' under %s" % (csv_ref, RUNS_EVAL_DIR)
        )
    if len(matches) > 1:
        rel_matches = [str(path.relative_to(RUNS_EVAL_DIR)) for path in matches]
        raise ValueError(
            "Ambiguous CSV '%s'. Multiple matches under runs_eval:\n- %s\n"
            "Use a relative path like signal_controller_benchmark/..."
            % (csv_ref, "\n- ".join(rel_matches))
        )
    return matches[0].resolve()


def _load_raw_stats(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if "t" not in df.columns:
        raise ValueError("CSV missing required 't' column: %s" % csv_path)

    rollout_cols = [col for col in df.columns if col != "t"]
    if not rollout_cols:
        raise ValueError("CSV has no rollout columns: %s" % csv_path)

    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
    values = df[rollout_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if values.ndim != 2:
        raise ValueError("Unexpected CSV shape in %s" % csv_path)

    mean = np.nanmean(values, axis=1)
    vmin = np.nanmin(values, axis=1)
    vmax = np.nanmax(values, axis=1)
    return t, mean, vmin, vmax


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_FILENAME

    plt.figure(figsize=(14, 8))

    for pair in DATA_PAIRS:
        algo = pair["algo_name"]
        color = pair["color"]
        base_csv_str = pair["baseline_csv"]
        retrain_csv_str = pair["retrained_csv"]
        
        # 使用正则表达式动态提取 group 名称 (例如 "group10")
        base_group_match = re.search(r'(group\d+)', base_csv_str)
        retrain_group_match = re.search(r'(group\d+)', retrain_csv_str)
        
        base_group = base_group_match.group(1) if base_group_match else "unknown"
        retrain_group = retrain_group_match.group(1) if retrain_group_match else "unknown"

        # 构造你想要的图例标签，自动注入算法名和组名
        base_label = f"Baseline {algo} Worst Group ({base_group})"
        retrain_label = f"Retrained {algo} Worst Group ({retrain_group})"
        
        base_path = _resolve_csv(base_csv_str)
        retrain_path = _resolve_csv(retrain_csv_str)

        b_t, b_mean, b_min, b_max = _load_raw_stats(base_path)
        r_t, r_mean, r_min, r_max = _load_raw_stats(retrain_path)

        # Baseline 画图
        plt.plot(
            b_t, b_mean, 
            label=base_label, 
            color=color, 
            linestyle='--', 
            linewidth=1.5, 
            alpha=0.6
        )
        plt.fill_between(b_t, b_min, b_max, color=color, alpha=0.1)

        # Retrained 画图
        plt.plot(
            r_t, r_mean, 
            label=retrain_label, 
            color=color, 
            linestyle='-', 
            linewidth=2.5, 
            alpha=1.0
        )
        plt.fill_between(r_t, r_min, r_max, color=color, alpha=0.15)
        
        print(f"Processed {algo}: Baseline({base_group}) vs Retrained({retrain_group})")

    plt.xlabel("Simulation second")
    plt.ylabel(Y_LABEL)
    plt.title(PLOT_TITLE)
    
    # 因为图例变长了，可以放在图外或通过缩小字体来适应
    plt.legend(ncol=2, loc="upper left")
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("========================================")
    print("Saved combined plot to:", output_path)


if __name__ == "__main__":
    main()
