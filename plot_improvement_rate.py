#!/usr/bin/env python3
"""Plot Relative Improvement Percentage (Reduction in Queue Length).
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
        "legend.fontsize": 12,
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

    mpl.rcParams.update(params)

_configure_plot_style()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===========================
# Configuration block
# ===========================
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_EVAL_DIR = PROJECT_ROOT / "runs_eval"
OUTPUT_DIR = RUNS_EVAL_DIR / "manual_comparisons_real"

OUTPUT_FILENAME = "All_Algorithms_Improvement_Rate.png"
PLOT_TITLE = "Relative Reduction in Queue Length (%) in the Monaco City dataset"
Y_LABEL = "Improvement (%)"

DATA_PAIRS = [
    {
        "algo_name": "IA2C",
        "baseline_csv": "signal_controller_benchmark_real/ia2c_real_group10_uniform_queue_raw.csv",
        "retrained_csv": "signal_controller_benchmark_real/ia2c_retrained_group01_s_to_n_queue_raw.csv",
        "color": "#1f77b4" # Matplotlib tab:blue
    },
    {
        "algo_name": "MA2C",
        "baseline_csv": "signal_controller_benchmark_real/ma2c_real_group01_s_to_n_queue_raw.csv",
        "retrained_csv": "signal_controller_benchmark_real/ma2c_retrained_group01_s_to_n_queue_raw.csv",
        "color": "#ff7f0e" # Matplotlib tab:orange
    },
    {
        "algo_name": "IQL-LR",
        "baseline_csv": "signal_controller_benchmark_real/iqll_real_group10_uniform_queue_raw.csv",
        "retrained_csv": "signal_controller_benchmark_real/iqll_retrained_group05_se_to_nw_queue_raw.csv",
        "color": "#2ca02c" # Matplotlib tab:green
    },
    {
        "algo_name": "PPO",
        "baseline_csv": "signal_controller_benchmark_real/ppo_real_group10_uniform_queue_raw.csv",
        "retrained_csv": "signal_controller_benchmark_real/ppo_retrained_group01_s_to_n_queue_raw.csv",
        "color": "#d62728" # Matplotlib tab:red
    }
]

def _resolve_csv(csv_ref: str) -> Path:
    candidate = Path(csv_ref).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    direct_in_runs_eval = RUNS_EVAL_DIR / candidate
    if direct_in_runs_eval.exists():
        return direct_in_runs_eval.resolve()
    direct_in_project = PROJECT_ROOT / candidate
    if direct_in_project.exists():
        return direct_in_project.resolve()
    
    matches = sorted(RUNS_EVAL_DIR.rglob(candidate.name))
    if not matches:
        raise FileNotFoundError(f"Could not find '{csv_ref}'")
    return matches[0].resolve()

def _load_raw_stats(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    rollout_cols = [col for col in df.columns if col != "t"]
    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
    values = df[rollout_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
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
        
        base_path = _resolve_csv(pair["baseline_csv"])
        retrain_path = _resolve_csv(pair["retrained_csv"])

        # 读取数据
        b_t, b_mean, _, _ = _load_raw_stats(base_path)
        r_t, r_mean, _, _ = _load_raw_stats(retrain_path)

        # ----------------------------------------------------
        # 计算提升率: Improvement = (Baseline - Retrained) / Baseline * 100
        # ----------------------------------------------------
        # 为避免仿真初期 (Baseline=0) 导致除以 0 的情况，我们加一个 Mask
        # 只计算 Baseline 排队车辆大于 2 的情况，其余时刻算作 0 提升
        valid_mask = b_mean > 2.0 
        
        improvement_pct = np.zeros_like(b_mean)
        improvement_pct[valid_mask] = ((b_mean[valid_mask] - r_mean[valid_mask]) / b_mean[valid_mask]) * 100.0
        
        # 计算该算法的总体平均提升率 (仅针对排队形成后的有效时段)
        avg_improvement = np.nanmean(improvement_pct[valid_mask])

        # 1. 绘制随时间变化的相对提升率折线 (实线)
        plt.plot(
            b_t, improvement_pct, 
            color=color, 
            linestyle='-', 
            linewidth=2.0, 
            alpha=0.85
        )
        
        # 2. 绘制该算法平均提升率的基准线 (虚线)
        # 图例标签中直接带上平均提升的数值百分比，直观明了
        plt.axhline(
            y=avg_improvement, 
            color=color, 
            linestyle='--', 
            linewidth=2.0, 
            alpha=0.7,
            label=f"{algo} (Avg: {avg_improvement:.1f}\%)"
        )
        
        print(f"Processed {algo} ... Average Improvement: {avg_improvement:.2f}%")

    plt.xlabel("Simulation second")
    plt.ylabel(Y_LABEL)
    plt.title(PLOT_TITLE)
    
    # 将图例分列显示
    plt.legend(ncol=2, loc="lower right") # 提升率图一般右上角线比较多，放右下角
    
    # 辅助 Y=0 基准线 (如果不为0意味着有负优化的情况)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # 限制 Y 轴的上下限，确保视觉美观 (如果存在短暂超过 100% 的异常点可以根据需要调整)
    plt.ylim(-20, 110) 
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("========================================")
    print("Saved improvement plot to:", output_path)

if __name__ == "__main__":
    main()