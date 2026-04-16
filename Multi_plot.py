#!/usr/bin/env python3
"""
为 IEEE 论文设计的双栏并排绘图脚本。
优化了标题重叠、图例拥挤问题，并将中心虚线延伸至图例底部。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
import pandas as pd

_MPLCONFIGDIR = Path("/tmp/matplotlib")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

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
        "font.size": 8,             
        "axes.labelsize": 9,        
        "axes.titlesize": 9,       
        "xtick.labelsize": 8,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,       # 略微调大一点，因为我们增加了图片高度
        "figure.titlesize": 9,     # 共享大标题加大
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    missing_bins = [exe for exe in ("latex", "kpsewhich", "dvipng") if shutil.which(exe) is None]
    required_sty = ("newtxtext.sty", "newtxmath.sty", "type1ec.sty")
    missing_sty = [sty for sty in required_sty if not _tex_package_available(sty)]

    if missing_bins or missing_sty:
        params["text.usetex"] = False
        params.pop("text.latex.preamble", None)
        warnings.warn("Falling back to Matplotlib rendering (TeX missing).", RuntimeWarning)

    mpl.rcParams.update(params)

_configure_plot_style()

# ===========================
# 数据与配置
# ===========================
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_EVAL_DIR = PROJECT_ROOT / "runs_eval"

# 将子图标题缩短，避免左右重叠。具体的场景名称会由 suptitle 统一接管。
PLOT_CONFIGS = [
    {
        "output_dir": "manual_comparisons_real",
        "output_filename": "All_Algorithms_Queue_Comparison_real.png",
        "plot_title": "Queue Length",
        "y_label": "Total queued vehicles (veh)",
        "data_pairs": [
            {"algo_name": "IA2C", "baseline_csv": "signal_controller_benchmark_real/ia2c_real_group10_uniform_queue_raw.csv", "retrained_csv": "signal_controller_benchmark_real/ia2c_retrained_group01_s_to_n_queue_raw.csv", "color": "#1f77b4"},
            {"algo_name": "MA2C", "baseline_csv": "signal_controller_benchmark_real/ma2c_real_group01_s_to_n_queue_raw.csv", "retrained_csv": "signal_controller_benchmark_real/ma2c_retrained_group01_s_to_n_queue_raw.csv", "color": "#ff7f0e"},
            {"algo_name": "IQL-LR", "baseline_csv": "signal_controller_benchmark_real/iqll_real_group10_uniform_queue_raw.csv", "retrained_csv": "signal_controller_benchmark_real/iqll_retrained_group05_se_to_nw_queue_raw.csv", "color": "#2ca02c"},
            {"algo_name": "PPO", "baseline_csv": "signal_controller_benchmark_real/ppo_real_group10_uniform_queue_raw.csv", "retrained_csv": "signal_controller_benchmark_real/ppo_retrained_group01_s_to_n_queue_raw.csv", "color": "#d62728"}
        ]
    },
    {
        "output_dir": "manual_comparisons_real",
        "output_filename": "All_Algorithms_Speed_Comparison_real.png",
        "plot_title": "Average Speed",
        "y_label": "Average vehicle speed (km/h)",
        "data_pairs": [
            {"algo_name": "IA2C", "baseline_csv": "signal_controller_benchmark_real/ia2c_real_group10_uniform_speed_raw.csv", "retrained_csv": "signal_controller_benchmark_real/ia2c_retrained_group01_s_to_n_speed_raw.csv", "color": "#1f77b4"},
            {"algo_name": "MA2C", "baseline_csv": "signal_controller_benchmark_real/ma2c_real_group01_s_to_n_speed_raw.csv", "retrained_csv": "signal_controller_benchmark_real/ma2c_retrained_group01_s_to_n_speed_raw.csv", "color": "#ff7f0e"},
            {"algo_name": "IQL-LR", "baseline_csv": "signal_controller_benchmark_real/iqll_real_group10_uniform_speed_raw.csv", "retrained_csv": "signal_controller_benchmark_real/iqll_retrained_group05_se_to_nw_speed_raw.csv", "color": "#2ca02c"},
            {"algo_name": "PPO", "baseline_csv": "signal_controller_benchmark_real/ppo_real_group10_uniform_speed_raw.csv", "retrained_csv": "signal_controller_benchmark_real/ppo_retrained_group01_s_to_n_speed_raw.csv", "color": "#d62728"}
        ]
    },
    {
        "output_dir": "manual_comparisons",
        "output_filename": "All_Algorithms_Speed_Comparison.png",
        "plot_title": "Average Speed",
        "y_label": "Average vehicle speed (km/h)",
        "data_pairs": [
            {"algo_name": "IA2C", "baseline_csv": "signal_controller_benchmark/ia2c_marl_group02_demand_w_to_e_speed_raw.csv", "retrained_csv": "signal_controller_benchmark/ia2c_retrained_group06_demand_sw_to_ne_speed_raw.csv", "color": "#1f77b4"},
            {"algo_name": "MA2C", "baseline_csv": "signal_controller_benchmark/ma2c_marl_group06_demand_sw_to_ne_speed_raw.csv", "retrained_csv": "signal_controller_benchmark/ma2c_retrained_group06_demand_sw_to_ne_speed_raw.csv", "color": "#ff7f0e"},
            {"algo_name": "IQL-LR", "baseline_csv": "signal_controller_benchmark/iqll_marl_group07_demand_ne_to_sw_speed_raw.csv", "retrained_csv": "signal_controller_benchmark/iqll_retrained_group06_demand_sw_to_ne_speed_raw.csv", "color": "#2ca02c"},
            {"algo_name": "PPO", "baseline_csv": "signal_controller_benchmark/ppo_marl_group02_demand_w_to_e_speed_raw.csv", "retrained_csv": "signal_controller_benchmark/ppo_retrained_group05_demand_se_to_nw_speed_raw.csv", "color": "#d62728"}
        ]
    },
    {
        "output_dir": "manual_comparisons",
        "output_filename": "All_Algorithms_Queue_Comparison.png",
        "plot_title": "Queue Length",
        "y_label": "Average vehicle queue length (vehicles)",
        "data_pairs": [
            {"algo_name": "IA2C", "baseline_csv": "signal_controller_benchmark/ia2c_marl_group02_demand_w_to_e_queue_raw.csv", "retrained_csv": "signal_controller_benchmark/ia2c_retrained_group02_demand_w_to_e_queue_raw.csv", "color": "#1f77b4"},
            {"algo_name": "MA2C", "baseline_csv": "signal_controller_benchmark/ma2c_marl_group02_demand_w_to_e_queue_raw.csv", "retrained_csv": "signal_controller_benchmark/ma2c_retrained_group06_demand_sw_to_ne_queue_raw.csv", "color": "#ff7f0e"},
            {"algo_name": "IQL-LR", "baseline_csv": "signal_controller_benchmark/iqll_marl_group02_demand_w_to_e_queue_raw.csv", "retrained_csv": "signal_controller_benchmark/iqll_retrained_group02_demand_w_to_e_queue_raw.csv", "color": "#2ca02c"},
            {"algo_name": "PPO", "baseline_csv": "signal_controller_benchmark/ppo_marl_group02_demand_w_to_e_queue_raw.csv", "retrained_csv": "signal_controller_benchmark/ppo_retrained_group05_demand_se_to_nw_queue_raw.csv", "color": "#d62728"}
        ]
    }
]

# 组合场景并定义全局大标题
PAIRED_SCENARIOS = [
    {
        "suptitle": "Worst-case Performance Comparison in Monaco City",
        "out_name": "Monaco_City_SideBySide.png",
        "queue_cfg": PLOT_CONFIGS[0],
        "speed_cfg": PLOT_CONFIGS[1]
    },
    {
        "suptitle": "Worst-case Performance Comparison in 5x5 Grid",
        "out_name": "5x5_Grid_SideBySide.png",
        "queue_cfg": PLOT_CONFIGS[3],
        "speed_cfg": PLOT_CONFIGS[2]
    }
]

def _resolve_csv(csv_ref: str) -> Path:
    candidate = Path(csv_ref).expanduser()
    if candidate.is_absolute() and candidate.exists(): return candidate.resolve()
    
    for root in [RUNS_EVAL_DIR, PROJECT_ROOT]:
        path = root / candidate
        if path.exists(): return path.resolve()
    
    matches = sorted(RUNS_EVAL_DIR.rglob(candidate.name))
    if not matches: raise FileNotFoundError(f"Missing: {csv_ref}")
    return matches[0].resolve()

def _load_raw_stats(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    rollout_cols = [c for c in df.columns if c != "t"]
    t = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=float)
    values = df[rollout_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return t, np.nanmean(values, axis=1), np.nanmin(values, axis=1), np.nanmax(values, axis=1)

def plot_single_axis(ax, config):
    for pair in config["data_pairs"]:
        algo = pair["algo_name"]
        color = pair["color"]
        
        base_path = _resolve_csv(pair["baseline_csv"])
        retrain_path = _resolve_csv(pair["retrained_csv"])
        
        bt, b_mean, b_min, b_max = _load_raw_stats(base_path)
        rt, r_mean, r_min, r_max = _load_raw_stats(retrain_path)

        bg_match = re.search(r'(group\d+)', pair["baseline_csv"])
        rg_match = re.search(r'(group\d+)', pair["retrained_csv"])
        bg = bg_match.group(1) if bg_match else "unk"
        rg = rg_match.group(1) if rg_match else "unk"

        base_label = f"Base {algo} ({bg})"
        retrain_label = f"Retrained {algo} ({rg})"

        # 绘制 Baseline
        ax.plot(bt, b_mean, color=color, linestyle='--', linewidth=1.0, alpha=0.8, label=base_label)
        ax.fill_between(bt, b_min, b_max, color=color, alpha=0.1)
        
        # 绘制 Retrained
        ax.plot(rt, r_mean, color=color, linestyle='-', linewidth=1.5, alpha=1.0, label=retrain_label)
        ax.fill_between(rt, r_min, r_max, color=color, alpha=0.15)
        
    ax.set_title(config["plot_title"], pad=8)
    ax.set_ylabel(config["y_label"])
    ax.set_xlabel("Simulation second (s)")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 强制在当前子图下方绘制独立的图例 (2列排列)
    # 进一步下移 bbox_to_anchor 以免挡住 X 轴标签，增加 columnspacing 让排版不拥挤
    ax.legend(
        ncol=2, 
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        handlelength=1.5,
        columnspacing=1.0 
    )

def main():
    output_dir = RUNS_EVAL_DIR / "ieee_comparisons_paired"
    output_dir.mkdir(parents=True, exist_ok=True)

    for scenario in PAIRED_SCENARIOS:
        # 增加总高度至 4.8，为底部的多行图例留出充足空间
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
        
        # 添加共享的主标题，位置稍微靠上
        fig.suptitle(scenario["suptitle"], y=0.96, fontweight='bold')

        # 绘制左右子图
        plot_single_axis(ax1, scenario["queue_cfg"])
        plot_single_axis(ax2, scenario["speed_cfg"])

        # 在两个子图正中间绘制延伸的虚线分隔线
        # Y的范围拉伸到从图例最底端(0.02) 到 主标题下方(0.9)
        separator_line = lines.Line2D(
            [0.5, 0.5], [0.25, 0.90], 
            transform=fig.transFigure, 
            color="black", 
            linestyle="--", 
            linewidth=1.0, 
            alpha=0.3
        )
        fig.add_artist(separator_line)

        # 调整布局比例
        # bottom 设为 0.45 给下方的图例让出近一半的空间，wspace=0.3 让左右两图隔开一点以免互相侵占空间
        plt.subplots_adjust(bottom=0.35, top=0.88, wspace=0.3)
        
        out_path = output_dir / scenario["out_name"]
        plt.savefig(out_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Generated paired plot: {out_path}")

if __name__ == "__main__":
    main()