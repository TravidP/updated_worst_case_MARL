#!/usr/bin/env python3
"""Manually generate Absolute Value Heatmaps from raw eval CSVs.
Generates combined 1x2 heatmap plots for both Monaco City and 5x5 Grid scenarios.
Optimized for a 10x6 figure size with Group 11 highlighted.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

# 1. 确保在没有X11服务器时Matplotlib不会报错
_MPLCONFIGDIR = Path("/tmp/matplotlib")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib as mpl
mpl.use("Agg")

# 2. 字体和排版配置 (针对 10x6 尺寸，全面下调了字号以防止文字重叠)
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
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 6,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
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

ALGORITHMS = ["IA2C", "MA2C", "IQL-LR", "PPO"]

# Configuration for plotting both scenarios sequentially
PLOT_CONFIGS = [
    {
        "file_path": Path('runs_eval/signal_controller_benchmark_real/full_performance_comparison_real.xlsx'),
        "fallback_csv": "full_performance_comparison_real.xlsx - Sheet1.csv",
        "output_dir": Path("runs_eval/manual_comparisons_real"),
        "title": 'Horizon- and Rollout-averaged Performance Comparison by Groups: MARL vs. DR-MARL in Monaco City',
        "out_name": 'Absolute_Heatmap_Comparison_real_optimized.png'
    },
    {
        "file_path": Path('runs_eval/signal_controller_benchmark/full_performance_comparison.xlsx'),
        "fallback_csv": "full_performance_comparison.xlsx - Sheet1.csv",
        "output_dir": Path("runs_eval/manual_comparisons"),
        "title": 'Horizon- and Rollout-averaged Performance Comparison by Groups: MARL vs. DR-MARL in 5x5 Grid',
        "out_name": 'Absolute_Heatmap_Comparison_optimized.png'
    }
]

def process_scenario(config: dict) -> None:
    file_path = config["file_path"]
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check file existence and fallbacks
    if not file_path.exists():
        fallback = Path(config["fallback_csv"])
        if fallback.exists():
            file_path = fallback
        else:
            print(f"Error: Data file not found for {config['out_name']}. Checked {file_path} and {fallback}")
            return
        
    df = pd.read_excel(file_path, header=None) if str(file_path).endswith('.xlsx') else pd.read_csv(file_path, header=None)

    col_names = []
    for alg in ALGORITHMS:
        col_names.extend([f"{alg}\nMARL", f"{alg}\nRetrained"])
        
    # Group labels: Add a specific note to the Y-axis for Group 11
    group_labels = [f"Group {i}" for i in range(11)]
    group_labels.append("Group 11\n(Unseen)")
    # group_labels.append(
    #     r"Group 11" + "\n" + 
    #     r"{\small (Unseen)}"
    # )
    # group_labels.append("Group 11")

    queue_cols = [1, 2, 7, 8, 13, 14, 19, 20]
    queue_data = df.iloc[2:-1, queue_cols].dropna().astype(float)
    queue_data.columns = col_names
    queue_data.index = group_labels

    speed_cols = [4, 5, 10, 11, 16, 17, 22, 23]
    speed_data = df.iloc[2:-1, speed_cols].dropna().astype(float)
    speed_data.columns = col_names
    speed_data.index = group_labels

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), dpi=150)

    sns.heatmap(queue_data, annot=True, fmt=".1f", cmap="OrRd", ax=ax1, 
                cbar_kws={'label': 'Queue Length (veh)'}, annot_kws={"size": 7.5}, 
                linewidths=0.5, linecolor='white')
    
    ax1.set_title('Absolute Queue Length\n(Lower is Better)', pad=10)
    ax1.set_ylabel('Demand Groups')

    sns.heatmap(speed_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax2, 
                cbar_kws={'label': 'Average Speed (m/s)'}, annot_kws={"size": 7.5}, 
                linewidths=0.5, linecolor='white')
    ax2.set_title('Absolute Average Speed\n(Higher is Better)', pad=10)

    fig.suptitle(config["title"], fontsize=12, fontweight='bold', y=0.98) 

    for ax in [ax1, ax2]:
        for x_pos in [2, 4, 6]:
            ax.axvline(x=x_pos, color='gray', linewidth=1.5, linestyle='--', alpha=0.7)
            
        # Add a prominent bounding box around Group 11 (which is at index 11)
        # The coordinates are (x, y) where x=0 and y=11. Width spans all columns, height is 1 row.
        rect = patches.Rectangle((0, 11), len(col_names), 1, linewidth=2.5, 
                                 edgecolor='#e74c3c', facecolor='none', zorder=10)
        ax.add_patch(rect)
        
        ax.tick_params(axis='x', rotation=0)
            
    plt.tight_layout()

    output_path = output_dir / config["out_name"]
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved optimized 10x6 heatmap to: {output_path}")

def main():
    _configure_plot_style()
    
    print("========================================")
    for config in PLOT_CONFIGS:
        process_scenario(config)
    print("========================================")

if __name__ == "__main__":
    main()