#!/usr/bin/env python3
"""Manually generate Absolute Value Heatmaps from raw eval CSVs.
Generates a combined 1x2 heatmap plot plotting raw MARL and DR-MARL values.
Optimized for a 10x6 figure size.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        "font.size": 10,           # 全局基准字号变小
        "axes.labelsize": 11,
        "axes.titlesize": 12,      # 子图标题字号变小
        "xtick.labelsize": 6,      # X轴(算法名称)字号变小
        "ytick.labelsize": 10,     # Y轴(Group 0-11)字号变小
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
OUTPUT_DIR = Path("runs_eval/manual_comparisons_real")
# OUTPUT_DIR = Path("runs_eval/manual_comparisons")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    _configure_plot_style()

    file_path = Path('runs_eval') / 'signal_controller_benchmark_real' / 'full_performance_comparison_real.xlsx'
    # file_path = Path('runs_eval') / 'signal_controller_benchmark' / 'full_performance_comparison.xlsx'
    if not Path(file_path).exists():
        file_path = 'full_performance_comparison_real.xlsx - Sheet1.csv'
        if not Path(file_path).exists():
            print(f"Error: 找不到数据文件 {file_path}")
            return
        
    df = pd.read_excel(file_path, header=None) if str(file_path).endswith('.xlsx') else pd.read_csv(file_path, header=None)

    col_names = []
    for alg in ALGORITHMS:
        col_names.extend([f"{alg}\nMARL", f"{alg}\nRetrained"])  # DR-MARL 改为 Retrained，避免在 10x6 下过长导致拥挤
        

    group_labels = [f"Group {i}" for i in range(12)]

    queue_cols = [1, 2, 7, 8, 13, 14, 19, 20]
    queue_data = df.iloc[2:-1, queue_cols].dropna().astype(float)
    queue_data.columns = col_names
    queue_data.index = group_labels

    speed_cols = [4, 5, 10, 11, 16, 17, 22, 23]
    speed_data = df.iloc[2:-1, speed_cols].dropna().astype(float)
    speed_data.columns = col_names
    speed_data.index = group_labels

    # --- 关键修改 1: 尺寸调整为 10x6 ---
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), dpi=150)

    # --- 关键修改 2: 减小 annot_kws={"size": 7.5}，防止热图格子内的数字溢出 ---
    sns.heatmap(queue_data, annot=True, fmt=".1f", cmap="OrRd", ax=ax1, 
                cbar_kws={'label': 'Queue Length (veh)'}, annot_kws={"size": 7.5}, 
                linewidths=0.5, linecolor='white')
    
    # 标题使用 \n 换行，防止在 10x6 下拥挤
    ax1.set_title('Absolute Queue Length\n(Lower is Better)', pad=10)
    ax1.set_ylabel('Demand Groups')

    sns.heatmap(speed_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax2, 
                cbar_kws={'label': 'Average Speed (m/s)'}, annot_kws={"size": 7.5}, 
                linewidths=0.5, linecolor='white')
    ax2.set_title('Absolute Average Speed\n(Higher is Better)', pad=10)

    fig.suptitle('Horizon- and Rollout-averaged Performance Comparison by Groups: MARL vs. DR-MARL in Monaco City', 
                 fontsize=12, fontweight='bold', y=0.98) # 抬高 y 值留出空间
    # fig.suptitle('Horizon- and Rollout-averaged Performance Comparison by Groups: MARL vs. DR-MARL in 5x5 Grid', 
    #              fontsize=12, fontweight='bold', y=0.98)

    for ax in [ax1, ax2]:
        for x_pos in [2, 4, 6]:
            # 分隔线变细一点，不那么抢眼
            ax.axvline(x=x_pos, color='gray', linewidth=1.5, linestyle='--', alpha=0.7)
        # 强制 x 轴不倾斜，保持紧凑
        ax.tick_params(axis='x', rotation=0)
            
    plt.tight_layout()

    # Save to directory
    output_path = OUTPUT_DIR / 'Absolute_Heatmap_Comparison_real_optimized.png'
    # output_path = OUTPUT_DIR / 'Absolute_Heatmap_Comparison_optimized.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"========================================")
    print(f"Saved optimized 10x6 absolute value heatmap to: {output_path}")

if __name__ == "__main__":
    main()