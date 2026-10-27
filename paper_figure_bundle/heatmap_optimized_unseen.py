#!/usr/bin/env python3
"""Manually generate Absolute Value Heatmaps from raw eval CSVs.
Generates combined 1x2 heatmap plots for both Monaco City and 5x5 Grid scenarios.
Optimized for speed and 10x6 figure size with Group 11 highlighted.
"""

import os
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

# 2. 字体和排版配置 (禁用外部 LaTeX 渲染以大幅提升速度，但保留学术字体风格)
def _configure_plot_style() -> None:
    params = {
        "text.usetex": False,  # 核心加速点：禁用外部 LaTeX，使用内置渲染
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"], # 替代 newtxtext
        "mathtext.fontset": "stix", # 替代 newtxmath 的数学字体
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 6,
        "ytick.labelsize": 10,
        "figure.titlesize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    mpl.rcParams.update(params)

ALGORITHMS = ["IA2C", "MA2C", "IQL-LR", "PPO"]

PLOT_CONFIGS = [
    {
        "file_path": Path('runs_eval/signal_controller_benchmark_real/full_performance_comparison_real.xlsx'),
        "fallback_csv": Path('runs_eval/signal_controller_benchmark_real/full_performance_comparison_real.xlsx - Sheet1.csv'),
        "output_dir": Path("runs_eval/manual_comparisons_real"),
        "title": 'Horizon- and Rollout-averaged Performance Comparison by Groups: MARL vs. DR-MARL in Monaco City',
        "out_name": 'Absolute_Heatmap_Comparison_real_optimized.png'
    },
    {
        "file_path": Path('runs_eval/signal_controller_benchmark/full_performance_comparison.xlsx'),
        "fallback_csv": Path('runs_eval/signal_controller_benchmark/full_performance_comparison.xlsx - Sheet1.csv'),
        "output_dir": Path("runs_eval/manual_comparisons"),
        "title": 'Horizon- and Rollout-averaged Performance Comparison by Groups: MARL vs. DR-MARL in 5x5 Grid',
        "out_name": 'Absolute_Heatmap_Comparison_optimized.png'
    }
]

def process_scenario(config: dict) -> None:
    file_path = config["file_path"]
    fallback = config["fallback_csv"]
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # 核心加速点：优先读取读取速度极快的 CSV，如果没有 CSV 才尝试读取慢速的 Excel
    if fallback.exists():
        df = pd.read_csv(fallback, header=None)
    elif file_path.exists():
        if str(file_path).endswith('.xlsx'):
            df = pd.read_excel(file_path, header=None, engine='openpyxl')
        else:
            df = pd.read_csv(file_path, header=None)
    else:
        print(f"Error: Data file not found for {config['out_name']}. Checked {file_path} and {fallback}")
        return

    col_names = []
    for alg in ALGORITHMS:
        col_names.extend([f"{alg}\nMARL", f"{alg}\nRetrained"])
        
    group_labels = [f"Group {i}" for i in range(1, 12)]
    group_labels.append("Group 12\n(Unseen)")

    queue_cols = [1, 2, 7, 8, 13, 14, 19, 20]
    queue_data = df.iloc[2:-1, queue_cols].dropna().astype(float)
    queue_data.columns = col_names
    queue_data.index = group_labels

    speed_cols = [4, 5, 10, 11, 16, 17, 22, 23]
    speed_data = df.iloc[2:-1, speed_cols].dropna().astype(float)
    speed_data.columns = col_names
    speed_data.index = group_labels

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(4.3, 7.5), dpi=150)

    # 使用内置字体渲染，速度会非常快
    sns.heatmap(queue_data, annot=True, fmt=".1f", cmap="OrRd", ax=ax1, 
                cbar_kws={'label': 'Queue Length (veh)'}, annot_kws={"size": 7.5}, 
                linewidths=0.5, linecolor='white')
    
    ax1.set_title('Absolute Queue Length (Lower is Better)', fontsize=10, pad=5)
    ax1.set_ylabel('Demand Groups')
    ax1.set_xticklabels([])
    ax1.set_xlabel('')

    sns.heatmap(speed_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax2, 
                cbar_kws={'label': 'Average Speed (m/s)'}, annot_kws={"size": 7.5}, 
                linewidths=0.5, linecolor='white')
    ax2.set_title('Absolute Average Speed (Higher is Better)', fontsize=10, pad=5)
    ax2.set_ylabel('Demand Groups')

    for ax in [ax1, ax2]:
        for x_pos in [2, 4, 6]:
            ax.axvline(x=x_pos, color='gray', linewidth=1.5, linestyle='--', alpha=0.7)
            
        rect = patches.Rectangle((0, 11), len(col_names), 1, linewidth=2.5, 
                                 edgecolor='#e74c3c', facecolor='none', zorder=10)
        ax.add_patch(rect)
        
    ax2.tick_params(axis='x', rotation=0)
            
    plt.tight_layout()

    output_path = output_dir / config["out_name"]
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig) # 显式传入 fig 以确保资源正确释放
    
    print(f"Saved optimized 10x6 heatmap to: {output_path}")

def main():
    _configure_plot_style()
    
    print("========================================")
    for config in PLOT_CONFIGS:
        process_scenario(config)
    print("========================================")

if __name__ == "__main__":
    main()