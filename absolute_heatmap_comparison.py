#!/usr/bin/env python3
"""Manually generate Absolute Value Heatmaps from raw eval CSVs.
Generates a combined 1x2 heatmap plot plotting raw MARL and DR-MARL values.
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

# 2. 字体和排版配置 (基于您的 LaTeX 设置，稍微减小字号以容纳 8 列)
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
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 15,
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

    # 创建列名：每个算法有 MARL 和 DR-MARL 两列
    col_names = []
    for alg in ALGORITHMS:
        # 为了排版美观，使用 \n 换行，并简写 DR-MARL 为 DR
        col_names.extend([f"{alg}\nMARL", f"{alg}\nRetrained"]) 

    group_labels = [f"Group {i}" for i in range(12)]

    # 提取 Queue Length 的绝对值数据 (列索引 1,2 对应 IA2C, 7,8 对应 MA2C, 等等)
    queue_cols = [1, 2, 7, 8, 13, 14, 19, 20]
    queue_data = df.iloc[2:-1, queue_cols].dropna().astype(float)
    queue_data.columns = col_names
    queue_data.index = group_labels

    # 提取 Avg Speed 的绝对值数据 (列索引 4,5 对应 IA2C, 10,11 对应 MA2C, 等等)
    speed_cols = [4, 5, 10, 11, 16, 17, 22, 23]
    speed_data = df.iloc[2:-1, speed_cols].dropna().astype(float)
    speed_data.columns = col_names
    speed_data.index = group_labels

    # --- 尺寸设为14x7 以便装下 8 列数据 ---
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), dpi=150)

    # 1. 绘制 Queue Length 绝对值热图
    # 排队长度越低越好，使用 OrRd (深红代表排队严重，浅色代表畅通)
    sns.heatmap(queue_data, annot=True, fmt=".1f", cmap="OrRd", ax=ax1, 
                cbar_kws={'label': 'Queue Length (veh)'}, annot_kws={"size": 10}, 
                linewidths=0.5, linecolor='white')
    ax1.set_title('Absolute Queue Length (Lower is Better)', pad=15)
    ax1.set_ylabel('Demand Groups')

    # 2. 绘制 Avg Speed 绝对值热图
    # 速度越高越好，使用 YlGnBu (深蓝/绿代表速度快，浅黄代表速度慢)
    sns.heatmap(speed_data, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax2, 
                cbar_kws={'label': 'Average Speed (m/s)'}, annot_kws={"size": 10}, 
                linewidths=0.5, linecolor='white')
    ax2.set_title('Absolute Average Speed (Higher is Better)', pad=15)

    # 添加全局大标题
    fig.suptitle('Absolute Performance Comparison by Groups: MARL vs. DR-MARL in Monaco City', 
                 fontsize=15, fontweight='bold', y=1.02)
        # 添加全局大标题
    # fig.suptitle('Absolute Performance Comparison by Groups: MARL vs. DR-MARL in 5x5 Grid', 
    #              fontsize=15, fontweight='bold', y=1.02)

    # 优化：为每个热图添加垂直辅助虚线，把 4 个基础算法分隔开
    for ax in [ax1, ax2]:
        for x_pos in [2, 4, 6]:
            ax.axvline(x=x_pos, color='gray', linewidth=2, linestyle='--', alpha=0.7)
            
    plt.tight_layout()

    # Save to directory
    output_path = OUTPUT_DIR / 'Absolute_Heatmap_Comparison_real.png'
    # output_path = OUTPUT_DIR / 'Absolute_Heatmap_Comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"========================================")
    print(f"Saved absolute value heatmap to: {output_path}")

if __name__ == "__main__":
    main()