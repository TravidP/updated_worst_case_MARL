#!/usr/bin/env python3
"""Manually compare Box Plots from raw eval CSVs.
Generates a combined 1x2 plot for Queue Length and Average Speed.
Annotates mean values and percentage change.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# 1. 确保在没有X11服务器时Matplotlib不会报错
_MPLCONFIGDIR = Path("/tmp/matplotlib")
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib as mpl
mpl.use("Agg")

# 2. 字体和排版配置 (为了适应10x6，稍微减小了字号)
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
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
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

# 3. 颜色配置
DATA_PAIRS = [
    {"algo_name": "IA2C",   "color": "#1f77b4"}, 
    {"algo_name": "MA2C",   "color": "#ff7f0e"}, 
    {"algo_name": "IQL-LR", "color": "#2ca02c"}, 
    {"algo_name": "PPO",    "color": "#d62728"}  
]

# OUTPUT_DIR = Path("runs_eval/manual_comparisons_real")
OUTPUT_DIR = Path("runs_eval/manual_comparisons")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_boxplot_metric_on_ax(ax, df, metric_name, col_marl, col_drmarl):
    """在一个指定的 ax (子图) 上绘制箱线图，并标注平均值和变化幅度"""
    
    # data_marl = [df.iloc[2:, c].dropna().astype(float) for c in col_marl]
    # data_drmarl = [df.iloc[2:, c].dropna().astype(float) for c in col_drmarl]
    data_marl = [df.iloc[2:-1, c].dropna().astype(float) for c in col_marl]
    data_drmarl = [df.iloc[2:-1, c].dropna().astype(float) for c in col_drmarl]
    positions_marl = [1, 4, 7, 10]
    positions_drmarl = [2, 5, 8, 11]

    for i in range(4):
        color = DATA_PAIRS[i]['color']
        
        # # 计算平均值
        # mean_marl = data_marl[i].mean()
        # mean_drmarl = data_drmarl[i].mean()
        
        # base_val = mean_marl if mean_marl != 0 else 1e-5
        # pct_change = (mean_drmarl - mean_marl) / abs(base_val) * 100
        # 替换为直接从最后一行 (-1) 读取对应列的数据
        mean_marl = float(df.iloc[-1, col_marl[i]])
        mean_drmarl = float(df.iloc[-1, col_drmarl[i]])
        
        # 百分比位于 drmarl 列的紧挨着的右侧一列，因此列索引是 col_drmarl[i] + 1
        pct_change = float(df.iloc[-1, col_drmarl[i] + 1])
        sign = "+" if pct_change > 0 else ""
        
        max_val = max(data_marl[i].max(), data_drmarl[i].max())
        y_range = max_val - min(data_marl[i].min(), data_drmarl[i].min())
        text_y_pos = max_val + y_range * 0.05 

        # --- Baseline (marl) ---
        light_color = mcolors.to_rgba(color, 0.6)
        face_light = mcolors.to_rgba(color, 0.1)
        
        ax.boxplot(data_marl[i], 
                   positions=[positions_marl[i]], 
                   widths=0.6, patch_artist=True,
                   showmeans=True, meanline=True, 
                   medianprops=dict(visible=False), 
                   boxprops=dict(facecolor=face_light, color=light_color, linestyle='--', linewidth=1.5),
                   capprops=dict(color=light_color, linestyle='--', linewidth=1.5),
                   whiskerprops=dict(color=light_color, linestyle='--', linewidth=1.5),
                   meanprops=dict(color=light_color, linestyle='--', linewidth=2), 
                   flierprops=dict(marker='o', markerfacecolor=face_light, markeredgecolor=light_color, markersize=4))

        # 标注 Baseline 平均数值
        ax.text(positions_marl[i], mean_marl, f'{mean_marl:.2f}', 
                ha='center', va='center', fontsize=8, color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'))

        # --- Retrain (drmarl) ---
        face_dark = mcolors.to_rgba(color, 0.85)
        
        ax.boxplot(data_drmarl[i], 
                   positions=[positions_drmarl[i]], 
                   widths=0.6, patch_artist=True,
                   showmeans=True, meanline=True, 
                   medianprops=dict(visible=False), 
                   boxprops=dict(facecolor=face_dark, color=color, linestyle='-', linewidth=1.5),
                   capprops=dict(color=color, linestyle='-', linewidth=1.5),
                   whiskerprops=dict(color=color, linestyle='-', linewidth=1.5),
                   meanprops=dict(color=color, linestyle='-', linewidth=2), 
                   flierprops=dict(marker='o', markerfacecolor=face_dark, markeredgecolor=color, markersize=4))
                               
        # 标注 Retrain 平均数值
        ax.text(positions_drmarl[i], mean_drmarl, f'{mean_drmarl:.2f}', 
                ha='center', va='center', fontsize=8, color='white', fontweight='bold',
                bbox=dict(facecolor=color, edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2'))

        # --- 标注变化比例 ---
        ax.text((positions_marl[i] + positions_drmarl[i]) / 2, text_y_pos, 
                f'{sign}{pct_change:.1f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333333',
                bbox=dict(facecolor='#f0f0f0', edgecolor='gray', alpha=0.8, boxstyle='round,pad=0.3'))

    ax.set_xticks([1.5, 4.5, 7.5, 10.5])
    ax.set_xticklabels([item["algo_name"] for item in DATA_PAIRS], fontsize=11)
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison', pad=15)
    ax.grid(axis='y', linestyle=':', alpha=0.7)

    # === 新增：添加垂直虚线区分不同的算法 ===
    # 算法的组中间位置分别在 x=3.0 (1.5和4.5之间), x=6.0 (4.5和7.5之间), x=9.0 (7.5和10.5之间)
    for vline_x in [3.0, 6.0, 9.0]:
        ax.axvline(x=vline_x, color='gray', linestyle='--', linewidth=1.2, alpha=0.5)
    # ==================================

    # 扩展 Y 轴上限
    ylim_bottom, ylim_top = ax.get_ylim()
    ax.set_ylim(ylim_bottom, ylim_top + (ylim_top - ylim_bottom) * 0.1)

    baseline_patch = mpatches.Patch(facecolor=mcolors.to_rgba('gray', 0.1), edgecolor=mcolors.to_rgba('gray', 0.6), 
                                    linestyle='--', linewidth=1.5, label='Baseline')
    retrain_patch = mpatches.Patch(facecolor=mcolors.to_rgba('gray', 0.85), edgecolor='gray', 
                                   linestyle='-', linewidth=1.5, label='Retrain')

    # 图例改为1列，节省空间
    ax.legend(handles=[baseline_patch, retrain_patch], loc='upper left', ncol=1, fontsize=9)
    # ax.legend(handles=[baseline_patch, retrain_patch], loc='best', ncol=1, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def main():
    _configure_plot_style()

    file_path = Path('runs_eval') / 'signal_controller_benchmark' / 'full_performance_comparison.xlsx'
    # file_path = Path('runs_eval') / 'signal_controller_benchmark_real' / 'full_performance_comparison_real.xlsx'
    if not Path(file_path).exists():
        print(f"Error: 找不到数据文件 {file_path}")
        return
        
    df = pd.read_excel(file_path, header=None)

    # --- 尺寸修改为 10x6 ---
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), dpi=150)

    # 画左侧 Queue Length
    plot_boxplot_metric_on_ax(ax1, df, 
                              metric_name='Queue Length (veh)', 
                              col_marl=[1, 7, 13, 19], 
                              col_drmarl=[2, 8, 14, 20])

    # 画右侧 Average Speed
    plot_boxplot_metric_on_ax(ax2, df, 
                              metric_name='Average Speed (m/s)', 
                              col_marl=[4, 10, 16, 22], 
                              col_drmarl=[5, 11, 17, 23])

    # 整体大标题字号稍微缩减
    fig.suptitle('Algorithm Controller Performance Retrain Comparison in 5x5 Grid', fontsize=14, fontweight='bold', y=0.95)
    # fig.suptitle('Algorithm Controller Performance Retrain Comparison in Monaco City', fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    # 保存为一张总图
    output_path = OUTPUT_DIR / 'Combined_Performance_Comparison.png'
    # output_path = OUTPUT_DIR / 'Combined_Performance_Comparison_real.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"========================================")
    print(f"Saved combined plot to: {output_path}")

if __name__ == "__main__":
    main()