#!/usr/bin/env python3
"""Manually compare Box Plots from raw eval CSVs.
Generates a combined plot for Queue Length and Average Speed.
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

# 2. 从参考代码移植的字体和排版配置函数
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
        "legend.fontsize": 11,
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
            "Falling back to Matplotlib text rendering because TeX dependencies are missing.",
            RuntimeWarning,
        )

    mpl.rcParams.update(params)

# 3. 配置颜色和输入/输出路径
DATA_PAIRS = [
    {"algo_name": "IA2C",   "color": "#1f77b4"}, # tab:blue
    {"algo_name": "MA2C",   "color": "#ff7f0e"}, # tab:orange
    {"algo_name": "IQL-LR", "color": "#2ca02c"}, # tab:green
    {"algo_name": "PPO",    "color": "#d62728"}  # tab:red
]

# 统一保存目录：匹配原参考代码位置
OUTPUT_DIR = Path("runs_eval/manual_comparisons_real")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_boxplot_metric(ax, df, metric_name, col_marl, col_drmarl):
    """通用的画图函数：可以画不同指标的箱线图并在指定的ax上绘图"""
    
    # 提取数据，跳过前两行表头
    data_marl = [df.iloc[2:, c].dropna().astype(float) for c in col_marl]
    data_drmarl = [df.iloc[2:, c].dropna().astype(float) for c in col_drmarl]
    
    # 位置规划
    positions_marl = [1, 4, 7, 10]
    positions_drmarl = [2, 5, 8, 11]

    for i in range(4):
        color = DATA_PAIRS[i]['color']
        
        # --- Baseline (marl) 样式配置 ---
        light_color = mcolors.to_rgba(color, 0.6)
        face_light = mcolors.to_rgba(color, 0.1)
        
        # 绘制 Baseline
        bp_marl = ax.boxplot(data_marl[i], 
                             positions=[positions_marl[i]], 
                             widths=0.6,
                             patch_artist=True,
                             showmeans=True, meanline=True,
                             boxprops=dict(facecolor=face_light, color=light_color, linestyle='--', linewidth=1.5),
                             capprops=dict(color=light_color, linestyle='--', linewidth=1.5),
                             whiskerprops=dict(color=light_color, linestyle='--', linewidth=1.5),
                             medianprops=dict(color=light_color, linestyle='--', linewidth=2),
                             meanprops=dict(color='black', linestyle=':', linewidth=1.5),
                             flierprops=dict(marker='o', markerfacecolor=face_light, markeredgecolor=light_color, markersize=4))

        # 提取并标注 Baseline 平均数 (直接在平均线位置标注)
        mean_val_marl = data_marl[i].mean()
        ax.text(positions_marl[i], mean_val_marl, f'{mean_val_marl:.2f}', 
                ha='center', va='center', fontsize=9, color='black',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'))

        # --- Retrain (drmarl) 样式配置 ---
        face_dark = mcolors.to_rgba(color, 0.85)
        
        # 绘制 Retrain
        bp_drmarl = ax.boxplot(data_drmarl[i], 
                               positions=[positions_drmarl[i]], 
                               widths=0.6,
                               patch_artist=True,
                               showmeans=True, meanline=True,
                               boxprops=dict(facecolor=face_dark, color=color, linestyle='-', linewidth=1.5),
                               capprops=dict(color=color, linestyle='-', linewidth=1.5),
                               whiskerprops=dict(color=color, linestyle='-', linewidth=1.5),
                               medianprops=dict(color=color, linestyle='-', linewidth=2),
                               meanprops=dict(color='white', linestyle=':', linewidth=1.5),
                               flierprops=dict(marker='o', markerfacecolor=face_dark, markeredgecolor=color, markersize=4))
                               
        # 提取并标注 Retrain 平均数
        mean_val_drmarl = data_drmarl[i].mean()
        ax.text(positions_drmarl[i], mean_val_drmarl, f'{mean_val_drmarl:.2f}', 
                ha='center', va='center', fontsize=9, color='white', fontweight='bold',
                bbox=dict(facecolor=color, edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2'))

        # 同一个算法里面标注出平均数值改变的比例幅度
        pct_change = (mean_val_drmarl - mean_val_marl) / mean_val_marl * 100
        
        # 画箭头连线指示变化
        ax.annotate('', xy=(positions_drmarl[i] - 0.35, mean_val_drmarl), 
                    xytext=(positions_marl[i] + 0.35, mean_val_marl),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.0, alpha=0.6))
        
        mid_x = (positions_marl[i] + positions_drmarl[i]) / 2
        mid_y = (mean_val_marl + mean_val_drmarl) / 2
        
        ax.text(mid_x, mid_y, f'{pct_change:+.1f}%', ha='center', va='center',
                fontsize=9, fontweight='bold', color='black',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, boxstyle='round,pad=0.2', linewidth=0.5))

    # X轴与标签
    ax.set_xticks([1.5, 4.5, 7.5, 10.5])
    ax.set_xticklabels([item["algo_name"] for item in DATA_PAIRS])
    ax.set_ylabel(metric_name)
    
    # 动态匹配标题
    ax.set_title(f'Algorithm Performance: {metric_name}', pad=15)
    ax.grid(axis='y', linestyle=':', alpha=0.7)

    # 去除多余边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def main():
    # 应用配置
    _configure_plot_style()

    # 统一读取数据文件
    file_path = Path('runs_eval') / 'signal_controller_benchmark_real' / 'full_performance_comparison_real.xlsx'
    if not file_path.exists():
        print(f"Error: 找不到数据文件 {file_path}")
        return
        
    df = pd.read_excel(file_path, header=None)

    # 创建一个包含两个子图的画布 (1行2列)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=120)

    # 1. 画 Queue Length
    # 根据表格索引关系：
    # IA2C: col 1,2 | MA2C: col 7,8 | IQL-LR: col 13,14 | PPO: col 19,20
    plot_boxplot_metric(axes[0], df, 
                        metric_name='Queue Length', 
                        col_marl=[1, 7, 13, 19], 
                        col_drmarl=[2, 8, 14, 20])

    # 2. 画 Average Speed
    # 根据表格索引关系：
    # IA2C: col 4,5 | MA2C: col 10,11 | IQL-LR: col 16,17 | PPO: col 22,23
    plot_boxplot_metric(axes[1], df, 
                        metric_name='Average Speed', 
                        col_marl=[4, 10, 16, 22], 
                        col_drmarl=[5, 11, 17, 23])

    # 制作全局图例
    baseline_patch = mpatches.Patch(facecolor=mcolors.to_rgba('gray', 0.1), edgecolor=mcolors.to_rgba('gray', 0.6), 
                                    linestyle='--', linewidth=1.5, label='Baseline (marl)')
    retrain_patch = mpatches.Patch(facecolor=mcolors.to_rgba('gray', 0.85), edgecolor='gray', 
                                   linestyle='-', linewidth=1.5, label='Retrain (drmarl)')

    # 添加全局图例到底部中心
    fig.legend(handles=[baseline_patch, retrain_patch], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), fontsize=12)

    plt.tight_layout()
    # 稍微留出底部空间给图例
    fig.subplots_adjust(bottom=0.15)
    
    # 保存并关闭
    output_filename = 'Combined_Algorithm_Performance_boxplot.png'
    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"========================================")
    print(f"Saved combined plot to: {output_path}")

if __name__ == "__main__":
    main()
