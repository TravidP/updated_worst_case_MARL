#!/bin/bash

# 1. 确保 shell 可以识别 conda 命令
source $(conda info --base)/etc/profile.d/conda.sh

echo "开始创建 Conda 环境: deeprlsc..."

# 2. 检查环境是否已存在，存在则删除（可选，视需求而定）
# conda env remove -n deeprlsc -y

# 3. 根据 environment.yml 安装环境
conda env create -f environment.yml

# 4. 激活环境
conda activate deeprlsc

echo "环境部署完成！请使用 'conda activate deeprlsc' 进入环境。"