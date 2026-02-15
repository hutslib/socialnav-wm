#!/bin/bash
set -e

# 初始化 Conda
source /opt/conda/etc/profile.d/conda.sh

# 激活 Conda 环境
conda activate falcon

# 设置 PYTHONPATH（确保 habitat_baselines 可import）
export PYTHONPATH=$(pwd)

# 运行主脚本
CUDA_VISIBLE_DEVICES=0 python -u -m habitat-baselines.habitat_baselines.eval --config-name=falcon_hm3d_minival_team1.yaml
