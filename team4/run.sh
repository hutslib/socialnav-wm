#!/bin/bash
set -e

# 初始化 Conda
source /opt/conda/etc/profile.d/conda.sh

# 激活 Conda 环境
conda activate falcon

# 安装缺失的包
pip install einops

# 设置 PYTHONPATH（确保 habitat_baselines 可import）
export PYTHONPATH=$(pwd)

# 删除single_agent_access_mgr.py文件
rm -f /mnt/nvme2/tianshuaih/competition/SocialNav/Falcon/habitat-baselines/habitat_baselines/rl/ppo/single_agent_access_mgr.py

# 复制新的single_agent_access_mgr.py文件
cp /mnt/nvme2/tianshuaih/competition/SocialNav/Falcon/team4/single_agent_access_mgr.py /mnt/nvme2/tianshuaih/competition/SocialNav/Falcon/habitat-baselines/habitat_baselines/rl/ppo/single_agent_access_mgr.py

# 运行主脚本
CUDA_VISIBLE_DEVICES=0 python -u -m team4.eval --config-name=falcon_hm3d_eval_zjm.yaml
