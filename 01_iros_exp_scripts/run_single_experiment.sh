#!/bin/bash

# 运行单个实验（通过 single_node_falcon.sh，实现续训+日志+可选自动重启）
# 使用方法: ./run_single_experiment.sh <experiment>
# 支持输入:
#   1) 01_baseline_falcon
#   2) exp_01_baseline_falcon
#   3) exp_01_baseline_falcon.yaml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALCON_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SINGLE_NODE_SCRIPT="${FALCON_ROOT}/habitat-baselines/habitat_baselines/rl/ddppo/single_node_falcon.sh"

if [ $# -eq 0 ]; then
    echo "错误: 请指定实验"
    echo "使用方法: ./run_single_experiment.sh <experiment>"
    echo "示例:"
    echo "  ./run_single_experiment.sh 01_baseline_falcon"
    echo "  ./run_single_experiment.sh exp_01_baseline_falcon"
    echo "  ./run_single_experiment.sh exp_01_baseline_falcon.yaml"
    exit 1
fi

EXP_INPUT=$1
if [[ "$EXP_INPUT" == *.yaml ]]; then
    CONFIG_NAME="$(basename "$EXP_INPUT")"
    EXP_NAME="${CONFIG_NAME%.yaml}"
else
    EXP_NAME="$EXP_INPUT"
    if [[ "$EXP_NAME" != exp_* ]]; then
        EXP_NAME="exp_${EXP_NAME}"
    fi
    CONFIG_NAME="${EXP_NAME}.yaml"
fi

# 配置环境变量（与 run_experiments.sh 保持一致）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOTAL_GPU=8
export MASTER_PORT=29501
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HYDRA_FULL_ERROR=1
export AUTO_RESTART=0

echo "========================================="
echo "开始运行实验: ${EXP_NAME}"
echo "配置文件: ${CONFIG_NAME}"
echo "日志目录: experiments/${EXP_NAME}/train_*.log"
echo "========================================="

if [ ! -f "${SINGLE_NODE_SCRIPT}" ]; then
    echo "错误: 未找到 ${SINGLE_NODE_SCRIPT}"
    exit 1
fi

(cd "${FALCON_ROOT}" && bash "${SINGLE_NODE_SCRIPT}" "${CONFIG_NAME}")
echo "实验完成: ${EXP_NAME}"
