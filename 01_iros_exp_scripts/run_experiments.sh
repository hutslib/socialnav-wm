#!/bin/bash

# World Model 实验批量运行脚本（按顺序自动跑，通过 single_node_falcon.sh 实现续训+日志+可选自动重启）
# 使用方法:
#   ./run_experiments.sh              # 运行所有实验
#   ./run_experiments.sh all          # 同上
#   ./run_experiments.sh 01_baseline_falcon 02_full_wm   # 只跑指定实验
#   ./run_experiments.sh exp_01_baseline_falcon exp_02_full_wm
#   ./run_experiments.sh exp_01_baseline_falcon.yaml exp_02_full_wm.yaml
# 可选: AUTO_RESTART=1 ./run_experiments.sh all   # 每个实验中断后自动重启

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FALCON_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SINGLE_NODE_SCRIPT="${FALCON_ROOT}/habitat-baselines/habitat_baselines/rl/ddppo/single_node_falcon.sh"

# GPU 与训练环境（由此脚本统一设置，会覆盖 single_node_falcon.sh 的默认值）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOTAL_GPU=8
export MASTER_PORT=29501
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HYDRA_FULL_ERROR=1
export AUTO_RESTART=0

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') $1"
}

# 运行单个实验：调用 single_node_falcon.sh（续训、日志到 experiments/${EXP_NAME}/train_*.log、可选 AUTO_RESTART）
run_experiment() {
    local exp_input=$1
    local exp_name
    local config_name

    if [[ "$exp_input" == *.yaml ]]; then
        config_name="$(basename "$exp_input")"
        exp_name="${config_name%.yaml}"
    else
        exp_name="$exp_input"
        if [[ "$exp_name" != exp_* ]]; then
            exp_name="exp_${exp_name}"
        fi
        config_name="${exp_name}.yaml"
    fi

    log_info "========================================="
    log_info "开始运行实验: ${exp_name}"
    log_info "配置文件: ${config_name}"
    log_info "日志目录: experiments/${exp_name}/train_*.log"
    log_info "========================================="

    if [ ! -f "${SINGLE_NODE_SCRIPT}" ]; then
        log_error "未找到 ${SINGLE_NODE_SCRIPT}"
        return 1
    fi

    (cd "${FALCON_ROOT}" && bash "${SINGLE_NODE_SCRIPT}" "$config_name")
    local ret=$?
    if [ $ret -eq 0 ]; then
        log_info "✅ 实验 ${exp_name} 完成"
    else
        log_error "❌ 实验 ${exp_name} 失败 (exit $ret)"
        return 1
    fi
    return 0
}

# 所有实验列表（与 config 下 exp_*.yaml 对应；08 需先跑 02 后单独跑）
ALL_EXPERIMENTS=(
    # "01_baseline_falcon"
    # "02_full_wm"
    # "03_ablation_no_depth"
    # "04_ablation_no_traj"
    # "05_ablation_no_reward"
    # "06_ablation_no_goal_cond"
    # "07_ablation_fusion_only"
    # "08_ablation_frozen_wm"
    # "09_ablation_wm_only"
    # "10a_horizon_4"
    # "10b_horizon_8"
    # "10c_horizon_12"
    # "10d_horizon_16"
    # "11a_rssm_small"
    # "11b_rssm_medium"
    # "11c_rssm_large"
    # "12_ablation_falcon_encoder"
    # "13a_train_ratio_005"
    # "13b_train_ratio_010"
    # "13c_train_ratio_020"
    # "13d_train_ratio_050"
    # "13e_train_ratio_100"
    # "14_ablation_no_aux_loss"
    # "15_wm_with_aux_loss"
)

# 解析命令行参数
if [ $# -eq 0 ] || [ "$1" = "all" ]; then
    log_info "运行所有实验"
    EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
else
    EXPERIMENTS=()
    for arg in "$@"; do
        EXPERIMENTS+=("$arg")
    done
fi

# 运行实验
SUCCESS_COUNT=0
FAIL_COUNT=0

for exp_id in "${EXPERIMENTS[@]}"; do
    if run_experiment "$exp_id"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
done

log_info "========================================="
log_info "所有实验完成！"
log_info "成功: ${SUCCESS_COUNT}, 失败: ${FAIL_COUNT}"
log_info "结果保存在: experiments/"
log_info "========================================="
log_info ""
log_info "查看TensorBoard: tensorboard --logdir experiments/ --port 6006"
log_info "可选: AUTO_RESTART=1 时，每个实验中断后会自动重启并续训"
