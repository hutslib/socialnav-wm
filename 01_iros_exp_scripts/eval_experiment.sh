#!/bin/bash

# 评估实验
# 使用方法: ./eval_experiment.sh <experiment> [checkpoint_step] [num_environments]
# 支持输入:
#   1) 01_baseline_falcon
#   2) exp_01_baseline_falcon
#   3) exp_01_baseline_falcon.yaml
# 示例:
#   ./eval_experiment.sh 01_baseline_falcon
#   ./eval_experiment.sh exp_01_baseline_falcon 100

if [ $# -eq 0 ]; then
    echo "错误: 请指定实验"
    echo "使用方法: ./eval_experiment.sh <experiment> [checkpoint_step]"
    echo "示例:"
    echo "  ./eval_experiment.sh 01_baseline_falcon"
    echo "  ./eval_experiment.sh exp_01_baseline_falcon 100"
    echo "  ./eval_experiment.sh exp_01_baseline_falcon.yaml latest"
    exit 1
fi

EXP_INPUT=$1
CKPT_STEP=${2:-"latest"}
NUM_ENVS=${3:-4}

# 配置环境变量
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HYDRA_FULL_ERROR=1

if [[ "$EXP_INPUT" == *.yaml ]]; then
    CONFIG_NAME="$(basename "$EXP_INPUT")"
    EXP_NAME="${CONFIG_NAME%.yaml}"
    # Strip _eval suffix for checkpoint dir lookup
    EXP_NAME_BASE="${EXP_NAME%_eval}"
else
    EXP_NAME="$EXP_INPUT"
    if [[ "$EXP_NAME" != exp_* ]]; then
        EXP_NAME="exp_${EXP_NAME}"
    fi
    EXP_NAME_BASE="$EXP_NAME"
    # Prefer _eval config if it exists
    EVAL_CONFIG="00_iros_exp_config/${EXP_NAME}_eval.yaml"
    if [ -f "$EVAL_CONFIG" ]; then
        CONFIG_NAME="${EXP_NAME}_eval.yaml"
        echo "使用 eval 专用配置: ${CONFIG_NAME}"
    else
        CONFIG_NAME="${EXP_NAME}.yaml"
    fi
fi

CKPT_DIR="experiments/${EXP_NAME_BASE}/checkpoints"

if [ ! -d "$CKPT_DIR" ]; then
    echo "错误: 找不到checkpoint目录: $CKPT_DIR"
    exit 1
fi

# 查找checkpoint
if [ "$CKPT_STEP" = "latest" ]; then
    CKPT_PATH=$(ls -t ${CKPT_DIR}/ckpt.*.pth 2>/dev/null | head -1)
else
    CKPT_PATH="${CKPT_DIR}/ckpt.${CKPT_STEP}.pth"
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "错误: 找不到checkpoint: $CKPT_PATH"
    exit 1
fi

echo "评估实验: ${EXP_NAME}"
echo "使用checkpoint: ${CKPT_PATH}"
echo "并行环境数: ${NUM_ENVS}"

python -u -m habitat_baselines.run \
    --config-name="${CONFIG_NAME}" \
    habitat_baselines.evaluate=True \
    habitat_baselines.load_resume_state_config=False \
    habitat_baselines.eval_ckpt_path_dir=${CKPT_PATH} \
    habitat_baselines.eval.should_load_ckpt=True \
    "habitat_baselines.eval.video_option=[disk]" \
    habitat_baselines.video_dir="eval_experiments/${EXP_NAME_BASE}/video" \
    habitat_baselines.num_environments=${NUM_ENVS} \
    habitat_baselines.test_episode_count=50

echo "评估完成: ${EXP_NAME}"
