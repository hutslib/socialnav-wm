#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 确保使用本仓库的 habitat_baselines（Falcon 根目录优先，避免用到 site-packages 里的旧版本）
export PYTHONPATH="$(pwd):$(pwd)/habitat-baselines:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
TOTAL_GPU="${TOTAL_GPU:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ==============================================================================
# 默认配置
# ==============================================================================
CONFIG_NAME="${1:-exp_01_baseline_falcon.yaml}"

# 中断后自动重启：默认开启（异常退出时会等待一段时间后重新运行，并从 checkpoint 续训）
# 关闭请设置: AUTO_RESTART=0；可选: RESTART_DELAY=秒数（默认 10），MAX_RESTARTS=最大重启次数（默认 0 表示不限制）
AUTO_RESTART="${AUTO_RESTART:-1}"
RESTART_DELAY="${RESTART_DELAY:-10}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"

# 从配置文件名提取实验名称
EXP_NAME=$(basename "$CONFIG_NAME" .yaml)

# ==============================================================================
# 创建必要的目录
# ==============================================================================
echo "======================================================================"
echo "准备运行实验: $EXP_NAME"
echo "配置文件: $CONFIG_NAME"
echo "GPU 个数: $TOTAL_GPU   (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo "AUTO_RESTART: $AUTO_RESTART"
if [ "$AUTO_RESTART" = "1" ]; then
    echo "已开启: 实验中断后将自动重新启动并续训 (间隔 ${RESTART_DELAY}s)"
fi
echo "======================================================================"

# 创建实验目录（与 config 中 tensorboard_dir / video_dir / checkpoint_folder 对应）
mkdir -p "experiments/${EXP_NAME}/tb"
mkdir -p "experiments/${EXP_NAME}/video"
mkdir -p "experiments/${EXP_NAME}/checkpoints"

echo "✓ 目录已创建"
echo ""

RESUME_ARGS=()
if [[ "$CONFIG_NAME" != *"exp_08"* ]]; then
    RESUME_ARGS=(habitat_baselines.load_resume_state_config=True)
fi

# 日志与实验目录对应，带时间戳避免覆盖：experiments/${EXP_NAME}/train_YYYYMMDD_HHMMSS.log
# 支持在自后台重启时复用同一个时间戳，避免父子进程日志路径不一致
LOG_TIMESTAMP="${LOG_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="experiments/${EXP_NAME}/train_${LOG_TIMESTAMP}.log"

# 分布式端口：优先使用 MASTER_PORT，若被占用则自动顺延查找可用端口
pick_master_port() {
    local start_port=$1
    local max_tries="${2:-100}"
    local port=$start_port
    local i=0

    while [ "$i" -lt "$max_tries" ]; do
        if python - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    s.close()
sys.exit(0)
PY
        then
            echo "$port"
            return 0
        fi
        port=$((port + 1))
        i=$((i + 1))
    done
    return 1
}

# 默认后台启动：首次执行会自动转为后台进程并释放当前终端
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
if [ "$RUN_IN_BACKGROUND" = "1" ] && [ "${FALCON_BG_CHILD:-0}" != "1" ]; then
    echo "正在后台启动训练..."
    nohup env FALCON_BG_CHILD=1 RUN_IN_BACKGROUND=1 LOG_TIMESTAMP="$LOG_TIMESTAMP" \
        bash "$0" "$CONFIG_NAME" > /dev/null 2>&1 &
    BG_PID=$!
    echo "✓ 已在后台启动 (PID: $BG_PID)"
    echo "训练日志: $LOG_FILE"
    echo "查看日志: tail -f \"$LOG_FILE\""
    exit 0
fi

# ==============================================================================
# 运行训练（支持中断后自动重启）
# ==============================================================================
run_train() {
    local append_log=$1
    local selected_port

    selected_port=$(pick_master_port "$MASTER_PORT" 200)
    if [ -z "$selected_port" ]; then
        echo "✗ 无法找到可用分布式端口 (起始端口: $MASTER_PORT)" | tee -a "$LOG_FILE"
        return 1
    fi
    export MASTER_PORT="$selected_port"

    echo "使用分布式端口: $MASTER_PORT" | tee -a "$LOG_FILE"

    if [ "$append_log" = "1" ]; then
        echo "" >> "$LOG_FILE"
        echo "========== 自动重启于 $(date) ==========" >> "$LOG_FILE"
        python -u -m torch.distributed.launch \
            --use_env \
            --nproc_per_node $TOTAL_GPU \
            --master_port "$MASTER_PORT" \
            habitat-baselines/habitat_baselines/run.py \
            --config-name="$CONFIG_NAME" \
            "${RESUME_ARGS[@]}" \
            >> "$LOG_FILE" 2>&1
    else
        python -u -m torch.distributed.launch \
            --use_env \
            --nproc_per_node $TOTAL_GPU \
            --master_port "$MASTER_PORT" \
            habitat-baselines/habitat_baselines/run.py \
            --config-name="$CONFIG_NAME" \
            "${RESUME_ARGS[@]}" \
            > "$LOG_FILE" 2>&1
    fi
    return $?
}

RESTART_COUNT=0
while true; do
    if [ $RESTART_COUNT -eq 0 ]; then
        echo "开始训练..."
        echo "日志文件: $LOG_FILE"
        echo "若未训练完可再次执行本脚本，将自动从当前实验的 checkpoint 续训"
        echo ""
        set -x
        run_train 0
    else
        set -x
        run_train 1
    fi
    EXIT_CODE=$?
    set +x

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "======================================================================"
        echo "✓ 训练完成: $EXP_NAME"
        echo "======================================================================"
        exit 0
    fi

    # 未开启自动重启则直接退出
    if [ "$AUTO_RESTART" != "1" ]; then
        echo ""
        echo "======================================================================"
        echo "✗ 训练失败: $EXP_NAME (退出码: $EXIT_CODE)"
        echo "请查看日志: $LOG_FILE"
        echo "需自动重启时请使用: AUTO_RESTART=1 $0 $CONFIG_NAME"
        echo "======================================================================"
        exit $EXIT_CODE
    fi

    # 检查是否超过最大重启次数
    if [ "$MAX_RESTARTS" -gt 0 ] && [ $RESTART_COUNT -ge "$MAX_RESTARTS" ]; then
        echo ""
        echo "======================================================================"
        echo "✗ 已达最大重启次数 ($MAX_RESTARTS)，停止重启"
        echo "======================================================================"
        exit $EXIT_CODE
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo ""
    echo "======================================================================"
    echo "训练异常退出 (退出码: $EXIT_CODE)，${RESTART_DELAY}s 后自动重启 (第 ${RESTART_COUNT} 次)"
    echo "======================================================================"
    sleep "$RESTART_DELAY"
done
