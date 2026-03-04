#!/bin/bash
# ============================================================
# 训练启动脚本
# 用法:
#   bash scripts/train.sh act          # ACT策略
#   bash scripts/train.sh diffusion    # Diffusion Policy
#   bash scripts/train.sh vqbet        # VQ-BeT
#   bash scripts/train.sh smolvla      # SmolVLA (需 12GB+ VRAM)
#   bash scripts/train.sh pi0          # Pi0 (需 24GB+ VRAM)
#
# 断点续训:
#   bash scripts/train.sh act resume
# ============================================================
set -e

POLICY=${1:-act}
RESUME=${2:-""}

CONFIG_DIR="configs"
CONFIG_FILE="${CONFIG_DIR}/${POLICY}_realman.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    echo ""
    echo "可用策略:"
    ls ${CONFIG_DIR}/*_realman.yaml 2>/dev/null | sed 's|.*\/||' | sed 's|_realman.yaml||'
    exit 1
fi

echo "============================================"
echo "  Training: ${POLICY}"
echo "  Config:   ${CONFIG_FILE}"
echo "============================================"

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# 构建训练命令
CMD="python -m lerobot.scripts.lerobot_train --config ${CONFIG_FILE}"

# 断点续训
if [ "$RESUME" == "resume" ]; then
    # 找到最新的checkpoint
    OUTPUT_DIR=$(grep "output_dir:" "$CONFIG_FILE" | awk '{print $2}')
    LAST_CKPT="${OUTPUT_DIR}/checkpoints/last/pretrained_model"
    
    if [ -d "$LAST_CKPT" ]; then
        CMD="python -m lerobot.scripts.lerobot_train \
            --config_path=${LAST_CKPT}/train_config.json \
            --resume=true"
        echo "📂 Resume from: ${LAST_CKPT}"
    else
        echo "⚠️  未找到 checkpoint，从头开始训练"
    fi
fi

echo "🚀 Command: ${CMD}"
echo ""

eval $CMD
