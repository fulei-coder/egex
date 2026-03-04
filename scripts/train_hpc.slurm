#!/bin/bash
#==============================================================================
# HPC Slurm 训练脚本示例
#
# 适用于有 GPU 集群（Slurm 调度）的训练场景。
# 这个脚本是在实际 A800 集群上调试通过的，踩了不少坑，注释里有说明。
#
# 用法:
#   sbatch scripts/train_hpc.slurm                    # 默认训练 ACT
#   sbatch --export=ALL,POLICY=smolvla scripts/train_hpc.slurm  # 训练 SmolVLA
#   sbatch --export=ALL,POLICY=act,RESUME=1 scripts/train_hpc.slurm  # 断点续训
#
# 批量提交:
#   for p in act diffusion smolvla; do
#     sbatch --export=ALL,POLICY=$p --job-name=train_$p scripts/train_hpc.slurm
#   done
#==============================================================================

#SBATCH --job-name=lerobot_train
#SBATCH --partition=gpuA800          # ← 按你的集群改，常见: gpu, gpuA100, gpuA800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12           # 视频解码吃CPU，给够
#SBATCH --gres=gpu:1                 # 单卡训练
#SBATCH --time=72:00:00              # 大模型可能需要2-3天
#SBATCH --output=logs/%j_%x.out      # %j=JobID, %x=job-name
#SBATCH --error=logs/%j_%x.err

# =============================================================================
# ⚠️ 踩坑记录：
# 1. logs/ 目录必须提前存在，否则 sbatch 秒退且无任何报错
# 2. 不要依赖 conda activate，Slurm 环境下 .bashrc 不一定执行
# 3. TRANSFORMERS_CACHE 已废弃，只用 HF_HOME 就够了
# 4. 输出目录已存在会报 FileExistsError，下面做了自动判断
# =============================================================================

set -e

# ==================== 参数 ====================
POLICY=${POLICY:-act}
RESUME=${RESUME:-0}

# ==================== 路径配置 ====================
# ⭐ 改成你自己的路径
PROJECT_DIR="/share/home/xxx/lerobot"       # 项目根目录
PYTHON="/share/home/xxx/.conda/envs/lerobot/bin/python"  # Python 绝对路径
HF_CACHE="/share/home/xxx/.cache/huggingface"            # HuggingFace 缓存

# ==================== 环境变量 ====================
# 集群计算节点通常无外网，不设这些会卡在尝试下载
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME="$HF_CACHE"

# Python PATH（不依赖 module load 和 conda init）
export PATH="$(dirname $PYTHON):$PATH"

# ==================== 工作目录 ====================
cd "$PROJECT_DIR"
mkdir -p logs  # 必须！否则 sbatch 秒退

# ==================== 配置文件映射 ====================
CONFIG="configs/${POLICY}_realman.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "❌ 找不到配置: $CONFIG"
    echo "可用策略: $(ls configs/*_realman.yaml 2>/dev/null | sed 's|.*/||;s|_realman.yaml||' | tr '\n' ' ')"
    exit 1
fi

# 从配置中提取输出目录
OUTPUT_DIR=$(grep 'output_dir:' "$CONFIG" | awk '{print $2}')

# ==================== 信息打印 ====================
echo "=========================================="
echo "  策略:     $POLICY"
echo "  配置:     $CONFIG"
echo "  输出:     $OUTPUT_DIR"
echo "  Job ID:   ${SLURM_JOB_ID:-local}"
echo "  节点:     ${SLURM_NODELIST:-$(hostname)}"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Python:   $PYTHON"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ==================== 构建训练命令 ====================
if [ "$RESUME" = "1" ]; then
    # 断点续训：找 checkpoint
    CKPT_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
    if [ -f "$CKPT_CONFIG" ]; then
        echo "📂 从 checkpoint 恢复: $CKPT_CONFIG"
        CMD="$PYTHON -m lerobot.scripts.lerobot_train \
            --config_path=$CKPT_CONFIG \
            --resume=true"
    else
        echo "⚠️ 未找到 checkpoint，从头开始"
        CMD="$PYTHON -m lerobot.scripts.lerobot_train --config $CONFIG"
    fi
else
    # 全新训练：检查输出目录
    if [ -d "$OUTPUT_DIR" ]; then
        # 有 checkpoint → 改为 resume（避免浪费已训练的部分）
        CKPT_CONFIG="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
        if [ -f "$CKPT_CONFIG" ]; then
            echo "⚠️ 输出目录已存在且有 checkpoint，自动 resume"
            CMD="$PYTHON -m lerobot.scripts.lerobot_train \
                --config_path=$CKPT_CONFIG \
                --resume=true"
        else
            echo "⚠️ 输出目录存在但无 checkpoint，清理残留"
            rm -rf "$OUTPUT_DIR"
            CMD="$PYTHON -m lerobot.scripts.lerobot_train --config $CONFIG"
        fi
    else
        CMD="$PYTHON -m lerobot.scripts.lerobot_train --config $CONFIG"
    fi
fi

# ==================== 执行训练 ====================
LOG_FILE="logs/train_${POLICY}_${SLURM_JOB_ID:-local}.log"
echo "🚀 启动: $CMD"
echo "📝 日志: $LOG_FILE"
echo ""

# 用 tee 同时输出到终端和日志文件
# srun 在 sbatch 内部可以不写（单节点单任务），但写上更规范
eval $CMD 2>&1 | tee "$LOG_FILE"

echo ""
echo "✅ 训练结束: $(date '+%Y-%m-%d %H:%M:%S')"
nvidia-smi 2>/dev/null || true
