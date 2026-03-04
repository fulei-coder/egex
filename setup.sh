#!/bin/bash
# ============================================================
# LeRobot-RealMan-VLA 一键环境搭建脚本
# ============================================================
set -e

echo "============================================"
echo "  LeRobot-RealMan-VLA Environment Setup"
echo "============================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. 创建conda环境
echo -e "\n${GREEN}[1/5] Creating conda environment...${NC}"
if conda info --envs | grep -q "lerobot"; then
    echo -e "${YELLOW}Environment 'lerobot' already exists. Skipping creation.${NC}"
else
    conda create -n lerobot python=3.10 -y
fi

# 2. 激活环境
echo -e "\n${GREEN}[2/5] Activating environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate lerobot

# 3. 安装PyTorch (CUDA 11.8)
echo -e "\n${GREEN}[3/5] Installing PyTorch...${NC}"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. 安装LeRobot
echo -e "\n${GREEN}[4/5] Installing LeRobot...${NC}"
pip install lerobot

# 5. 安装本项目依赖
echo -e "\n${GREEN}[5/5] Installing project dependencies...${NC}"
pip install -r requirements.txt

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Activate with: conda activate lerobot"
echo ""
echo "Next steps:"
echo "  1. Connect hardware (RM65, cameras, Vive)"
echo "  2. Start SteamVR (if using Vive teleoperation)"
echo "  3. Run: python scripts/collect_data.py --help"
