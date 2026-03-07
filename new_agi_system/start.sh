#!/bin/bash
# 统一认知架构启动脚本

set -e

echo "========================================="
echo "启动统一认知架构"
echo "========================================="

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python版本: $PYTHON_VERSION"

# 检查是否在src目录中
if [ ! -d "src" ]; then
    echo "错误: 请从new_agi_system目录运行"
    exit 1
fi

# 如果需要，安装依赖
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# 检查是否安装了torch
if ! python3 -c "import torch" 2>/dev/null; then
    echo "安装PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 启动API服务器
echo "启动API服务器，端口9000..."
cd src
python -m api.server --host 127.0.0.1 --port 9000