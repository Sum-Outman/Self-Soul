#!/bin/bash

# Self Soul 开发环境启动脚本
# Bash版本 - 适用于Linux/macOS开发环境

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    exit 1
}

# 检查环境
check_environment() {
    log "检查开发环境..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        error "Python3 未安装"
    fi
    python_version=$(python3 --version)
    log "Python版本: $python_version"
    
    # 检查pip
    if ! command -v pip3 &> /dev/null; then
        warn "pip3 未安装，尝试安装..."
        python3 -m ensurepip --upgrade
    fi
    log "pip 可用"
    
    # 检查Node.js
    if ! command -v node &> /dev/null; then
        error "Node.js 未安装"
    fi
    node_version=$(node --version)
    log "Node.js版本: $node_version"
    
    # 检查npm
    if ! command -v npm &> /dev/null; then
        error "npm 未安装"
    fi
    npm_version=$(npm --version)
    log "npm版本: $npm_version"
    
    # 检查虚拟环境
    if [ -d ".venv" ]; then
        log "Python虚拟环境已存在"
    else
        warn "未找到Python虚拟环境，请先运行: python3 -m venv .venv"
        warn "然后激活虚拟环境: source .venv/bin/activate"
        exit 1
    fi
}

# 加载开发环境变量
load_environment() {
    log "加载开发环境配置..."
    
    if [ -f ".env.development" ]; then
        # 从.env.development加载环境变量
        set -a
        source .env.development
        set +a
        log "  已加载 .env.development"
    else
        warn "未找到.env.development文件，使用默认配置"
    fi
}

# 启动后端服务器
start_backend() {
    log "启动后端服务器..."
    
    # 激活虚拟环境
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    
    # 检查后端依赖
    log "检查Python依赖..."
    pip install -r requirements.txt
    
    # 启动后端
    log "启动FastAPI服务器 (端口: ${BACKEND_PORT:-8080})..."
    python core/main.py &
    BACKEND_PID=$!
    
    # 等待后端启动
    log "等待后端服务器启动..."
    sleep 5
    
    # 检查后端是否运行
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${BACKEND_PORT:-8080}/api/system/stats" | grep -q "200"; then
        log "后端服务器启动成功 (PID: $BACKEND_PID)"
    else
        warn "后端服务器启动可能有问题，请检查日志"
    fi
    
    echo $BACKEND_PID
}

# 启动前端开发服务器
start_frontend() {
    log "启动前端开发服务器..."
    
    # 检查前端依赖
    if [ -d "app/node_modules" ]; then
        log "前端依赖已安装"
    else
        log "安装前端依赖..."
        cd app
        npm install
        cd ..
    fi
    
    # 启动前端
    log "启动Vite开发服务器 (端口: ${FRONTEND_PORT:-5173})..."
    cd app
    npm run dev &
    FRONTEND_PID=$!
    cd ..
    
    # 等待前端启动
    log "等待前端服务器启动..."
    sleep 10
    
    # 检查前端是否运行
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${FRONTEND_PORT:-5173}" | grep -q "200"; then
        log "前端开发服务器启动成功 (PID: $FRONTEND_PID)"
    else
        warn "前端服务器启动可能有问题，请检查日志"
    fi
    
    echo $FRONTEND_PID
}

# 显示状态信息
show_status() {
    echo ""
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${GREEN}        Self Soul 开发环境已启动         ${NC}"
    echo -e "${BLUE}==========================================${NC}"
    echo ""
    echo -e "后端API服务器:  ${GREEN}http://localhost:${BACKEND_PORT:-8080}${NC}"
    echo -e "前端开发服务器: ${GREEN}http://localhost:${FRONTEND_PORT:-5173}${NC}"
    echo -e "API文档:        ${GREEN}http://localhost:${BACKEND_PORT:-8080}/docs${NC}"
    echo ""
    echo -e "可用命令:"
    echo -e "  测试:          ${YELLOW}python core/test_core.py${NC}"
    echo -e "  代码质量检查:  ${YELLOW}pre-commit run --all-files${NC}"
    echo -e "  停止服务:      ${YELLOW}Ctrl+C${NC}"
    echo ""
    echo -e "${BLUE}==========================================${NC}"
    echo ""
}

# 清理函数
cleanup() {
    log "停止所有服务..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        log "后端服务器已停止"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        log "前端服务器已停止"
    fi
    log "服务已停止"
    exit 0
}

# 设置信号处理
trap cleanup INT TERM

# 主函数
main() {
    echo ""
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${GREEN}      Self Soul AGI 开发环境启动脚本     ${NC}"
    echo -e "${BLUE}==========================================${NC}"
    echo ""
    
    # 保存当前目录
    SCRIPT_DIR=$(pwd)
    
    # 检查环境
    check_environment
    
    # 加载环境变量
    load_environment
    
    # 启动服务
    BACKEND_PID=$(start_backend)
    FRONTEND_PID=$(start_frontend)
    
    # 显示状态
    show_status
    
    # 等待用户中断
    echo -e "${YELLOW}按 Ctrl+C 停止所有服务...${NC}"
    echo ""
    
    # 监控进程
    wait
}

# 运行主函数
main