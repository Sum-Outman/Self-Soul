#!/bin/bash

# Self Soul 启动脚本
# 适用于生产环境部署

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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
    log "检查系统环境..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        error "Python3 未安装"
    fi
    
    # 检查Node.js
    if ! command -v node &> /dev/null; then
        error "Node.js 未安装"
    fi
    
    # 检查Docker（可选）
    if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        log "Docker 环境可用"
        DOCKER_AVAILABLE=true
    else
        warn "Docker 环境不可用，将使用本地模式启动"
        DOCKER_AVAILABLE=false
    fi
    
    # 检查环境变量文件
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            warn "未找到 .env 文件，从 .env.example 创建"
            cp .env.example .env
        else
            error "未找到 .env 或 .env.example 文件"
        fi
    fi
}

# 启动后端服务
start_backend() {
    log "启动后端服务..."
    
    # 激活虚拟环境
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    else
        warn "未找到虚拟环境，使用系统Python"
    fi
    
    # 安装依赖
    if [ ! -d ".venv" ]; then
        log "安装Python依赖..."
        pip install -r requirements.txt
    fi
    
    # 启动服务
    log "启动FastAPI服务..."
    nohup python core/main.py > logs/backend.log 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > logs/backend.pid
    
    # 等待服务启动
    sleep 10
    
    # 检查服务状态
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        log "后端服务启动成功 (PID: $BACKEND_PID)"
    else
        error "后端服务启动失败"
    fi
}

# 启动前端服务
start_frontend() {
    log "启动前端服务..."
    
    cd app
    
    # 安装依赖
    if [ ! -d "node_modules" ]; then
        log "安装前端依赖..."
        npm install
    fi
    
    # 启动服务
    log "启动Vite开发服务器..."
    nohup npm run dev > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../logs/frontend.pid
    
    cd ..
    
    # 等待服务启动
    sleep 5
    
    # 检查服务状态
    if curl -f http://localhost:5175 > /dev/null 2>&1; then
        log "前端服务启动成功 (PID: $FRONTEND_PID)"
    else
        warn "前端服务可能启动较慢，请稍后检查"
    fi
}

# 使用Docker启动
start_with_docker() {
    log "使用Docker Compose启动服务..."
    
    # 检查Docker Compose文件
    if [ ! -f "docker-compose.yml" ]; then
        error "未找到 docker-compose.yml 文件"
    fi
    
    # 启动服务
    docker-compose up -d
    
    # 等待服务启动
    sleep 30
    
    # 检查服务状态
    if docker-compose ps | grep -q "Up"; then
        log "Docker服务启动成功"
    else
        error "Docker服务启动失败"
    fi
}

# 主函数
main() {
    log "开始启动 Self Soul AGI 系统..."
    
    # 创建日志目录
    mkdir -p logs
    
    # 检查环境
    check_environment
    
    # 选择启动方式
    if [ "$DOCKER_AVAILABLE" = true ] && [ "$1" = "--docker" ]; then
        start_with_docker
    else
        start_backend
        start_frontend
    fi
    
    log "系统启动完成！"
    log "后端服务: http://localhost:8000"
    log "前端服务: http://localhost:5175"
    log "API文档: http://localhost:8000/docs"
    log "监控面板: http://localhost:3000 (如果启用)"
    
    # 显示启动信息
    echo ""
    echo "=========================================="
    echo "  Self Soul AGI 系统已成功启动"
    echo "=========================================="
    echo "后端服务: http://localhost:8000"
    echo "前端界面: http://localhost:5175"
    echo "API文档: http://localhost:8000/docs"
    echo "=========================================="
}

# 处理命令行参数
case "$1" in
    --docker)
        main "$1"
        ;;
    --help|-h)
        echo "用法: $0 [选项]"
        echo "选项:"
        echo "  --docker     使用Docker启动"
        echo "  --help, -h   显示帮助信息"
        echo ""
        echo "示例:"
        echo "  $0           # 本地模式启动"
        echo "  $0 --docker  # Docker模式启动"
        ;;
    *)
        main
        ;;
esac