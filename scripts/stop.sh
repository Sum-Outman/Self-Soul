#!/bin/bash

# Self Soul 停止脚本
# 安全停止所有服务

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

# 停止本地服务
stop_local_services() {
    log "停止本地服务..."
    
    # 停止后端服务
    if [ -f "logs/backend.pid" ]; then
        BACKEND_PID=$(cat logs/backend.pid)
        if kill -0 $BACKEND_PID 2>/dev/null; then
            log "停止后端服务 (PID: $BACKEND_PID)"
            kill $BACKEND_PID
            sleep 3
            
            # 强制停止（如果需要）
            if kill -0 $BACKEND_PID 2>/dev/null; then
                warn "后端服务未正常停止，强制停止"
                kill -9 $BACKEND_PID
            fi
            
            rm logs/backend.pid
        fi
    fi
    
    # 停止前端服务
    if [ -f "logs/frontend.pid" ]; then
        FRONTEND_PID=$(cat logs/frontend.pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            log "停止前端服务 (PID: $FRONTEND_PID)"
            kill $FRONTEND_PID
            sleep 2
            
            # 强制停止（如果需要）
            if kill -0 $FRONTEND_PID 2>/dev/null; then
                warn "前端服务未正常停止，强制停止"
                kill -9 $FRONTEND_PID
            fi
            
            rm logs/frontend.pid
        fi
    fi
}

# 停止Docker服务
stop_docker_services() {
    log "停止Docker服务..."
    
    if command -v docker-compose &> /dev/null && [ -f "docker-compose.yml" ]; then
        docker-compose down
        log "Docker服务已停止"
    else
        warn "Docker Compose不可用或配置文件不存在"
    fi
}

# 清理临时文件
cleanup() {
    log "清理临时文件..."
    
    # 清理Python缓存
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # 清理Node.js缓存
    if [ -d "app/node_modules" ]; then
        warn "保留 node_modules 目录，如需清理请手动删除"
    fi
    
    # 清理日志文件（可选）
    if [ "$1" = "--clean-logs" ]; then
        if [ -d "logs" ]; then
            rm -f logs/*.log
            log "日志文件已清理"
        fi
    fi
}

# 主函数
main() {
    log "开始停止 Self Soul AGI 系统..."
    
    # 检查停止方式
    if command -v docker-compose &> /dev/null && [ -f "docker-compose.yml" ] && [ "$1" = "--docker" ]; then
        stop_docker_services
    else
        stop_local_services
    fi
    
    # 清理
    cleanup "$1"
    
    log "系统停止完成！"
    
    # 验证服务已停止
    sleep 2
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        warn "后端服务仍在运行，请检查"
    else
        log "后端服务已完全停止"
    fi
}

# 处理命令行参数
case "$1" in
    --docker)
        main "$1"
        ;;
    --clean-logs)
        main "$1"
        ;;
    --help|-h)
        echo "用法: $0 [选项]"
        echo "选项:"
        echo "  --docker      停止Docker服务"
        echo "  --clean-logs 清理日志文件"
        echo "  --help, -h   显示帮助信息"
        echo ""
        echo "示例:"
        echo "  $0            # 停止本地服务"
        echo "  $0 --docker   # 停止Docker服务"
        echo "  $0 --clean-logs # 停止服务并清理日志"
        ;;
    *)
        main
        ;;
esac