#!/bin/bash
# Self-Soul AGI 快速构建和测试脚本
# 用于开发和测试环境

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "命令 '$1' 未找到，请先安装"
        exit 1
    fi
    log_success "命令 '$1' 已安装"
}

# 显示帮助
show_help() {
    echo "Self-Soul AGI 快速构建和测试脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --build         构建Docker镜像"
    echo "  --start         启动服务"
    echo "  --stop          停止服务"
    echo "  --restart       重启服务"
    echo "  --logs          查看日志"
    echo "  --test          运行快速测试"
    echo "  --monitor       检查监控API"
    echo "  --clean         清理容器和镜像"
    echo "  --all           执行完整流程: build -> start -> test"
    echo "  --help          显示此帮助信息"
    echo ""
}

# 构建Docker镜像
build() {
    log_info "构建Docker镜像..."
    if docker-compose build; then
        log_success "Docker镜像构建成功"
    else
        log_error "Docker镜像构建失败"
        exit 1
    fi
}

# 启动服务
start() {
    log_info "启动服务..."
    if docker-compose up -d; then
        log_success "服务启动成功"
        log_info "等待服务就绪..."
        sleep 10
    else
        log_error "服务启动失败"
        exit 1
    fi
}

# 停止服务
stop() {
    log_info "停止服务..."
    docker-compose down
    log_success "服务已停止"
}

# 重启服务
restart() {
    log_info "重启服务..."
    stop
    start
}

# 查看日志
logs() {
    log_info "查看服务日志..."
    docker-compose logs -f
}

# 运行快速测试
test() {
    log_info "运行快速测试..."
    
    # 测试健康检查
    log_info "测试健康检查..."
    if curl -f http://localhost:8000/health 2>/dev/null; then
        log_success "健康检查通过"
    else
        log_error "健康检查失败"
        exit 1
    fi
    
    # 测试系统状态API
    log_info "测试系统状态API..."
    if curl -f http://localhost:8000/api/system/status 2>/dev/null; then
        log_success "系统状态API测试通过"
    else
        log_error "系统状态API测试失败"
        exit 1
    fi
    
    # 测试监控数据API
    log_info "测试监控数据API..."
    if curl -f http://localhost:8000/api/monitoring/data 2>/dev/null; then
        log_success "监控数据API测试通过"
    else
        log_error "监控数据API测试失败"
        exit 1
    fi
    
    log_success "所有测试通过"
}

# 检查监控API
monitor() {
    log_info "检查监控API..."
    echo "获取监控数据:"
    curl -s http://localhost:8000/api/monitoring/data | python -m json.tool || echo "监控API不可用"
}

# 清理容器和镜像
clean() {
    log_info "清理容器和镜像..."
    docker-compose down --volumes --remove-orphans 2>/dev/null || true
    log_success "清理完成"
}

# 完整流程
all() {
    log_info "执行完整流程..."
    check_command docker
    check_command docker-compose
    build
    start
    test
    monitor
    
    log_success "完整流程执行成功"
    log_info "前端访问: http://localhost:5175"
    log_info "后端API: http://localhost:8000"
    log_info "监控数据: http://localhost:8000/api/monitoring/data"
}

# 主函数
main() {
    local action="${1:---help}"
    
    case "$action" in
        --build)
            build
            ;;
        --start)
            start
            ;;
        --stop)
            stop
            ;;
        --restart)
            restart
            ;;
        --logs)
            logs
            ;;
        --test)
            test
            ;;
        --monitor)
            monitor
            ;;
        --clean)
            clean
            ;;
        --all)
            all
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "未知选项: $action"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"