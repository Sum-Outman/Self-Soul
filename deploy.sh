#!/bin/bash

# Self-Soul AGI 系统一键部署脚本
# 自动校验依赖 + 路径 + 提供部署失败自动回滚机制

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
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

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "命令 '$1' 未找到，请先安装"
        exit 1
    fi
    log_success "命令 '$1' 已安装"
}

# 检查目录是否存在，不存在则创建
check_directory() {
    if [ ! -d "$1" ]; then
        log_warning "目录 '$1' 不存在，正在创建..."
        mkdir -p "$1"
        log_success "目录 '$1' 创建成功"
    else
        log_success "目录 '$1' 已存在"
    fi
}

# 检查文件是否存在
check_file() {
    if [ ! -f "$1" ]; then
        log_error "文件 '$1' 不存在"
        exit 1
    fi
    log_success "文件 '$1' 已存在"
}

# 备份函数
backup_system() {
    local backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
    log_info "正在备份当前系统到 $backup_dir..."
    
    mkdir -p "$backup_dir"
    
    # 备份重要目录
    for dir in config data logs uploads models; do
        if [ -d "$dir" ]; then
            cp -r "$dir" "$backup_dir/" 2>/dev/null || true
        fi
    done
    
    # 备份关键文件
    cp docker-compose.yml "$backup_dir/" 2>/dev/null || true
    cp Dockerfile.backend "$backup_dir/" 2>/dev/null || true
    cp Dockerfile.frontend "$backup_dir/" 2>/dev/null || true
    
    log_success "系统备份完成: $backup_dir"
    echo "$backup_dir"  # 返回备份目录名
}

# 回滚函数
rollback_system() {
    local backup_dir="$1"
    if [ ! -d "$backup_dir" ]; then
        log_error "备份目录 '$backup_dir' 不存在，无法回滚"
        exit 1
    fi
    
    log_info "正在从 $backup_dir 回滚系统..."
    
    # 停止当前运行的服务
    docker-compose down 2>/dev/null || true
    
    # 恢复备份
    for dir in config data logs uploads models; do
        if [ -d "$backup_dir/$dir" ]; then
            rm -rf "$dir" 2>/dev/null || true
            cp -r "$backup_dir/$dir" ./
        fi
    done
    
    # 恢复关键文件
    if [ -f "$backup_dir/docker-compose.yml" ]; then
        cp "$backup_dir/docker-compose.yml" ./
    fi
    
    if [ -f "$backup_dir/Dockerfile.backend" ]; then
        cp "$backup_dir/Dockerfile.backend" ./
    fi
    
    if [ -f "$backup_dir/Dockerfile.frontend" ]; then
        cp "$backup_dir/Dockerfile.frontend" ./
    fi
    
    log_success "系统回滚完成"
}

# 主部署函数
deploy() {
    log_info "=== Self-Soul AGI 系统部署开始 ==="
    
    # 1. 检查必需命令
    log_info "步骤 1: 检查必需命令..."
    check_command docker
    check_command docker-compose
    
    # 2. 检查必需目录
    log_info "步骤 2: 检查必需目录..."
    for dir in config data logs uploads models; do
        check_directory "$dir"
    done
    
    # 3. 检查必需文件
    log_info "步骤 3: 检查必需文件..."
    check_file "docker-compose.yml"
    check_file "Dockerfile.backend"
    check_file "Dockerfile.frontend"
    check_file "requirements.txt"
    
    # 4. 备份当前系统
    log_info "步骤 4: 备份当前系统..."
    local backup_dir=$(backup_system)
    
    # 5. 检查目录权限
    log_info "步骤 5: 检查目录权限..."
    for dir in config data logs uploads models; do
        if [ -d "$dir" ]; then
            if [ ! -w "$dir" ]; then
                log_warning "目录 '$dir' 无写权限，正在尝试修复..."
                sudo chmod -R 755 "$dir" 2>/dev/null || chmod -R 755 "$dir" 2>/dev/null || true
            fi
        fi
    done
    
    # 6. 构建Docker镜像
    log_info "步骤 6: 构建Docker镜像..."
    if ! docker-compose build --pull; then
        log_error "Docker镜像构建失败，正在回滚..."
        rollback_system "$backup_dir"
        exit 1
    fi
    
    # 7. 启动服务
    log_info "步骤 7: 启动服务..."
    if ! docker-compose up -d; then
        log_error "服务启动失败，正在回滚..."
        rollback_system "$backup_dir"
        exit 1
    fi
    
    # 8. 等待服务就绪
    log_info "步骤 8: 等待服务就绪..."
    sleep 10
    
    # 9. 检查服务状态
    log_info "步骤 9: 检查服务状态..."
    if docker-compose ps | grep -q "Exit"; then
        log_error "部分服务启动失败，正在回滚..."
        docker-compose logs
        rollback_system "$backup_dir"
        exit 1
    fi
    
    # 10. 健康检查
    log_info "步骤 10: 执行健康检查..."
    max_attempts=30
    attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health 2>/dev/null; then
            log_success "健康检查通过"
            break
        fi
        
        log_info "等待服务就绪 ($attempt/$max_attempts)..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -gt $max_attempts ]; then
        log_error "健康检查超时，正在回滚..."
        docker-compose logs
        rollback_system "$backup_dir"
        exit 1
    fi
    
    log_success "=== Self-Soul AGI 系统部署成功 ==="
    log_info "前端访问: http://localhost:5175"
    log_info "后端API: http://localhost:8000"
    log_info "实时流管理: http://localhost:8766"
    log_info "查看日志: docker-compose logs -f"
    log_info "停止服务: docker-compose down"
}

# 清理函数
cleanup() {
    log_info "正在清理系统..."
    
    # 停止并删除容器
    docker-compose down --volumes --remove-orphans 2>/dev/null || true
    
    # 删除Docker镜像
    docker rmi self-soul-agi_backend self-soul-agi_frontend 2>/dev/null || true
    
    log_success "系统清理完成"
}

# 显示帮助
show_help() {
    echo "Self-Soul AGI 系统部署脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  deploy     部署系统 (默认)"
    echo "  rollback   回滚到上次部署"
    echo "  cleanup    清理所有容器和镜像"
    echo "  help       显示此帮助信息"
    echo ""
}

# 主函数
main() {
    local command="${1:-deploy}"
    
    case "$command" in
        deploy)
            deploy
            ;;
        rollback)
            if [ -z "$2" ]; then
                log_error "请指定备份目录"
                log_info "可用备份:"
                ls -d backup_* 2>/dev/null || echo "无可用备份"
                exit 1
            fi
            rollback_system "$2"
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"