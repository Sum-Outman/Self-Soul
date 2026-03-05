# Self Soul 开发工具脚本
# 包含常用的开发命令和工具

# 颜色定义
$RED = "`e[31m"
$GREEN = "`e[32m"
$YELLOW = "`e[33m"
$BLUE = "`e[34m"
$CYAN = "`e[36m"
$RESET = "`e[0m"

# 日志函数
function Write-Log {
    param([string]$Message)
    Write-Host "$GREEN[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')]$RESET $Message"
}

function Write-Info {
    param([string]$Message)
    Write-Host "$CYAN[INFO]$RESET $Message"
}

function Write-Warning {
    param([string]$Message)
    Write-Host "$YELLOW[WARNING]$RESET $Message"
}

function Write-Error {
    param([string]$Message)
    Write-Host "$RED[ERROR]$RESET $Message"
}

# 显示帮助
function Show-Help {
    Write-Host ""
    Write-Host "$BLUE==========================================$RESET"
    Write-Host "$GREEN        Self Soul 开发工具菜单          $RESET"
    Write-Host "$BLUE==========================================$RESET"
    Write-Host ""
    Write-Host "可用命令:"
    Write-Host "  $YELLOW./dev-tools.ps1 install$RESET       - 安装所有依赖"
    Write-Host "  $YELLOW./dev-tools.ps1 lint$RESET          - 运行代码质量检查"
    Write-Host "  $YELLOW./dev-tools.ps1 format$RESET        - 格式化代码"
    Write-Host "  $YELLOW./dev-tools.ps1 test$RESET          - 运行测试"
    Write-Host "  $YELLOW./dev-tools.ps1 build$RESET         - 构建项目"
    Write-Host "  $YELLOW./dev-tools.ps1 clean$RESET         - 清理构建文件"
    Write-Host "  $YELLOW./dev-tools.ps1 db-reset$RESET      - 重置数据库"
    Write-Host "  $YELLOW./dev-tools.ps1 check-deps$RESET    - 检查依赖更新"
    Write-Host "  $YELLOW./dev-tools.ps1 pre-commit$RESET    - 运行pre-commit"
    Write-Host "  $YELLOW./dev-tools.ps1 help$RESET          - 显示此帮助"
    Write-Host ""
    Write-Host "$BLUE==========================================$RESET"
    Write-Host ""
}

# 安装依赖
function Install-Dependencies {
    Write-Log "安装项目依赖..."
    
    # 激活虚拟环境
    if (Test-Path ".venv\Scripts\Activate.ps1") {
        Write-Log "激活Python虚拟环境..."
        .venv\Scripts\Activate.ps1
    }
    
    # 安装Python依赖
    Write-Log "安装Python依赖..."
    pip install -r requirements.txt
    
    # 安装开发依赖
    Write-Log "安装开发依赖..."
    pip install black flake8 isort mypy bandit pylint pre-commit
    
    # 安装前端依赖
    Write-Log "安装前端依赖..."
    cd app
    npm install
    cd ..
    
    # 设置pre-commit钩子
    Write-Log "设置pre-commit钩子..."
    pre-commit install
    
    Write-Log "依赖安装完成"
}

# 代码质量检查
function Run-Lint {
    Write-Log "运行代码质量检查..."
    
    # 运行flake8
    Write-Info "运行flake8..."
    flake8 core/ --count --select=E9,F63,F7,F82 --show-source --statistics
    flake8 core/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    # 运行mypy
    Write-Info "运行mypy..."
    mypy core/ --ignore-missing-imports
    
    # 运行bandit
    Write-Info "运行bandit..."
    bandit -r core/ -c pyproject.toml
    
    # 运行pylint
    Write-Info "运行pylint..."
    pylint core/ --rcfile=pyproject.toml
    
    Write-Log "代码质量检查完成"
}

# 代码格式化
function Run-Format {
    Write-Log "格式化代码..."
    
    # 运行black
    Write-Info "运行black..."
    black core/ --line-length 127
    
    # 运行isort
    Write-Info "运行isort..."
    isort core/ --profile black
    
    Write-Log "代码格式化完成"
}

# 运行测试
function Run-Tests {
    Write-Log "运行测试..."
    
    # 运行核心测试
    Write-Info "运行核心测试..."
    python core/test_core.py
    
    # 运行其他测试
    Write-Info "运行知识管理器测试..."
    python core/test_knowledge_manager.py
    
    Write-Log "测试完成"
}

# 构建项目
function Build-Project {
    Write-Log "构建项目..."
    
    # 构建前端
    Write-Info "构建前端..."
    cd app
    npm run build
    cd ..
    
    # 检查构建结果
    if (Test-Path "app\dist") {
        Write-Log "前端构建成功"
    } else {
        Write-Error "前端构建失败"
    }
    
    Write-Log "项目构建完成"
}

# 清理构建文件
function Clean-Build {
    Write-Log "清理构建文件..."
    
    # 清理Python缓存
    Write-Info "清理Python缓存..."
    Get-ChildItem -Path . -Include "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force
    Get-ChildItem -Path . -Include "*.pyc" -Recurse -File | Remove-Item -Force
    Get-ChildItem -Path . -Include "*.pyo" -Recurse -File | Remove-Item -Force
    Get-ChildItem -Path . -Include "*.pyd" -Recurse -File | Remove-Item -Force
    
    # 清理前端构建
    Write-Info "清理前端构建..."
    if (Test-Path "app\dist") {
        Remove-Item -Path "app\dist" -Recurse -Force
    }
    if (Test-Path "app\node_modules") {
        Write-Warning "保留node_modules目录（如需清理请手动删除）"
    }
    
    # 清理日志
    Write-Info "清理日志..."
    if (Test-Path "core\logs") {
        Get-ChildItem -Path "core\logs" -Filter "*.log" | Remove-Item -Force
    }
    
    Write-Log "清理完成"
}

# 重置数据库
function Reset-Database {
    Write-Log "重置数据库..."
    
    # 备份现有数据库
    if (Test-Path "core\self_soul.db") {
        $backupName = "self_soul_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').db"
        Copy-Item "core\self_soul.db" "core\$backupName"
        Write-Log "数据库已备份到: core\$backupName"
    }
    
    # 删除数据库文件
    if (Test-Path "core\self_soul.db") {
        Remove-Item -Path "core\self_soul.db" -Force
        Write-Log "数据库文件已删除"
    }
    
    # 重新初始化数据库
    Write-Info "重新初始化数据库..."
    try {
        python -c "from core.database.db_access_layer import DatabaseAccessLayer; db = DatabaseAccessLayer(); print('数据库初始化成功')"
        Write-Log "数据库重置完成"
    } catch {
        Write-Error "数据库初始化失败: $_"
    }
}

# 检查依赖更新
function Check-Dependencies {
    Write-Log "检查依赖更新..."
    
    # 检查Python依赖
    Write-Info "检查Python依赖更新..."
    pip list --outdated
    
    # 检查前端依赖
    Write-Info "检查前端依赖更新..."
    cd app
    npm outdated
    cd ..
    
    Write-Log "依赖检查完成"
}

# 运行pre-commit
function Run-PreCommit {
    Write-Log "运行pre-commit检查..."
    pre-commit run --all-files
    Write-Log "pre-commit检查完成"
}

# 主函数
function Main {
    param(
        [string]$Command = "help"
    )
    
    switch ($Command.ToLower()) {
        "install" {
            Install-Dependencies
        }
        "lint" {
            Run-Lint
        }
        "format" {
            Run-Format
        }
        "test" {
            Run-Tests
        }
        "build" {
            Build-Project
        }
        "clean" {
            Clean-Build
        }
        "db-reset" {
            Reset-Database
        }
        "check-deps" {
            Check-Dependencies
        }
        "pre-commit" {
            Run-PreCommit
        }
        "help" {
            Show-Help
        }
        default {
            Write-Error "未知命令: $Command"
            Show-Help
        }
    }
}

# 解析命令行参数
if ($args.Count -gt 0) {
    Main -Command $args[0]
} else {
    Show-Help
}