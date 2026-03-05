# Self Soul 开发环境启动脚本
# PowerShell版本 - 适用于Windows开发环境

# 设置错误处理
$ErrorActionPreference = "Stop"

# 颜色定义
$RED = "`e[31m"
$GREEN = "`e[32m"
$YELLOW = "`e[33m"
$BLUE = "`e[34m"
$RESET = "`e[0m"

# 日志函数
function Write-Log {
    param([string]$Message)
    Write-Host "$GREEN[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')]$RESET $Message"
}

function Write-Warning {
    param([string]$Message)
    Write-Host "$YELLOW[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] WARNING:$RESET $Message"
}

function Write-Error {
    param([string]$Message)
    Write-Host "$RED[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ERROR:$RESET $Message"
    exit 1
}

# 检查环境
function Check-Environment {
    Write-Log "检查开发环境..."
    
    # 检查Python
    try {
        $pythonVersion = python --version
        Write-Log "Python版本: $pythonVersion"
    } catch {
        Write-Error "Python未安装或不在PATH中"
    }
    
    # 检查pip
    try {
        $pipVersion = pip --version
        Write-Log "pip可用"
    } catch {
        Write-Warning "pip未安装，尝试安装..."
        python -m ensurepip --upgrade
    }
    
    # 检查Node.js
    try {
        $nodeVersion = node --version
        Write-Log "Node.js版本: $nodeVersion"
    } catch {
        Write-Error "Node.js未安装或不在PATH中"
    }
    
    # 检查npm
    try {
        $npmVersion = npm --version
        Write-Log "npm版本: $npmVersion"
    } catch {
        Write-Error "npm未安装或不在PATH中"
    }
    
    # 检查虚拟环境
    if (Test-Path ".venv") {
        Write-Log "Python虚拟环境已存在"
    } else {
        Write-Warning "未找到Python虚拟环境，请先运行: python -m venv .venv"
        Write-Warning "然后激活虚拟环境: .venv\Scripts\Activate.ps1"
        exit 1
    }
}

# 加载开发环境变量
function Load-Environment {
    Write-Log "加载开发环境配置..."
    
    if (Test-Path ".env.development") {
        Get-Content ".env.development" | ForEach-Object {
            if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
                $key = $matches[1].Trim()
                $value = $matches[2].Trim()
                [Environment]::SetEnvironmentVariable($key, $value, "Process")
                Write-Log "  设置环境变量: $key"
            }
        }
    } else {
        Write-Warning "未找到.env.development文件，使用默认配置"
    }
}

# 启动后端服务器
function Start-Backend {
    Write-Log "启动后端服务器..."
    
    # 检查后端依赖
    Write-Log "检查Python依赖..."
    pip install -r requirements.txt
    
    # 启动后端
    Write-Log "启动FastAPI服务器 (端口: 8080)..."
    $backendJob = Start-Job -ScriptBlock {
        cd $using:PWD
        python core/main.py
    } -Name "SelfSoul-Backend"
    
    # 等待后端启动
    Write-Log "等待后端服务器启动..."
    Start-Sleep -Seconds 5
    
    # 检查后端是否运行
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080/api/system/stats" -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Log "后端服务器启动成功"
            return $backendJob
        }
    } catch {
        Write-Warning "后端服务器启动可能有问题，请检查日志"
    }
    
    return $backendJob
}

# 启动前端开发服务器
function Start-Frontend {
    Write-Log "启动前端开发服务器..."
    
    # 检查前端依赖
    if (Test-Path "app\node_modules") {
        Write-Log "前端依赖已安装"
    } else {
        Write-Log "安装前端依赖..."
        cd app
        npm install
        cd ..
    }
    
    # 启动前端
    Write-Log "启动Vite开发服务器 (端口: 5173)..."
    $frontendJob = Start-Job -ScriptBlock {
        cd "$using:PWD\app"
        npm run dev
    } -Name "SelfSoul-Frontend"
    
    # 等待前端启动
    Write-Log "等待前端服务器启动..."
    Start-Sleep -Seconds 10
    
    # 检查前端是否运行
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Log "前端开发服务器启动成功"
        }
    } catch {
        Write-Warning "前端服务器启动可能有问题，请检查日志"
    }
    
    return $frontendJob
}

# 显示状态信息
function Show-Status {
    Write-Host ""
    Write-Host "$BLUE==========================================$RESET"
    Write-Host "$GREEN        Self Soul 开发环境已启动         $RESET"
    Write-Host "$BLUE==========================================$RESET"
    Write-Host ""
    Write-Host "后端API服务器:  $GREEN http://localhost:8080 $RESET"
    Write-Host "前端开发服务器: $GREEN http://localhost:5173 $RESET"
    Write-Host "API文档:        $GREEN http://localhost:8080/docs $RESET"
    Write-Host ""
    Write-Host "可用命令:"
    Write-Host "  测试:          $YELLOW python core/test_core.py $RESET"
    Write-Host "  代码质量检查:  $YELLOW pre-commit run --all-files $RESET"
    Write-Host "  停止服务:      $YELLOW Ctrl+C $RESET"
    Write-Host ""
    Write-Host "$BLUE==========================================$RESET"
    Write-Host ""
}

# 主函数
function Main {
    Write-Host ""
    Write-Host "$BLUE==========================================$RESET"
    Write-Host "$GREEN      Self Soul AGI 开发环境启动脚本     $RESET"
    Write-Host "$BLUE==========================================$RESET"
    Write-Host ""
    
    try {
        # 保存当前目录
        $script:PWD = Get-Location
        
        # 检查环境
        Check-Environment
        
        # 加载环境变量
        Load-Environment
        
        # 启动服务
        $backendJob = Start-Backend
        $frontendJob = Start-Frontend
        
        # 显示状态
        Show-Status
        
        # 等待用户中断
        Write-Host "$YELLOW按 Ctrl+C 停止所有服务...$RESET"
        Write-Host ""
        
        # 监控进程
        while ($true) {
            Start-Sleep -Seconds 1
        }
        
    } catch {
        Write-Error "启动过程中出现错误: $_"
    } finally {
        # 清理作业
        Write-Log "停止所有服务..."
        if ($backendJob) { Stop-Job $backendJob }
        if ($frontendJob) { Stop-Job $frontendJob }
        Write-Log "服务已停止"
    }
}

# 运行主函数
Main