# Self-Soul AGI 系统一键部署脚本 (Windows PowerShell版本)
# 自动校验依赖 + 路径 + 提供部署失败自动回滚机制

# 设置错误处理
$ErrorActionPreference = "Stop"

# 颜色函数
function Write-Color {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    
    $colors = @{
        "Red" = "`e[31m"
        "Green" = "`e[32m"
        "Yellow" = "`e[33m"
        "Blue" = "`e[34m"
        "Magenta" = "`e[35m"
        "Cyan" = "`e[36m"
        "White" = "`e[37m"
        "Reset" = "`e[0m"
    }
    
    if ($colors.ContainsKey($Color)) {
        Write-Host "$($colors[$Color])$Message$($colors['Reset'])" -NoNewline
    } else {
        Write-Host $Message -NoNewline
    }
}

function Write-Info {
    param([string]$Message)
    Write-Color "[INFO] " -Color "Blue"
    Write-Host $Message
}

function Write-Success {
    param([string]$Message)
    Write-Color "[SUCCESS] " -Color "Green"
    Write-Host $Message
}

function Write-Warning {
    param([string]$Message)
    Write-Color "[WARNING] " -Color "Yellow"
    Write-Host $Message
}

function Write-Error {
    param([string]$Message)
    Write-Color "[ERROR] " -Color "Red"
    Write-Host $Message
}

# 检查命令是否存在
function Test-Command {
    param([string]$Command)
    
    try {
        $null = Get-Command $Command -ErrorAction Stop
        Write-Success "命令 '$Command' 已安装"
        return $true
    }
    catch {
        Write-Error "命令 '$Command' 未找到，请先安装"
        return $false
    }
}

# 检查目录是否存在，不存在则创建
function Test-Directory {
    param([string]$Path)
    
    if (-not (Test-Path $Path -PathType Container)) {
        Write-Warning "目录 '$Path' 不存在，正在创建..."
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Success "目录 '$Path' 创建成功"
    }
    else {
        Write-Success "目录 '$Path' 已存在"
    }
}

# 检查文件是否存在
function Test-File {
    param([string]$Path)
    
    if (-not (Test-Path $Path -PathType Leaf)) {
        Write-Error "文件 '$Path' 不存在"
        throw "文件 '$Path' 不存在"
    }
    Write-Success "文件 '$Path' 已存在"
}

# 备份系统
function Backup-System {
    $backupDir = "backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Write-Info "正在备份当前系统到 $backupDir..."
    
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    
    # 备份重要目录
    $directories = @("config", "data", "logs", "uploads", "models")
    foreach ($dir in $directories) {
        if (Test-Path $dir -PathType Container) {
            Copy-Item -Path $dir -Destination $backupDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    
    # 备份关键文件
    $files = @("docker-compose.yml", "Dockerfile.backend", "Dockerfile.frontend")
    foreach ($file in $files) {
        if (Test-Path $file -PathType Leaf) {
            Copy-Item -Path $file -Destination $backupDir -Force -ErrorAction SilentlyContinue
        }
    }
    
    Write-Success "系统备份完成: $backupDir"
    return $backupDir
}

# 回滚系统
function Rollback-System {
    param([string]$BackupDir)
    
    if (-not (Test-Path $BackupDir -PathType Container)) {
        Write-Error "备份目录 '$BackupDir' 不存在，无法回滚"
        throw "备份目录不存在"
    }
    
    Write-Info "正在从 $BackupDir 回滚系统..."
    
    # 停止当前运行的服务
    try {
        docker-compose down 2>&1 | Out-Null
    }
    catch {
        # 忽略错误
    }
    
    # 恢复备份
    $directories = @("config", "data", "logs", "uploads", "models")
    foreach ($dir in $directories) {
        $backupPath = Join-Path $BackupDir $dir
        if (Test-Path $backupPath -PathType Container) {
            if (Test-Path $dir -PathType Container) {
                Remove-Item $dir -Recurse -Force -ErrorAction SilentlyContinue
            }
            Copy-Item -Path $backupPath -Destination . -Recurse -Force
        }
    }
    
    # 恢复关键文件
    $files = @("docker-compose.yml", "Dockerfile.backend", "Dockerfile.frontend")
    foreach ($file in $files) {
        $backupPath = Join-Path $BackupDir $file
        if (Test-Path $backupPath -PathType Leaf) {
            Copy-Item -Path $backupPath -Destination . -Force
        }
    }
    
    Write-Success "系统回滚完成"
}

# 部署系统
function Deploy-System {
    Write-Info "=== Self-Soul AGI 系统部署开始 ==="
    
    # 1. 检查必需命令
    Write-Info "步骤 1: 检查必需命令..."
    if (-not (Test-Command "docker")) { throw "Docker未安装" }
    if (-not (Test-Command "docker-compose")) { throw "Docker Compose未安装" }
    
    # 2. 检查必需目录
    Write-Info "步骤 2: 检查必需目录..."
    $directories = @("config", "data", "logs", "uploads", "models")
    foreach ($dir in $directories) {
        Test-Directory $dir
    }
    
    # 3. 检查必需文件
    Write-Info "步骤 3: 检查必需文件..."
    $files = @("docker-compose.yml", "Dockerfile.backend", "Dockerfile.frontend", "requirements.txt")
    foreach ($file in $files) {
        Test-File $file
    }
    
    # 4. 备份当前系统
    Write-Info "步骤 4: 备份当前系统..."
    $backupDir = Backup-System
    
    # 5. 检查目录权限
    Write-Info "步骤 5: 检查目录权限..."
    foreach ($dir in $directories) {
        if (Test-Path $dir -PathType Container) {
            try {
                # 尝试写入测试文件检查权限
                $testFile = Join-Path $dir ".permission_test"
                "test" | Out-File $testFile -ErrorAction Stop
                Remove-Item $testFile -Force -ErrorAction SilentlyContinue
            }
            catch {
                Write-Warning "目录 '$dir' 无写权限，请手动设置权限"
            }
        }
    }
    
    # 6. 构建Docker镜像
    Write-Info "步骤 6: 构建Docker镜像..."
    try {
        docker-compose build --pull
        Write-Success "Docker镜像构建成功"
    }
    catch {
        Write-Error "Docker镜像构建失败，正在回滚..."
        Rollback-System $backupDir
        throw "部署失败"
    }
    
    # 7. 启动服务
    Write-Info "步骤 7: 启动服务..."
    try {
        docker-compose up -d
        Write-Success "服务启动成功"
    }
    catch {
        Write-Error "服务启动失败，正在回滚..."
        Rollback-System $backupDir
        throw "部署失败"
    }
    
    # 8. 等待服务就绪
    Write-Info "步骤 8: 等待服务就绪..."
    Start-Sleep -Seconds 10
    
    # 9. 检查服务状态
    Write-Info "步骤 9: 检查服务状态..."
    $services = docker-compose ps 2>&1
    if ($services -match "Exit") {
        Write-Error "部分服务启动失败，正在回滚..."
        docker-compose logs
        Rollback-System $backupDir
        throw "部署失败"
    }
    
    # 10. 健康检查
    Write-Info "步骤 10: 执行健康检查..."
    $maxAttempts = 30
    $attempt = 1
    $healthCheckPassed = $false
    
    while ($attempt -le $maxAttempts) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Success "健康检查通过"
                $healthCheckPassed = $true
                break
            }
        }
        catch {
            # 忽略错误，继续重试
        }
        
        Write-Info "等待服务就绪 ($attempt/$maxAttempts)..."
        Start-Sleep -Seconds 5
        $attempt++
    }
    
    if (-not $healthCheckPassed) {
        Write-Error "健康检查超时，正在回滚..."
        docker-compose logs
        Rollback-System $backupDir
        throw "部署失败"
    }
    
    Write-Success "=== Self-Soul AGI 系统部署成功 ==="
    Write-Info "前端访问: http://localhost:5175"
    Write-Info "后端API: http://localhost:8000"
    Write-Info "实时流管理: http://localhost:8766"
    Write-Info "查看日志: docker-compose logs -f"
    Write-Info "停止服务: docker-compose down"
}

# 清理系统
function Cleanup-System {
    Write-Info "正在清理系统..."
    
    # 停止并删除容器
    try {
        docker-compose down --volumes --remove-orphans 2>&1 | Out-Null
        Write-Success "容器清理完成"
    }
    catch {
        Write-Warning "容器清理失败或未运行"
    }
    
    # 删除Docker镜像
    try {
        docker rmi self-soul-agi_backend self-soul-agi_frontend 2>&1 | Out-Null
        Write-Success "镜像清理完成"
    }
    catch {
        Write-Warning "镜像清理失败或不存在"
    }
    
    Write-Success "系统清理完成"
}

# 显示帮助
function Show-Help {
    Write-Host "Self-Soul AGI 系统部署脚本 (Windows PowerShell版本)"
    Write-Host ""
    Write-Host "用法: .\deploy.ps1 [命令]"
    Write-Host ""
    Write-Host "命令:"
    Write-Host "  deploy     部署系统 (默认)"
    Write-Host "  rollback   回滚到上次部署"
    Write-Host "  cleanup    清理所有容器和镜像"
    Write-Host "  help       显示此帮助信息"
    Write-Host ""
    Write-Host "示例:"
    Write-Host "  .\deploy.ps1 deploy"
    Write-Host "  .\deploy.ps1 rollback backup_20250101_120000"
    Write-Host "  .\deploy.ps1 cleanup"
    Write-Host ""
}

# 主函数
function Main {
    param(
        [string]$Command = "deploy",
        [string]$Argument = $null
    )
    
    switch ($Command.ToLower()) {
        "deploy" {
            Deploy-System
        }
        "rollback" {
            if ([string]::IsNullOrEmpty($Argument)) {
                Write-Error "请指定备份目录"
                Write-Info "可用备份:"
                Get-ChildItem -Directory -Filter "backup_*" | ForEach-Object { Write-Host "  $($_.Name)" }
                throw "未指定备份目录"
            }
            Rollback-System $Argument
        }
        "cleanup" {
            Cleanup-System
        }
        "help" {
            Show-Help
        }
        default {
            Write-Error "未知命令: $Command"
            Show-Help
            throw "未知命令"
        }
    }
}

# 执行主函数
try {
    # 启用ANSI颜色支持（Windows 10+）
    if ($Host.UI.RawUI -and $Host.UI.RawUI.SupportsVirtualTerminal) {
        $Host.UI.RawUI.VirtualTerminal = $true
    }
    
    # 解析参数
    $command = $args[0]
    $argument = $args[1]
    
    if ($null -eq $command) {
        $command = "deploy"
    }
    
    Main -Command $command -Argument $argument
}
catch {
    Write-Error "部署失败: $_"
    Write-Info "详细错误信息:"
    Write-Host $_.Exception.Message
    Write-Host $_.ScriptStackTrace
    exit 1
}