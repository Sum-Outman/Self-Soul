# Self Soul 系统启动脚本 (PowerShell)
Write-Host "正在启动Self Soul 开发服务器..." -ForegroundColor Cyan

# 切换到app目录
Set-Location -Path "app"

# 检查Node.js是否安装
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "错误: 未找到Node.js，请先安装Node.js" -ForegroundColor Red
    Write-Host "访问 https://nodejs.org/ 下载并安装Node.js"
    pause
    exit
}

# 检查依赖是否已安装
if (-not (Test-Path "node_modules")) {
    Write-Host "正在安装依赖..." -ForegroundColor Yellow
    npm install
}

# 启动开发服务器
Start-Process -NoNewWindow -FilePath "npm" -ArgumentList "run dev"

Write-Host "等待服务启动..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

Write-Host "正在打开主页..." -ForegroundColor Cyan
Start-Process "http://localhost:5174/index.html"

Write-Host "启动完成! 请访问 http://localhost:5174/index.html" -ForegroundColor Green
Write-Host "3秒后将自动跳转到主界面，或点击'进入Self Soul 主界面'按钮"
Write-Host "如果无法进入主界面，请运行fix_ui.bat修复缓存问题"
pause