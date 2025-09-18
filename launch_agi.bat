@echo off
cls
echo 正在启动Self Soul 开发服务器...

:: 切换到app目录
cd /d %~dp0
cd app

:: 检查Node.js是否安装
where node >nul 2>nul
if %errorlevel% neq 0 (
  echo 错误: 未找到Node.js，请先安装Node.js
  echo 访问 https://nodejs.org/ 下载并安装Node.js
  pause
  exit /b 1
)

:: 检查依赖是否已安装
if not exist "node_modules" (
  echo 正在安装依赖...
  npm install
)

:: 启动开发服务器
start "AGI开发服务器" npm run dev

echo 等待服务启动...
timeout /t 15 >nul

echo 正在打开主页...
start "" "http://localhost:5174/index.html"

echo 启动完成！请访问 http://localhost:5174/index.html
echo 3秒后将自动跳转到主界面，或点击"进入Self Soul 主界面"按钮
echo 如果无法进入主界面，请运行fix_ui.bat修复缓存问题
pause