@echo off
REM Self Soul 智能启动脚本 (Windows)

REM 检测当前目录
if exist "app\" (
  echo 当前目录: %cd%
  cd app
) else if exist "..\app\" (
  echo 切换到上级目录
  cd ..
  cd app
) else (
  echo 错误：未找到app目录
  echo 请确保在项目根目录(e:\Self Soul )或其子目录中运行此脚本
  pause
  exit /b 1
)

REM 检查依赖
if not exist "node_modules\" (
  echo 安装依赖...
  npm install
)

REM 启动服务
echo 启动AGI开发服务器...
start "AGI服务器" npm run dev

REM 获取服务URL
echo 服务已启动！请访问:
echo http://localhost:5175/
echo 等待5秒后自动打开浏览器...
ping -n 5 127.0.0.1 > nul
start "" "http://localhost:5175/"
pause