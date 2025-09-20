@echo off
REM 切换到项目根目录
cd /d %~dp0

REM 启动前端开发服务器（端口5175）
start "AGI Frontend" cmd /k "cd app && npm run dev"

REM 等待前端启动
ping 127.0.0.1 -n 5 > nul

REM 启动主API网关（端口8000）
start "AGI Backend" cmd /k "cd core && python main.py"

REM 显示启动信息
ECHO AGI系统已启动！前端: http://localhost:5175/，后端API: http://localhost:8000/
ECHO 请按任意键停止所有服务...

REM 等待用户输入终止
pause > nul

REM 终止所有服务
taskkill /FI "WINDOWTITLE eq AGI Frontend" /F
 taskkill /FI "WINDOWTITLE eq AGI Backend" /F
ECHO AGI系统已停止