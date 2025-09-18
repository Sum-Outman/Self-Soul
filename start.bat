@echo off
REM 切换到项目根目录
cd /d %~dp0

REM 启动前端开发服务器
cd app
npm run dev

REM 启动后端服务（可选）
REM cd ../core
REM python main.py