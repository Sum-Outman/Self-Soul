#!/bin/bash
 echo "启动Self Brain AGI系统..."
 cd "$(dirname "$0")"

 # 启动前端开发服务器（端口5175）
 cd app
 npm run dev &
 FRONTEND_PID=$!

 # 等待前端启动
 sleep 3

 # 返回根目录
 cd ..

 # 启动主API网关（端口8000）
 cd core
 python main.py &
 BACKEND_PID=$!

 echo "AGI系统已启动！前端: http://localhost:5175/，后端API: http://localhost:8000/"

 # 等待用户输入终止
 read -p "按Enter键停止系统..."

 # 终止所有服务
 kill $FRONTEND_PID $BACKEND_PID
 echo "AGI系统已停止"