#!/bin/bash
echo "启动Self Soul 项目..."
cd "$(dirname "$0")"

# 启动前端开发服务器
cd app
npm run dev

# 启动后端服务（可选）
# cd ../core
# python main.py