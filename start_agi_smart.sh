#!/bin/bash
# Self Soul 智能启动脚本

# 检测当前目录
if [ -d "app" ]; then
  echo "当前目录: $(pwd)"
  cd app
elif [ -d "../app" ]; then
  echo "切换到上级目录"
  cd ..
  cd app
else
  echo "错误：未找到app目录"
  echo "请确保在项目根目录(e:/Self Soul )或其子目录中运行此脚本"
  exit 1
fi

# 检查依赖
if [ ! -d "node_modules" ]; then
  echo "安装依赖..."
  npm install
fi

# 启动服务
echo "启动AGI开发服务器..."
npm run dev

# 获取服务URL
echo "服务已启动！请访问:"
echo "http://localhost:5174/"