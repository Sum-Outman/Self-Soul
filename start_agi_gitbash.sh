#!/bin/bash
# Self Soul Git Bash专用启动脚本
echo "正在启动Self Soul 系统..."

# 检测是否在项目根目录
if [ ! -d "app" ]; then
  echo "错误：未找到app目录，请确保在项目根目录(e:/Self Soul )运行此脚本"
  exit 1
fi

# 进入app目录
cd app

# 检查Node.js是否安装
if ! command -v node &> /dev/null; then
  echo "错误：未找到Node.js，请先安装Node.js"
  echo "访问 https://nodejs.org/ 下载并安装Node.js"
  exit 1
fi

# 检查依赖是否已安装
if [ ! -d "node_modules" ]; then
  echo "安装依赖..."
  npm install
fi

# 启动开发服务器
echo "启动AGI开发服务器..."
npm run dev &

# 等待服务启动
echo "等待服务启动..."
sleep 5

# 打开浏览器
echo "正在打开浏览器..."
if command -v start &> /dev/null; then
  start "http://localhost:5175/"
elif command -v xdg-open &> /dev/null; then
  xdg-open "http://localhost:5175/"
elif command -v open &> /dev/null; then
  open "http://localhost:5175/"
else
  echo "无法自动打开浏览器，请手动访问：http://localhost:5175/"
fi

echo "启动完成！请访问 http://localhost:5175/"