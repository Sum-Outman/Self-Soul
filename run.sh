#!/bin/bash
echo "启动Self Soul 项目..."

# 自动检测当前目录
if [ -d "app" ]; then
  cd app
  npm run dev
elif [ -d "../app" ]; then
  cd ../app
  npm run dev
else
  echo "错误：找不到app目录"
  echo "请确保在项目根目录或app同级目录运行此脚本"
fi