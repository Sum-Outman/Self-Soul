#!/bin/bash
# Self Soul 系统启动脚本 (Bash)
echo -e "\033[36m正在启动Self Soul 开发服务器...\033[0m"

# 切换到app目录
cd app

# 检查Node.js是否安装
if ! command -v node &> /dev/null; then
    echo -e "\033[31m错误: 未找到Node.js，请先安装Node.js\033[0m"
    echo "访问 https://nodejs.org/ 下载并安装Node.js"
    read -p "按回车键退出..."
    exit 1
fi

# 检查依赖是否已安装
if [ ! -d "node_modules" ]; then
    echo -e "\033[33m正在安装依赖...\033[0m"
    npm install
fi

# 启动开发服务器
npm run dev &

echo -e "\033[33m等待服务启动...\033[0m"
sleep 15

echo -e "\033[36m正在打开主页...\033[0m"
if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:5174/index.html"
elif command -v open &> /dev/null; then
    open "http://localhost:5174/index.html"
else
    start "" "http://localhost:5174/index.html"
fi

echo -e "\033[32m启动完成! 请访问 http://localhost:5174/index.html\033[0m"
echo "3秒后将自动跳转到主界面，或点击'进入Self Soul 主界面'按钮"
echo "如果无法进入主界面，请运行./fix_ui.sh修复缓存问题"
read -p "按回车键退出..."