#!/bin/bash

# 检查Git是否安装
if ! command -v git &> /dev/null
then
    echo "Git未安装，请先安装Git。"
    echo "在Ubuntu/Debian上：sudo apt install git"
    echo "在macOS上：brew install git 或从App Store安装Xcode Command Line Tools"
    exit 1
fi

# 设置新的仓库URL
NEW_REPO_URL="https://github.com/Sum-Outman/Self-Soul"

# 检查是否存在.git目录
if [ ! -d ".git" ]; then
    echo "初始化新的Git仓库..."
    git init
fi

# 检查.gitignore文件是否存在
if [ ! -f ".gitignore" ]; then
    echo "创建.gitignore文件..."
    echo "# 虚拟环境" >> .gitignore
    echo ".venv/" >> .gitignore
    echo "venv/" >> .gitignore
    echo "ENV/" >> .gitignore
    echo "env/" >> .gitignore
    echo "agi_env_311/" >> .gitignore
    echo "" >> .gitignore
    
    echo "# IDE配置" >> .gitignore
    echo ".vscode/" >> .gitignore
    echo ".idea/" >> .gitignore
    echo "" >> .gitignore
    
    echo "# 操作系统文件" >> .gitignore
    echo ".DS_Store" >> .gitignore
    echo "Thumbs.db" >> .gitignore
    echo "" >> .gitignore
    
    echo "# 日志文件" >> .gitignore
    echo "logs/" >> .gitignore
    echo "*.log" >> .gitignore
    echo "" >> .gitignore
    
    echo "# 依赖目录" >> .gitignore
    echo "node_modules/" >> .gitignore
fi

# 添加所有文件到Git
echo "添加所有文件到Git..."
git add .

# 提交更改
echo "提交更改..."
git commit -m "Initial commit to new repository"

# 添加远程仓库
echo "添加新的远程仓库..."
git remote remove origin 2> /dev/null
git remote add origin $NEW_REPO_URL

# 推送到远程仓库
echo "推送到新的GitHub仓库..."
git push -u origin master --force

# 检查推送是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "项目已成功推送到新的GitHub仓库：$NEW_REPO_URL"
    echo "请访问 https://github.com/Sum-Outman/Self-Soul 查看仓库。"
    echo ""
    echo "注意事项："
    echo "1. 虚拟环境目录 agi_env_311/ 已通过.gitignore排除，不会被推送到仓库。"
    echo "2. 旧的开源信息已清除，现在项目完全开源到新的仓库。"
else
    echo ""
    echo "推送失败！请检查以下几点："
    echo "1. 确保您已在GitHub上创建了 $NEW_REPO_URL 仓库。"
    echo "2. 确保您有足够的权限推送到该仓库。"
    echo "3. 确保您的网络连接正常。"
fi

echo "按Enter键退出..."
read -r