@echo off

REM 检查Git是否安装
where git >nul 2>nul
if %errorlevel% neq 0 (
echo Git is not installed. Please install Git first.
echo You can download it from https://git-scm.com/downloads.
pause
exit /b 1
)

REM 设置新的仓库URL
set NEW_REPO_URL=https://github.com/Sum-Outman/Self-Soul

REM 检查是否存在.git目录
if not exist .git (
echo Initializing new Git repository...
git init
)

REM 检查.gitignore文件是否存在，如果不存在则创建
if not exist .gitignore (
echo Creating .gitignore file...
echo # Virtual environments >> .gitignore
echo .venv/ >> .gitignore
echo venv/ >> .gitignore
echo ENV/ >> .gitignore
echo env/ >> .gitignore
echo agi_env_311/ >> .gitignore
echo # IDE >> .gitignore
echo .vscode/ >> .gitignore
echo .idea/ >> .gitignore
)

REM 添加所有文件到暂存区
echo Adding files to staging area...
git add .

REM 提交更改
echo Committing changes...
git commit -m "Initial commit for Self-Soul"

REM 检查是否已有origin远程仓库
for /f "delims=" %%i in ('git remote') do set remote_exists=%%i
if defined remote_exists (
echo Removing existing remote repository...
git remote remove origin
)

REM 添加新的远程仓库
echo Adding new remote repository...
git remote add origin %NEW_REPO_URL%

REM 推送代码到远程仓库
echo Pushing code to remote repository...
git push -u origin master --force

REM 检查推送是否成功
if %errorlevel% equ 0 (
echo.
echo Success! The code has been pushed to %NEW_REPO_URL%
echo.
echo Notes:
echo 1. The virtual environment directory agi_env_311/ is excluded via .gitignore and will not be pushed to the repository.
echo 2. Old open source information has been cleared, and the project is now fully open source in the new repository.
echo 3. Please visit https://github.com/Sum-Outman/Self-Soul to view the repository.
) else (
echo.
echo Push failed! Please check the following:
echo 1. Make sure you have created the %NEW_REPO_URL% repository on GitHub.
echo 2. Make sure you have sufficient permissions to push to this repository.
echo 3. Make sure your network connection is normal.
echo 4. You may need to configure Git credentials using 'git config --global user.name "Your Name"' and 'git config --global user.email "your.email@example.com"'
)

pause