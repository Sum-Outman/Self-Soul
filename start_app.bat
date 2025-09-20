@echo off
echo 正在启动Self Soul 开发服务器...
cd app
start "" npm run dev
timeout /t 10 >nul
echo 正在打开主应用界面...
start "" "http://localhost:5175/"
echo 启动完成！请访问 http://localhost:5175/
pause