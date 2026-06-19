@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === 競馬AI ダッシュボード (PC表示) ===
echo ブラウザが自動で開きます: http://localhost:5000
echo (このウィンドウは開いたまま。閉じると停止します)
python run_web.py
pause
