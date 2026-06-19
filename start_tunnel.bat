@echo off
chcp 65001 >nul
cd /d "%~dp0"
set KEIBA_HOST=0.0.0.0
echo === 競馬AIダッシュボード 起動 ===
echo サーバーを起動します...
start "keiba-server" cmd /k "set KEIBA_HOST=0.0.0.0 && python run_web.py"
echo 数秒待ってトンネルを開きます...
timeout /t 4 >nul
echo.
echo ↓↓ この下に出る https://xxxx.trycloudflare.com をスマホで開いてください ↓↓
echo (このウィンドウとサーバーのウィンドウは開いたままにしてください)
echo.
cloudflared tunnel --url http://localhost:5000
