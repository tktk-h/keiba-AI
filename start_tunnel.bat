@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === 競馬AI ダッシュボード ===
echo サーバーを起動中(数秒)...
set KEIBA_HOST=127.0.0.1
start "" /b python run_web.py
timeout /t 6 >nul
echo.
echo ============================================================
echo  この下に出る https://xxxx.trycloudflare.com を
echo  スマホで開いてください。
echo  （このウィンドウは開いたままにする / 閉じると停止します）
echo ============================================================
echo.
"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Cloudflare.cloudflared_Microsoft.Winget.Source_8wekyb3d8bbwe\cloudflared.exe" tunnel --url http://localhost:5000
pause
