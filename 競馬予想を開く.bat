@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo 競馬予想アプリを起動中... ブラウザが自動で開きます。
echo 終了するときは、この黒い画面で Ctrl + C を押すか、画面を閉じてください。
python run_web.py
pause
