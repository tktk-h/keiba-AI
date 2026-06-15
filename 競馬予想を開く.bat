@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Starting Keiba app... your browser will open automatically.
echo To stop: close this window or press Ctrl + C.
python run_web.py
pause
