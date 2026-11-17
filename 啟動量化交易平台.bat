@echo off
chcp 65001 >nul
title Antigravity 量化操盤平台一鍵啟動
echo ========================================================
echo   Antigravity 自動化量化操盤平台 (台股 + 加密合約 + WorldQuant)
echo ========================================================
echo.
set "VENV_PY=G:\我的雲端硬碟\.quant_envs\antigravity\Scripts\python.exe"
if not exist "%VENV_PY%" set "VENV_PY=python"
echo [1] 正在啟動本機 Web 操盤與 WorldQuant 遠端送測服務 (Port 8000)...
start http://127.0.0.1:8000
"%VENV_PY%" -m web_ui.server
pause
