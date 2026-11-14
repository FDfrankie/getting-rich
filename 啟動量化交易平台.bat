@echo off
chcp 65001 >nul
title Antigravity 量化操盤平台一鍵啟動
echo ========================================================
echo   Antigravity 自動化量化操盤平台 (台股 + 加密合約 + WorldQuant)
echo ========================================================
echo.
echo [1] 正在開啟雲端 OnRender Web 介面...
start https://getting-rich.onrender.com/
echo.
echo [2] 正在檢查本機 Python 操盤核心伺服器...
timeout /t 2 >nul
echo.
echo 若需於本機運行本地伺服器，請按任意鍵啟動本地 Web 服務 (連接埠 8999)：
pause >nul
start http://localhost:8999
C:\Users\mis\miniconda3\envs\dlcv\python.exe web_ui/app.py
pause
