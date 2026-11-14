@echo off
chcp 65001 >nul
title Cloudflare Tunnel 免費公網穿透 (手機看盤神器)
echo ========================================================
echo   Cloudflare Tunnel 一鍵公網穿透工具
echo ========================================================
echo.
echo 正在將本機量化操盤平台 (http://localhost:8999) 穿透至全球安全公網...
echo 穿透成功後，將輸出一串 https://xxx.trycloudflare.com 專屬安全網址。
echo 您可直接複製該網址，在手機 (iPhone / Android) 瀏覽器隨處連線操盤！
echo.
cloudflared tunnel --url http://localhost:8999
if errorlevel 1 (
    echo.
    echo [!] 提示: 若尚未下載 cloudflared 工具，請至 Cloudflare 官方下載或使用 winget:
    echo     winget install --id Cloudflare.cloudflared
    echo.
)
pause
