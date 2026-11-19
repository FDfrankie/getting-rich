@echo off
setlocal
cd /d "%~dp0"
title Antigravity Quant Platform (Port 8999)

set "VENV_PY=%~dp0..\.quant_envs\antigravity\Scripts\python.exe"
if not exist "%VENV_PY%" set "VENV_PY=python"

if "%~1"=="" (
    if exist "%~dp0data_persistence\watchdog.paused" del /f /q "%~dp0data_persistence\watchdog.paused" >nul 2>&1
)

netstat -ano | findstr /R /C:":8999 .*LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Antigravity Quant Web UI is already running on port 8999.
    if "%~1"=="" start http://127.0.0.1:8999
    exit /b 0
)

echo Starting Antigravity Quant Web UI on http://127.0.0.1:8999 ...
if /I not "%~1"=="--watchdog" if /I not "%~1"=="--no-browser" (
    start "" cmd /c "timeout /t 5 /nobreak >nul & start http://127.0.0.1:8999"
)

:SERVER_LOOP
netstat -ano | findstr /R /C:":8999 .*LISTENING" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Port 8999 is already active in another process. Exiting duplicate launcher.
    exit /b 0
)
"%VENV_PY%" -B -u web_ui\app.py
if exist "%~dp0data_persistence\watchdog.paused" (
    echo [INFO] Auto-restart is paused via data_persistence\watchdog.paused.
    exit /b 0
)
echo [%DATE% %TIME%] web_ui\app.py stopped (exit code %ERRORLEVEL%). Auto-restarting in 3 seconds...
timeout /t 3 /nobreak >nul
goto SERVER_LOOP
