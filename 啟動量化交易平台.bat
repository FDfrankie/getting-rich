@echo off
setlocal
cd /d "%~dp0"
set "VENV_PY=%~dp0..\.quant_envs\antigravity\Scripts\python.exe"
if not exist "%VENV_PY%" set "VENV_PY=python"
echo Starting Antigravity Quant Web UI on http://127.0.0.1:8999 ...
start "" cmd /c "timeout /t 5 /nobreak >nul & start http://127.0.0.1:8999"
"%VENV_PY%" web_ui\app.py
pause
