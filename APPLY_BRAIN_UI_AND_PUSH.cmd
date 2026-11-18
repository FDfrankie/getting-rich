@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PY=G:\我的雲端硬碟\.quant_envs\antigravity\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERROR] Python environment not found: %PY%
  exit /b 1
)

echo === [1/3] Compile & test Alpha Vault UI/Lifecycle ===
"%PY%" -m compileall -q worldquant_brain web_ui tests
if errorlevel 1 exit /b 1

"%PY%" -m pytest tests\test_alpha_vault_discovered_ui.py -q
if errorlevel 1 exit /b 1

"%PY%" -m pytest tests -m "not live and not brain_live and not binance_testnet_live and not bitunix_live and not e2e" -q > reports\stage0_pytest_final.txt 2>&1
if errorlevel 1 (
  echo [ERROR] pytest failed. Check reports\stage0_pytest_final.txt
  exit /b 1
)

echo === [2/3] Stage & commit UI/Lifecycle fix ===
git add -- worldquant_brain/alpha_vault.py web_ui/server.py web_ui/static/index.html tests/test_alpha_vault_discovered_ui.py
if errorlevel 1 exit /b 1

git diff --cached --check
if errorlevel 1 exit /b 1

git commit -m "fix(brain-ui): separate Alpha lifecycle states, fix vault stats, and show discovered Alphas"
if errorlevel 1 exit /b 1

git push origin fix/integrate-stage0-validated-fixes
if errorlevel 1 exit /b 1

echo === [3/3] Merge into main for Render ===
git checkout main
if errorlevel 1 exit /b 1

git pull --ff-only origin main
if errorlevel 1 exit /b 1

git merge --no-ff fix/integrate-stage0-validated-fixes -m "chore(release): merge Alpha Vault discovered UI and lifecycle fix"
if errorlevel 1 exit /b 1

git push origin main
if errorlevel 1 exit /b 1

git checkout fix/integrate-stage0-validated-fixes
echo [DONE] Brain UI & Lifecycle fix pushed to main!
pause
