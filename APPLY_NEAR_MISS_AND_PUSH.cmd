@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PY=G:\我的雲端硬碟\.quant_envs\antigravity\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERROR] Python environment not found: %PY%
  exit /b 1
)

echo === [1/4] Compile check ===
"%PY%" -m compileall -q worldquant_brain web_ui tests
if errorlevel 1 exit /b 1

echo === [2/4] Run Near-Miss unit tests ===
"%PY%" -m pytest tests\test_alpha_near_miss_refinement.py -q
if errorlevel 1 exit /b 1

echo === [3/4] Run full offline pytest suite ===
"%PY%" -m pytest tests -m "not live and not brain_live and not binance_testnet_live and not bitunix_live and not e2e" -q > reports\stage0_pytest_final.txt 2>&1
if errorlevel 1 (
  echo [ERROR] pytest failed. Check reports\stage0_pytest_final.txt
  exit /b 1
)

echo === [4/4] Git commit, push, and merge to main for Render ===
git add -- worldquant_brain/alpha_miner.py web_ui/server.py tests/test_alpha_near_miss_refinement.py
if errorlevel 1 exit /b 1

git diff --cached --check
if errorlevel 1 exit /b 1

git commit -m "feat(brain): add near-miss alpha auto-refinement engine"
if errorlevel 1 exit /b 1

git push origin fix/integrate-stage0-validated-fixes
if errorlevel 1 exit /b 1

git checkout main
if errorlevel 1 exit /b 1

git pull --ff-only origin main
if errorlevel 1 exit /b 1

git merge --no-ff fix/integrate-stage0-validated-fixes -m "chore(release): merge near-miss alpha auto-refinement engine"
if errorlevel 1 exit /b 1

git push origin main
if errorlevel 1 exit /b 1

git checkout fix/integrate-stage0-validated-fixes
echo [DONE] Near-Miss Alpha Refiner tested and pushed to main!
pause
