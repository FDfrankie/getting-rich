@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PY=G:\我的雲端硬碟\.quant_envs\antigravity\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERROR] Python environment not found: %PY%
  exit /b 1
)

echo === [1/3] Compile & test headless miner ===
"%PY%" -m compileall -q scripts\headless_brain_miner.py tests\test_headless_brain_miner.py
if errorlevel 1 exit /b 1

"%PY%" -m pytest tests\test_headless_brain_miner.py -q
if errorlevel 1 exit /b 1

echo === [2/3] Stage & commit GitHub Actions workflow ===
git add -- .github/workflows/brain_miner.yml scripts/headless_brain_miner.py tests/test_headless_brain_miner.py
if errorlevel 1 exit /b 1

git diff --cached --check
if errorlevel 1 exit /b 1

git commit -m "ci(brain): add hourly GitHub Actions headless alpha miner workflow"
if errorlevel 1 exit /b 1

git push origin fix/integrate-stage0-validated-fixes
if errorlevel 1 exit /b 1

echo === [3/3] Merge into main for GitHub Actions schedule ===
git checkout main
if errorlevel 1 exit /b 1

git pull --ff-only origin main
if errorlevel 1 exit /b 1

git merge --no-ff fix/integrate-stage0-validated-fixes -m "chore(release): enable GitHub Actions hourly headless alpha miner"
if errorlevel 1 exit /b 1

git push origin main
if errorlevel 1 exit /b 1

git checkout fix/integrate-stage0-validated-fixes
echo [DONE] GitHub Actions workflow pushed to main!
pause
