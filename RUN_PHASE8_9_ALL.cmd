@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PY=G:\我的雲端硬碟\.quant_envs\antigravity\Scripts\python.exe"

if not exist "%PY%" (
  echo [ERROR] Python environment not found: %PY%
  exit /b 1
)

echo === [1/6] Compile check ===
"%PY%" -m compileall -q crypto_trading scripts tests
if errorlevel 1 exit /b 1

echo === [2/6] Offline pytest suite ===
"%PY%" -m pytest tests -m "not live and not brain_live and not binance_testnet_live and not bitunix_live and not e2e" -q > reports\stage0_pytest_final.txt 2>&1
if errorlevel 1 (
  echo [ERROR] pytest failed. Check reports\stage0_pytest_final.txt
  exit /b 1
)

echo === [3/6] LightGBM Global-Date / LOAO / Cost-Stress Validation ===
"%PY%" scripts\validate_lgbm_oos_and_costs.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 1m --horizon 15 --update-manifests
if errorlevel 1 exit /b 1

echo === [4/6] PPO V2 5-Seed Retraining and Gate ===
"%PY%" scripts\retrain_ppo_v2_5seed.py --episodes 8 --max-bars 90000
if errorlevel 1 exit /b 1

echo === [5/6] Re-verify Governance & Gate Tests ===
"%PY%" -m pytest tests\test_model_governance_phase8.py tests\test_lgbm_validation_gate.py -q
if errorlevel 1 exit /b 1

echo === [6/6] Git commit, push, and merge to main for Render ===
git add -- worldquant_brain/brain_client.py crypto_trading/model_governance.py crypto_trading/dataset_manifest.py crypto_trading/lgbm_validation_gate.py crypto_trading/rl_env.py crypto_trading/strategies/ml_strategy.py crypto_trading/strategies/rl_strategy.py scripts/apply_master_fix_and_cleanup.py scripts/fetch_historical_crypto.py scripts/train_models.py scripts/benchmark_timeframes.py scripts/validate_lgbm_oos_and_costs.py scripts/retrain_ppo_v2_5seed.py tests/test_model_governance_phase8.py tests/test_lgbm_validation_gate.py data_persistence/models/crypto_lgbm_manifest.json data_persistence/models/crypto_lgbm_weights_manifest.json data_persistence/models/crypto_ppo_agent_manifest.json data_persistence/models/taiwan_stock_manifest.json
if errorlevel 1 exit /b 1

git diff --cached --check
if errorlevel 1 exit /b 1

git commit -m "fix(governance): apply phase 8-9 governance, OOS stress gate, and 5-seed PPO gate"
if errorlevel 1 exit /b 1

git push origin fix/integrate-stage0-validated-fixes
if errorlevel 1 exit /b 1

git checkout main
if errorlevel 1 exit /b 1

git pull --ff-only origin main
if errorlevel 1 exit /b 1

git merge --no-ff fix/integrate-stage0-validated-fixes -m "chore(release): merge verified phase 8-9 governance and validation gate"
if errorlevel 1 exit /b 1

git push origin main
if errorlevel 1 exit /b 1

git checkout fix/integrate-stage0-validated-fixes
echo [DONE] All steps passed and pushed to main!
pause
