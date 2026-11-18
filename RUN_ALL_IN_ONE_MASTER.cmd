@echo off
setlocal
cd /d "%~dp0"
set "PY=%~dp0..\.quant_envs\antigravity\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo === [1/5] Compile check across all packages ===
"%PY%" -m compileall -q crypto_trading taiwan_stock worldquant_brain observability web_ui scripts tests
if errorlevel 1 exit /b 1

echo === [2/5] Migrate Phase 9/10 SQLite governance schema ===
if exist "worldquant_brain\alpha_vault.db" (
  "%PY%" scripts\migrate_phase9_brain_governance.py --db worldquant_brain\alpha_vault.db
  if errorlevel 1 exit /b 1
)

echo === [3/5] Run full offline pytest suite ===
"%PY%" -m pytest tests -m "not live and not brain_live and not binance_testnet_live and not bitunix_live and not e2e" -q > reports\stage0_pytest_final.txt 2>&1
if errorlevel 1 (
  echo [ERROR] Offline pytest failed. See reports\stage0_pytest_final.txt
  exit /b 1
)

echo === [4/5] Run LightGBM Strict OOS/LOAO/Cost-Stress Gate & PPO V2 5-Seed Gate ===
"%PY%" scripts\validate_lgbm_oos_and_costs.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 1m --horizon 15 --update-manifests
if errorlevel 1 exit /b 1

"%PY%" scripts\retrain_ppo_v2_5seed.py --episodes 8 --max-bars 90000
if errorlevel 1 exit /b 1

echo === [5/5] Stage verified source files, commit, and push to main for Render ===
git add -- ^
  .github/workflows/brain_miner.yml ^
  .github/workflows/future-offline-ci.yml ^
  render.yaml ^
  deployment/render.future.yaml ^
  FUTURE_READINESS_ROADMAP.md ^
  RESEARCH_SOURCES.md ^
  research_catalog/video_catalog.json ^
  research_catalog/strategy_hypotheses.yaml ^
  research_catalog/research_backlog.json ^
  crypto_trading/brain_feature_bridge.py ^
  crypto_trading/feature_engineer.py ^
  crypto_trading/indicator_library.py ^
  crypto_trading/model_governance.py ^
  taiwan_stock/universe_registry.py ^
  crypto_trading/dataset_manifest.py ^
  crypto_trading/lgbm_validation_gate.py ^
  crypto_trading/rl_env.py ^
  crypto_trading/strategies/__init__.py ^
  crypto_trading/strategies/brain_alpha_strategy.py ^
  crypto_trading/strategies/ml_strategy.py ^
  crypto_trading/strategies/rl_strategy.py ^
  observability/__init__.py ^
  observability/audit_log.py ^
  worldquant_brain/alpha_lifecycle.py ^
  worldquant_brain/alpha_experiment_store.py ^
  worldquant_brain/brain_scheduler.py ^
  worldquant_brain/manual_priority_queue.py ^
  worldquant_brain/score_tracker.py ^
  worldquant_brain/adaptive_submission.py ^
  worldquant_brain/alpha_miner.py ^
  worldquant_brain/alpha_vault.py ^
  worldquant_brain/brain_client.py ^
  worldquant_brain/brain_job_queue.py ^
  web_ui/server.py ^
  web_ui/static/index.html ^
  scripts/apply_master_fix_and_cleanup.py ^
  scripts/fetch_historical_crypto.py ^
  scripts/train_models.py ^
  scripts/benchmark_timeframes.py ^
  scripts/validate_lgbm_oos_and_costs.py ^
  scripts/retrain_ppo_v2_5seed.py ^
  scripts/headless_brain_miner.py ^
  scripts/migrate_phase9_brain_governance.py ^
  scripts/brain_ops.py ^
  scripts/brain_score.py ^
  scripts/build_research_backlog.py ^
  scripts/sync_taiwan_universe.py ^
  tests/test_model_governance_phase8.py ^
  tests/test_lgbm_validation_gate.py ^
  tests/test_alpha_near_miss_refinement.py ^
  tests/test_headless_brain_miner.py ^
  tests/test_alpha_vault_discovered_ui.py ^
  tests/test_phase9_brain_operations.py ^
  tests/test_phase9_server_callbacks.py ^
  tests/test_phase10_score_policy.py ^
  tests/test_taiwan_universe_and_yuanta_ui.py ^
  tests/test_future_foundations.py ^
  data_persistence/models/crypto_lgbm_manifest.json ^
  data_persistence/models/crypto_lgbm_weights_manifest.json ^
  data_persistence/models/crypto_ppo_agent_manifest.json ^
  data_persistence/models/taiwan_stock_manifest.json
if errorlevel 1 exit /b 1

git diff --cached --check
if errorlevel 1 exit /b 1

git commit -m "feat(all-in-one): integrate phase 8-10 governance, direct submit, daily max-profit top-2 flush, UI fix, and OOS gates"
if errorlevel 1 exit /b 1

git push origin fix/integrate-stage0-validated-fixes
if errorlevel 1 exit /b 1

git checkout main
if errorlevel 1 exit /b 1

git pull --ff-only origin main
if errorlevel 1 exit /b 1

git merge --no-ff fix/integrate-stage0-validated-fixes -m "chore(release): merge verified phase 8-10 all-in-one master upgrade"
if errorlevel 1 exit /b 1

git push origin main
if errorlevel 1 exit /b 1

git checkout fix/integrate-stage0-validated-fixes
echo [DONE] All-in-one upgrade verified and pushed to main!
exit /b 0
