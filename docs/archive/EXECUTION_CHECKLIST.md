# Phase 8 Model Governance Hardening

Target: `fix/integrate-stage0-validated-fixes` after Phase 6/7 files have been applied.

## Implemented fixes

- Fail-closed shared model-manifest validation with artifact SHA-256 verification.
- LightGBM remains `SHADOW`; order entry is explicitly disabled.
- PPO is `QUARANTINED`; inference and order entry are explicitly disabled.
- Taiwan ranker is `REPORTING_ONLY`; order entry is explicitly disabled.
- Strategy loaders reject missing approval fields and hash mismatches.
- Master cleanup no longer changes the real `.env` to enable BRAIN simulation or auto-submit.
- Historical crawler writes per-dataset integrity manifests and reports missing candles.
- PPO environment adds a 0.15 position deadband, minimum holding bars, rebalance cooldown, funding cost, configurable turnover penalty, and configurable drawdown penalty.
- Multi-timeframe benchmark rejects results with fewer than 100 trades or fewer than 500 OOS bars and no longer recommends a statistically inadequate timeframe.

## Apply

Extract this ZIP into the repository root and overwrite matching files.

## Verification

```cmd
cd /d "G:\我的雲端硬碟\炒幣"
"G:\我的雲端硬碟\.quant_envs\antigravity\Scripts\python.exe" -m compileall -q crypto_trading scripts tests
"G:\我的雲端硬碟\.quant_envs\antigravity\Scripts\python.exe" -m pytest tests\test_math_models.py tests\test_conditional_prob_strategy.py tests\test_model_governance_phase8.py -q
"G:\我的雲端硬碟\.quant_envs\antigravity\Scripts\python.exe" -m pytest tests -m "not live and not brain_live and not binance_testnet_live and not bitunix_live and not e2e" -q
```

## Required runtime states

- `crypto_lgbm`: `SHADOW`, inference allowed, order entry denied.
- `crypto_ppo`: `QUARANTINED`, inference denied, order entry denied.
- `taiwan_ranker`: `REPORTING_ONLY`, inference allowed, order entry denied.
- `BRAIN_AUTO_SUBMIT_ENABLED`: must remain opt-in and must not be changed by migration or cleanup scripts.

## Explicit staging

Do not stage reports, cached CSVs, model binaries, `.env`, DBs, ZIPs, caches, or runtime JSON.

Suggested source/test paths:

```cmd
git add -- crypto_trading/model_governance.py crypto_trading/dataset_manifest.py crypto_trading/rl_env.py crypto_trading/strategies/ml_strategy.py crypto_trading/strategies/rl_strategy.py scripts/apply_master_fix_and_cleanup.py scripts/fetch_historical_crypto.py scripts/train_models.py scripts/benchmark_timeframes.py tests/test_model_governance_phase8.py data_persistence/models/crypto_lgbm_manifest.json data_persistence/models/crypto_lgbm_weights_manifest.json data_persistence/models/crypto_ppo_agent_manifest.json data_persistence/models/taiwan_stock_manifest.json
```

Then:

```cmd
git diff --cached --check
git commit -m "fix(governance): quarantine failed PPO and harden model, dataset, and benchmark gates"
git push origin fix/integrate-stage0-validated-fixes
```

## Remaining work not falsely marked complete

- Global-date pooled LightGBM split.
- Leave-one-asset-out validation.
- Base/stress/severe cost scenarios.
- Five-seed PPO retraining and promotion gate.
- OU ADF/KPSS and rolling-parameter stability.
- Hurst confidence intervals and regime hysteresis.
- Empirical-Bayes credible lower bounds and minimum effective sample size.
- Moving model binaries to Git LFS or external artifact storage.
