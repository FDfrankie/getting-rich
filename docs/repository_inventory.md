# Antigravity Quant Repository Inventory (Phase 1 Consolidation)
- Generated At: `2026-11-18T07:39:20.549878+00:00`
- Total Tracked Files: **234**

## Category Breakdown
| Category | Count |
| :--- | ---: |
| `DATA` | 22 |
| `DOCUMENTATION` | 25 |
| `MIGRATION` | 8 |
| `MODEL_ARTIFACT` | 7 |
| `REPORT` | 6 |
| `RUNTIME_STATE` | 8 |
| `SECRET_RISK` | 1 |
| `SOURCE` | 102 |
| `TEST` | 55 |

## Five-Plane Architecture Mapping
- **DATA PLANE**: `taiwan_stock/universe_registry.py`, `crypto_trading/data_loader.py`, `crypto_trading/dataset_manifest.py`, `data_persistence/migrations.py`
- **RESEARCH PLANE**: `crypto_trading/feature_engineer.py`, `crypto_trading/indicator_library.py`, `crypto_trading/lgbm_validation_gate.py`, `worldquant_brain/alpha_miner.py`, `worldquant_brain/meta_model.py`
- **DECISION PLANE**: `worldquant_brain/candidate_utility.py`, `worldquant_brain/daily_candidate_selector.py`, `crypto_trading/math_models.py`, `crypto_trading/strategies/`
- **EXECUTION PLANE**: `crypto_trading/bot_runner.py`, `crypto_trading/shadow_trading.py`, `worldquant_brain/submission_service.py`, `worldquant_brain/direct_submit_service.py`
- **GOVERNANCE PLANE**: `crypto_trading/model_governance.py`, `worldquant_brain/alpha_lifecycle.py`, `worldquant_brain/score_tracker.py`, `worldquant_brain/score_attribution.py`, `observability/`
