# AI Master Handover Guide — Antigravity Quant Platform

> **Purpose**: Single-source-of-truth onboarding document for any AI agent or engineer taking over `G:\我的雲端硬碟\炒幣` (`Antigravity Quant`). Read this + `GEMINI.md` before making changes.

---

## 1. System Architecture & Runtime Topology

Antigravity Quant is a 24/7 multi-market quantitative research & execution platform covering **WorldQuant BRAIN (IQC 2026)**, **Crypto Futures (Binance Testnet / Bitunix)**, **Taiwan Stock (TWSE/TPEx)**, and **US Stock (SEC EDGAR PIT)**.

| Component | Entrypoint / Module | Runtime / Schedule |
| :--- | :--- | :--- |
| **Web UI & API Server** | `web_ui/app.py` -> `web_ui/server.py` (Port `8999`) | Windows ScheduledTask `Quant_WebUI` (via `scripts/silent_run.vbs`), OnRender cloud (`render.yaml`) |
| **WorldQuant BRAIN Miner** | `scripts/headless_brain_miner.py` | Windows ScheduledTask `Quant_BrainMiner` (24/7, 2 workers, batch-size 5) |
| **Watchdog Guardian** | `scripts/watchdog_check.ps1` | Windows ScheduledTask `Quant_Watchdog` (enforces single-instance miner; respects `data_persistence/watchdog.paused`) |
| **Headless Crypto Bot** | `scripts/headless_crypto_bot.py` | Multi-symbol (`BTC/USDT`, `ETH/USDT`, `SOL/USDT`, `BNB/USDT`), `1h` causal strategies |
| **Python Environment** | `G:\我的雲端硬碟\.quant_envs\antigravity\Scripts\python.exe` | Active Windows venv (outside repo root; never recreate `.venv/` or `quant_envs/` inside repo) |

---

## 2. Dual-SQLite & Google Drive Sync Isolation

Because the repo lives inside Google Drive (`G:\我的雲端硬碟\炒幣`), opening SQLite WAL databases directly inside Google Drive causes `database is locked` / `disk I/O error`.

- **Live State Directory**: `data_persistence/state_paths.py` automatically redirects live SQLite DBs to local NTFS **`C:\quant_envs\state\`**:
  - `C:\quant_envs\state\alpha_vault.db` (BRAIN alphas, candidate pool, `qualified_alpha_archive`, funnel RL/reference DB)
  - `C:\quant_envs\state\system_state.db` (Crypto orders, positions, equity snapshots, shadow `order_intents`, `audit_events`)
- **Snapshot Sync**: `sqlite3.backup()` periodically mirrors snapshots back to `worldquant_brain/alpha_vault.db` and `data_persistence/system_state.db`, plus JSON exports (`data_persistence/qualified_alpha_formulas_vault.json`, `observability/qualified_alpha_formulas_vault.json`, `data_persistence/brain_pending_status.json`) and `.xlsx` reports in `BRAIN_官方知識庫與研究紀錄/`.
- **SQLite Resilience**: `worldquant_brain/sqlite_resilience.py` provides `managed_connection()` (guaranteed close + WAL/DELETE pragma selection), `ensure_schema_from_ddl()` (idempotent column auto-migration), and `execute_with_sqlite_retry()`. Always use `managed_connection()` rather than `with open_sqlite_wal():` (raw `sqlite3.Connection` context manager commits but does not close).
- **Schema Migrations**: Idempotent SQL files in `migrations/006`–`012` (with rollbacks) plus `data_persistence/migrations.py` (`apply_sqlite_migrations`).

---

## 3. WorldQuant BRAIN & IQC 2026 Pipeline (P0–P43 Evolution)

### Core Pipeline (`worldquant_brain/`)
1. **Two-Stage RL-Guided Funnel (`two_stage_funnel.py`)**:
   - **Stage 1**: Curated economic seeds (`V8_ECONOMIC_HYPOTHESIS_SEEDS`, 69 Bronze-unlocked seeds across P30–P42) + UCB1 contextual bandit over `STAGE1_PROBES` (strictly 66 Bronze-unlocked operators in `BRONZE_UNLOCKED_OPERATORS`).
   - **Pre-Sim FastExpr Linter (`fastexpr_linter.py`, P42)**: Enforces KB v78 `FE01–FE15` rules before spending BRAIN simulation quota, plus `min_decay_for_expression()` (option `decay >= 6` turnover floor).
   - **Stage 2**: Immediate in-session near-miss refinement (sign-flip, turnover crush, `hump`, `trade_when`, `group_rank`, `KB-304`/`KB-430` SC-breakers, neutralization rotation).
2. **Submit Channel, Consultant Reserve & Auto-Prune (`brain_client.py`, P37–P43)**:
   - **Bronze/Silver Mode (default)**: `BRAIN_SUBMIT_ORDER=worst_first` submits lowest-fitness qualifying alphas first while protecting high-value gems (`Sharpe >= 1.50` or `Fitness >= 1.20`) in `CONSULTANT_RESERVE`.
   - **Full-History Backfill & Export (`scripts/backfill_brain_passed_alphas.py`, P42–P43)**: Paginates official `/users/self/alphas` (`ACTIVE` + `UNSUBMITTED`), archives into `qualified_alpha_archive`, and exports `BRAIN_官方知識庫與研究紀錄/worldquant_brain_qualified_alphas_latest.xlsx`.
   - **Official Hard Gates (`pre_submission_gate.py`, `iqc_scoring.py`, `pnl_jump_scoring.py`)**: `WORLDQUANT_BRAIN` provenance only, `D1 Sharpe > 1.25` / `D0 Sharpe > 2.00`, `Fitness > 1.00`, `0.01 < Turnover < 0.70`, official `SELF` PnL correlation `< 0.70`, max weight `< 0.10`, and all official IS checks passing.
3. **Read-Only Diagnostic APIs (`web_ui/server.py`)**:
   - `/api/brain/alphas/{id}/improvement-plan`, `/pnl-diagnosis`, `/improvement-checklist`, `/correlations`, `/top-correlated`, `/delay-pair`, `/neutralization-ablation`
   - `/api/brain/iqc/candidates`, `/daily-plan`, `POST /api/brain/iqc/recalculate`

---

## 4. Crypto, Taiwan Stock & US Stock Modules (P44)

- **Crypto (`crypto_trading/`, `portfolio/`)**:
  - Execution guardrails: `instruments.py` (`step_size`/`tick_size`/`min_quantity`), `risk_manager.py`, `order_manager.py` (Decimal idempotency), `order_execution_service.py`, `reconciliation_service.py`.
  - Strategies: `NativeCryptoEnsembleStrategy` (`1h`, causal, zero-lookahead, `ConditionalProbRL` + `PerpCarryBasis` + `HierarchicalRegimeFilter` + `execution_cost_filter`), while unverified ML/RL models (`crypto_lgbm`, `crypto_ppo`) remain strictly quarantined in `SHADOW` via `model_governance/` (`validate_model_manifest`).
  - Offline data cache: `crypto_trading/data_cache/*_1h_180d.csv`.
- **Taiwan Stock (`taiwan_stock/`)**: `universe_registry.py` (point-in-time TWSE/TPEx + 6-layer tradability funnel), `ml_ranker.py`, `market_analyzer.py`, `report_generator.py`, `scheduler.py` (Asia/Taipei 16:30 & 08:00).
- **US Stock (`us_stock/`)**: `sec_edgar_client.py` (PIT company facts) and `ml_ranker.py` (exports wide `.pkl` panel for `worldquant_brain/local_cs_screener.py`).

---

## 5. Development, Testing & Operational Commands

- **Run Targeted Pytest Suites**:
  ```powershell
  G:\我的雲端硬碟\.quant_envs\antigravity\Scripts\python.exe -m pytest -v tests/test_p39_kill_neg_sharpe_probes_and_rreonlgn_patterns.py tests/test_p40_kb304_cap_ts_family_near_miss_rescue_and_corr_breaker.py tests/test_p41_kb430_live_ab_scale_promotion_and_sc_breakers.py tests/test_p42_kb78_full_vault_backfill_fe_linter_and_option_turnover_rescue.py
  ```
- **User Launchers (Root)**:
  - `重開機的時候點我.bat`: Starts Web UI (Port 8999) + `headless_brain_miner.py` silently.
  - `停止背景程式.bat`: Writes `data_persistence/watchdog.paused` and cleanly stops `Quant_BrainMiner` & `Quant_WebUI`.
  - `查看加密貨幣盈虧.bat`: Displays live `data_persistence/crypto_pnl_status.json`.
- **Security Rules**:
  - NEVER commit `.env`, `*.db*`, `*.pkl`, `*.log`, or runtime JSON vaults.
  - Always use explicit `git add <file1> <file2>` (never `git add -A` or `git add .`).

---

## 6. Common Troubleshooting

| Symptom | Root Cause | Resolution |
| :--- | :--- | :--- |
| `sqlite3.OperationalError: database is locked` / `disk I/O error` | Opening DB directly in Google Drive or unclosed `open_sqlite_wal()` handle | Route DB via `state_paths.py` (`C:\quant_envs\state\`) and use `managed_connection()` |
| Legacy BRAIN unit tests stage 0 / submit 0 | P37 `CONSULTANT_RESERVE` (`Sharpe >= 1.50` / `Fitness >= 1.20`) & `worst_first` active by default | In legacy tests, set `BRAIN_RESERVE_MIN_SHARPE=99`, `BRAIN_RESERVE_MIN_FITNESS=99`, `BRAIN_SUBMIT_ORDER=best_first` |
| `PermissionError: [WinError 32]` on Windows `TemporaryDirectory` | SQLite WAL file handle not closed before `rmtree` | Use `managed_connection()` + `TemporaryDirectory(ignore_cleanup_errors=True)` |
| `Quant_BrainMiner` restarts immediately after kill | `Quant_Watchdog` ScheduledTask is active | Create `data_persistence/watchdog.paused` (or run `停止背景程式.bat`) before stopping |
