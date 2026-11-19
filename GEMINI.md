# GEMINI.md - 量化操盤與工程開發規範指南 (Antigravity Quant Engine Specifications)

本規範檔案定義了本專案的核心架構原則、交易安全準則、WorldQuant BRAIN / IQC 2026 研究管線、跨市場數據模型以及持久化協定。

---

## 一、系統核心架構與常駐執行原則

1. **24/7 不間斷背景操盤**：
   - 核心交易引擎（`QuantAppContext` / `MultiSymbolBotManager` / `CryptoTradingBot`）常駐於後台守護線程（Background Daemon Thread）運行。
   - 不得因使用者切換前端分頁或重新整理瀏覽器而終止交易迴圈。
2. **SQLite 持久化與雲端同步隔離 (`data_persistence/state_paths.py`, `sqlite_resilience.py`)**：
   - 在 Windows 雲端同步目錄（如 `G:\我的雲端硬碟\炒幣`）下，`system_state.db` 與 `alpha_vault.db` 自動導向本機 `C:\quant_envs\state\`，並以 `sqlite3.backup()` 定期快照回 `data_persistence/`，避免 Google Drive 檔案鎖造成 `database is locked` 或 `disk I/O error`。
   - SQL Schema Migration 位於 `migrations/006` ~ `009`，由 `EvidenceRepository.migrate()`、`IQCScoringRepository.migrate()` 與 `CorrelationEvidenceRepository.migrate()` 冪等執行（本專案不使用 `data_persistence.migrations` Python 模組）。
3. **無縫雲端部署架構 (OnRender)**：
   - 本地沙盒模擬與雲端服務完全解耦；推送至 `origin main` 後由 Render (`getting-rich.onrender.com`) 自動建置與部署。

---

## 二、加密貨幣多幣種操盤準則 (`crypto_trading/`, `portfolio/`)

1. **多幣種規格與風險配置**：
   - `crypto_trading/instruments.py` 與 `multi_symbol_runner.py` 支援 `BTC/USDT`、`ETH/USDT`、`SOL/USDT`、`BNB/USDT` 的 `step_size` / `tick_size` 正規化與多幣種並行。
   - `portfolio/risk_allocator.py` 負責組合風險上限與槓桿配置；`data_loader.py` 支援 `crypto_trading/data_cache/*_1h_180d.csv` 離線快取降級。
2. **嚴格 OOS 驗證隔離**：
   - 未通過 Walk-Forward / Cost Stress 門檻的模型（如 LightGBM / PPO v2）維持 `SHADOW` 隔離狀態，不得直接接管實盤資金。

---

## 三、WorldQuant BRAIN 與 IQC 2026 Alpha 工廠架構 (`worldquant_brain/`)

1. **核心管線**：
   ```text
   Field Survey → Alpha Blueprint → Candidate Generation (MCTS + Frequent-Subtree Avoidance)
     → Novelty / Duplicate / Local CS Pre-screen (local_cs_screener.py)
     → Objective-aware Lanes (RETURN / SHARPE / FITNESS / CAPACITY / BALANCED)
     → ExpectedYieldPerMinute Ranking → Official BRAIN Simulation & Result Parser
     → Priority Matrix (P0–P6) + PnL Jump Diagnosis + Neutralization Ablation
     → IQC 3-Layer Scoring (Hard Gates -> 100-pt Candidate Score -> 20-pt Timing -> Daily Planner)
     → Evidence & DSR Trial Registry → PreSubmissionGate
   ```
2. **官方提交硬門檻 (`pre_submission_gate.py`, `iqc_scoring.py`, `pnl_jump_scoring.py`)**：
   - **Provenance**：必須為 `WORLDQUANT_BRAIN` 官方模擬結果，具備 `official_simulation_id` 與 `brain_alpha_id`（`LOCAL_PROXY` 永遠 `BLOCKED`）。
   - **Delay-aware Sharpe**：`Delay 1` 嚴格 `> 1.25`；`Delay 0` 嚴格 `> 2.00`。
   - **Fitness**：嚴格 `> 1.00`，公式分母使用 `max(Turnover, 0.125)`。
   - **Turnover 分層**：`0.01 < Turnover < 0.70`（`12.5%–30%` 為成本效率優先區、`30%–40%` 需 Cost Stress、`40%–70%` 需完整成本與 Margin 證據）。
   - **Self-Correlation & Weight**：僅限非過期官方 `SELF` PnL 相關性嚴格 `< 0.70`；最大單股權重 `< 0.10`；通過 `Bias`、`Weight`、`Sub-universe` 官方檢查。
3. **PnL 相關性與中性化原則 (`correlation_evidence.py`, `neutralization.py`)**：
   - **Correlation 是 PnL Graph 相關性**：相同 Expression 的 `D0` 與 `D1` 不自動視為高相關，依 `evaluate_delay_pair()` 比對 PnL 相關性；公式相似度僅用於研究去重與 `Research Novelty`。
   - **Neutralization 雙層分離**：區分 Expression 內層 `group_neutralize()` 與 Simulation Setting 層；以同 Baseline 多目標消融（Sharpe、Return 保留 `>= 70%`、Drawdown、Market Beta `<= 0.30`、OS 穩定性）擇優。
   - **Signal 純度**：單條 Alpha 維持 1 個主假說、最多 3 個條件（`if_else` / `trade_when`），多訊號組合交由 SuperAlpha 或組合層處理。

---

## 四、只讀診斷與評分 API (`web_ui/server.py`)

- `GET /api/brain/alphas/{id}/improvement-plan`：Alpha 改進優先級矩陣（P0–P6）
- `GET /api/brain/iqc/candidates`、`GET /api/brain/iqc/candidates/{id}`、`GET /api/brain/iqc/daily-plan`、`POST /api/brain/iqc/recalculate`：IQC 2026 候選評分、每日 1～3 條計畫與冪等重算
- `GET /api/brain/alphas/{id}/pnl-diagnosis`、`GET /api/brain/alphas/{id}/improvement-checklist`：PnL 跳變診斷與 10 大檢查清單
- `GET /api/brain/alphas/{id}/correlations`、`GET /api/brain/alphas/{id}/top-correlated`、`GET /api/brain/alphas/{id}/delay-pair`：PnL 相關性證據與 D0/D1 配對決策
- `GET /api/brain/alphas/{id}/neutralization-ablation`：中性化層級與有效最終中性化診斷

---

## 五、開發與測試協定

1. 離線測試（不觸發真實下單或真實 BRAIN Submit）：
   `python -m unittest tests.test_novelty_and_survey tests.test_antigravity_enhancements tests.test_multicoin_and_alpha_resilience tests.test_all_5_improvements tests.test_submit_corr_rejection_cleanup tests.test_local_cs_screener tests.test_alpha_yield_booster tests.test_objective_aware_integration tests.test_alpha_improvement_priority_and_dsr tests.test_iqc_scoring tests.test_iqc_scoring_boundaries tests.test_iqc_scoring_properties tests.test_iqc_api_contract tests.test_pnl_jump_scoring tests.test_correlation_semantics tests.test_neutralization -v`
2. 嚴禁將 `.env`、`*.db*`、`*.pkl` 納入版本控制。
