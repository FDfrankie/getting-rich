/goal /browser /plan /teamwork /learn /boost /migrate-workflows /swarm

# ROUND2_TRAIN_READY_COMPACT.md

> **Execution Scope**: 完成 3 支模型開始真實訓練前的最後技術門檻：**P4（Crypto Purged Walk-Forward）+ 台股無洩漏時間切分 + RL 最小修正包 + `scripts/train_models.py` 獨立訓練入口與 SHA-256 Manifest**。
> **嚴禁**：修改策略下單規則、啟動 `BITUNIX_LIVE`、發送任何真實／Testnet 訂單、在 `pytest` 中連外抓資料。

## 0. Token 紀律
1. 不要重述本規格、不要建新規格檔、不要先輸出長篇計畫，直接改程式與測試。
2. 保留既有的 `P0–P3` 與 Cockpit 可觀測性修改。
3. 每完成一個子項即跑 `python -m compileall -q .` 與 `pytest -m "not live" -q`，不得把 `NOT RUN` 寫成 `PASS`。

## 1. Crypto LightGBM 無洩漏與 P4 驗證（`crypto_trading/feature_engineer.py`、`ml_trainer.py`）
1. `feature_engineer.py`：`target_direction_{horizon}` 尾端因 `shift(-horizon)` 產生的 `NaN` 必須保留為 `pd.NA`，禁止 `(> 0).astype("Int64")` 把未知未來轉成 `0`；確認無 `.bfill()` 且僅使用已收盤 candle。
2. `ml_trainer.py`：修復 `KeyError: 'target_binary'`；移除 `predict()` 的 `fillna(0)`；實作 `purged_walk_forward_splits(n_samples, n_splits, purge_bars=12, embargo_bars=0)` 保證 `train_idx.max() + purge_bars + embargo_bars < validation_idx.min()`；所有 imputer/scaler/selector/early stopping 只能在 fold Train 內 fit；保留 Final Untouched Test；輸出 Rank IC、net Sharpe（non-overlapping 或 Newey-West/HAC + 明確手續費假設）與 turnover。

## 2. Taiwan Ranker 無洩漏切分與 Manifest（`taiwan_stock/ml_ranker.py`）
1. 移除 `concat` 後直接 `80/20` index 切分，改為按交易日 `date` 分組的 Purged Walk-Forward（`purge >= 5` 交易日），並保留 Final Untouched Test。
2. 修正 `Volume` 分母 `+ 1e-9` 的失真；區分原始 5 日報酬與真正的橫截面超額報酬（減去同日市場／股票池均值）。
3. 計算並輸出橫截面 Rank IC、ICIR 與多空換手率，產出 `taiwan_stock_ranker.joblib` 與對應 SHA-256 `manifest.json`。

## 3. PPO RL 最小修正包（`crypto_trading/rl_env.py`、`rl_agent.py`）
1. `rl_env.py`：修正局部調倉覆寫 `entry_price = next_price` 的 bug，改為加權平均持倉成本與逐 bar mark-to-market 淨損益；step reward 扣除真實手續費與滑點；更正獎勵函數命名（不誤標為 `Differential Sharpe`）；嚴格切分 Train / Validation / Final Untouched Test 時間區間。
2. `rl_agent.py`：改為 tanh-squashed Gaussian（含 Jacobian log-det 修正），並修正 `num_updates` 計算錯誤。

## 4. 統一訓練 CLI 與 3 個獨立模型 Manifest（`scripts/train_models.py`）
支援 `--target crypto_lgbm`、`--target crypto_ppo`、`--target taiwan_ranker`，抓取真實歷史資料（`allow_synthetic=False`）、執行無洩漏訓練、印出 OOS 績效並原子寫入：
- `models/crypto_lgbm_weights.joblib` + `models/crypto_lgbm_manifest.json`
- `models/crypto_ppo_agent.pt` + `models/crypto_ppo_manifest.json`
- `models/taiwan_stock_ranker.joblib` + `models/taiwan_stock_manifest.json`

## 5. 必建 Mock 測試（一律 mock，不連外）
`tests/test_purged_walk_forward.py`、`tests/test_taiwan_ranker_no_leakage.py`、`tests/test_rl_minimal_fixes.py`、`tests/test_train_models_cli.py`。
最終狀態只能是 `PHASE_COMPLETE`、`PARTIAL` 或 `BLOCKED`（禁止輸出 `GOAL_COMPLETE`）。
