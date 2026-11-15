/goal /browser /plan /teamwork /learn /boost /migrate-workflows /swarm

# Round 1 極簡執行規格（僅限 P0–P3，完成即停）

## 執行紀律

- 不重述需求、不建新規格檔、不輸出長計畫、先改程式。
- 每次只回報：修改的檔案、測試結果、阻塞問題。
- 不得依賴舊行號，一律以最新 Git 檔案之類別與函式名稱為準。
- **本輪只做到 P3 完成即停**（狀態只能是 `PHASE_COMPLETE` / `PARTIAL` / `BLOCKED`，嚴禁輸出 `GOAL_COMPLETE`）。
- 嚴禁啟動 `BITUNIX_LIVE`、送出真實委託、連外測試、顯示 Secret 或 force push；每 Phase 結束皆跑 `python -m compileall -q .` 與 `pytest -m "not live" -q` 並建立獨立 commit。

## P0：Secret 清理與安全封裝 (`security: remove secrets and add safe packaging`)

1. `git rm --cached` 移除已追蹤的 `.env`、`__pycache__`、`*.db`、`cloud_state_snapshot.json`（不刪本機資料，禁止 `reset --hard`）；補齊 `.gitignore`（含 `*.db*`、`*.sqlite*`、`*.joblib`、`*.pt`、`data_persistence/backups/`、`.env`、`!.env.example`）。
2. 建立只含空值的 `.env.example`（含 `TRADING_MODE=LOCAL_PAPER`、`BRAIN_SIMULATION_ENABLED=false`、`BRAIN_AUTO_SUBMIT_ENABLED=false` 等）；加入 `.gitleaks.toml`。
3. 建立 `scripts/package_for_review.py`（allowlist 打包 + `MANIFEST.json` SHA-256 + 掃描拒絕 `.env`/`.git`/`*.db`/Secret）與 `tests/test_package_allowlist.py`。

## P1：集中式 Settings 與 TradingMode (`config: introduce validated trading modes`)

1. 建立 `TradingMode`（`LOCAL_PAPER` / `BINANCE_TESTNET` / `BITUNIX_LIVE`），預設 `LOCAL_PAPER`；移除所有 Key/Secret 非空預設值。
2. `BITUNIX_LIVE` 需同時滿足 `ENABLE_LIVE_TRADING=I_UNDERSTAND_THE_RISK`、Key/Secret 非空、`MAX_ORDER_NOTIONAL_USDT > 0`、`MAX_TOTAL_EXPOSURE_USDT > 0`、`MAX_DAILY_LOSS_USDT > 0`、`MAX_LEVERAGE > 0`、`LIVE_OPERATOR_ACK_VERSION` 正確；`BitunixClient` 由 Settings 注入，不得再用 `bool(api_key)` 判定 `is_live`。`PAPER_TRADING` 僅留相容層並發 `DeprecationWarning`。
3. 移除 `testnet_client.py` `__main__` 自動 `create_order`（需 `ALLOW_MANUAL_TESTNET_ORDER=YES` + CLI 明確參數，預設 `quantity=0`）。新增 `tests/test_trading_mode_safety.py`、`tests/test_testnet_cli_guard.py`。

## P2：DataSource、OHLCVResult 與資料品質閘門 (`data: enforce causal OHLCV policy`)

1. 移除 `CryptoDataLoader.fetch_ohlcv()` 的 `except Exception: pass` 與靜默合成 fallback；建立 `DataSource`（`LIVE`/`CACHE`/`DELAYED`/`SYNTHETIC`）與 `OHLCVResult`，交易與真實行情預設 `allow_synthetic=False`，失敗拋 `DataUnavailableError`；timestamp 一律 UTC-aware，排除未收盤 candle；synthetic seed 改用 `hashlib.sha256`。
2. 新增 `crypto_trading/data_quality.py`（`validate_ohlcv()` 驗證遞增、無重複、OHLC 邏輯、無未收盤 candle、延遲 `<= 2` 根），失敗時觸發 `DATA_QUALITY_ERROR` 禁止開新倉。新增 `tests/test_data_source_policy.py`、`tests/test_data_quality.py`、`tests/test_closed_candle_policy.py`。

## P3：跨資產因果對齊、Instrument 與 15s 狀態 (`features: add causal cross-asset alignment`)

1. `feature_engineer.py`：移除 `compute_multi_asset_cross_features()` 的 `.bfill()`（改 `.ffill(limit=max_gap)`、stale 排除與最低覆蓋率）；修正 `target_direction` 尾端 NaN（保留 `pd.NA` 不錯標為 0）；`obv_slope_10` 改用 10 期滾動淨主動量比。
2. 建立 `Instrument` / `InstrumentRegistry` 取代 `"BTC" in close_df` 字串猜測。
3. 15s 未具備真實 aggTrade 聚合前標為 `EXPERIMENTAL_RESEARCH_ONLY` 並禁止實盤下單。新增 `tests/test_multi_asset_no_lookahead.py`、`tests/test_instrument_registry.py`、`tests/test_multi_timeframe_causality.py`。
