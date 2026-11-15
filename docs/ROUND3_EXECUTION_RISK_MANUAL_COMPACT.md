/goal /browser /plan /teamwork /learn /boost /migrate-workflows /swarm

# ROUND3_EXECUTION_RISK_MANUAL_COMPACT.md

> **Execution Scope**: P5（t+1 回測與成本）+ P6（訂單冪等、對帳、客戶端修復、**合約人工手動控盤**）+ P7（集中式 RiskManager）+ P8（SQLite 持久化與 Bot 接線）。
> **嚴禁**：啟動 `BITUNIX_LIVE`、發送任何真實／Testnet 訂單、在 `pytest` 中連外。

## 0. Token 紀律
1. 不要重述規格、不要建新規格檔、不要先輸出長篇計畫，直接改程式與測試。
2. 每完成一個 Phase 即跑 `python -m compileall -q .` 與 `pytest -m "not live" -q` 並獨立 commit，不得把 `NOT RUN` 寫成 `PASS`。

## 1. P5：Backtest `t+1` Execution 與真實交易成本
- `executed_position = target_signal.shift(1)`；成本分開計算 maker、taker、spread、slippage、funding、delay、tick、step、min notional；建立有來源與生效時間的 `FeeSchedule`。
- 輸出 gross/net return、Sharpe、Sortino、Calmar、drawdown、turnover、各項成本、trade count、exposure、hit rate、profit factor、signal delay、data source。
- 新增 `tests/test_backtest_execution_delay.py`。

## 2. P6：訂單冪等、對帳、客戶端修復與合約手動控盤
1. `OrderState`（`CREATED, SUBMITTING, ACKNOWLEDGED, PARTIALLY_FILLED, FILLED, CANCEL_PENDING, CANCELLED, REJECTED, UNKNOWN`）與 SHA-256 `client_order_id`／`idempotency_key`；DB claim → `SUBMITTING` → exchange → `ACK/FILLED`；timeout → `UNKNOWN` 先查遠端，嚴禁盲目重送；DB unique constraints。
2. `testnet_client.py`、`bitunix_client.py`：移除例外時假回傳 `success: True`；平倉補 `reduceOnly=True`，修正空單/雙向持倉損益計算與 `closed_count`。
3. Reconciliation：比對 orders、fills、positions、balances、partial fills、pending cancels；Restart 後先 reconcile，完成前只允許 `reduce-only`。
4. **合約人工手動控盤（`manual_override`）**：支援單筆全平、部分減倉、設定/修改 SL/TP 保護單、鎖定 `manual_override`（含 TTL 到期機制）；Bot cycle 不得覆蓋被人工接管的倉位；所有手動操作走 `RiskManager` 與結構化事件。
5. 新增（含 fault injection）：`tests/test_order_idempotency.py`、`tests/test_reconciliation_lifecycle.py`、`tests/test_manual_position_control.py`。

## 3. P7：集中式 `RiskManager`
- 所有下單入口共用單一 Gate；`RiskDecision` 檢查 TradingMode、unlock、DataSource、DataQuality、stale data、reconciliation freshness、unknown orders、DB writable、exchange connectivity、notional、exposure、leverage、positions、daily loss、drawdown、min notional、step、tick。Fail-closed，不確定時只允許 `reduce-only`。
- 新增 `tests/test_risk_gate.py`。

## 4. P8：SQLite Persistence、誠實命名與 Bot 接線
- SQLite 啟用 `WAL`、`synchronous=FULL`、`foreign_keys=ON`、`busy_timeout`，建立完整資料表與可重跑 migration（升級前 `sqlite3.backup()` 並測 restore）。
- `GDrivePersistenceSync` 若無真實 OAuth／Service Account 雙向同步，改名 `LocalPersistenceService`。
- 接線 `bot_runner.py` 與 `web_ui/server.py`：移除 `iterations=3` 限制，在 `BINANCE_TESTNET` 模式下走正式 `RiskManager → OrderIdempotency → testnet_client`。
- 新增：`tests/test_database_migration.py`、`tests/test_backup_restore.py`、`tests/test_gdrive_provider.py`。
最終狀態只能是 `PHASE_COMPLETE`、`PARTIAL` 或 `BLOCKED`。
