# BATCH 2: 委託訂單持久化、幣安真實下單與雲端同步 (Orders SQLite & Cloud Sync)

## 一、核心目標與任務清單
1. **幣安合約測試網真實下單 (Binance Futures Testnet Real Execution)**：
   - 修正 CryptoTradingBot 交易迴圈，不再僅記錄記憶體變數，而是真實透過 CCXT 送出委託單至幣安測試網 (https://testnet.binancefuture.com)。
   - 支援市價單 (Market Order) 與限價單 (Limit Order)，並精確管理槓桿倍數 (Leverage) 與保證金模式 (ISOLATED / CROSSED)。
2. **真實 Order ID 追蹤與持倉同步 (Order Lifecycle & Position Tracking)**：
   - 記錄並追蹤每次下單之真實 exchange_order_id、成交價、成交手數與手續費。
   - 定期調用幣安帳戶端點，精確回傳真實 Mark Price、未平倉損益 (Unrealized PnL) 與強平價格。
3. **SQLite 訂單歷史與持倉持久化 (data_persistence/system_state.db)**：
   - 重構資料庫 schema：建立 orders, positions, 	rade_logs, ccount_equity 等資料表。
   - 伺服器冷重啟或斷線重連時，自動還原當前倉位與未完成委託，避免狀態遺失。
4. **Google Drive API 雙向備份同步 (GDrive Cloud Backup)**：
   - 實現 data_persistence/gdrive_sync.py，支援 SQLite 資料庫與 JSON 快照定期自動上傳至 Google Drive 雲端專屬目錄。
