# 專案全面健檢、檔案整理合併與冗餘清理報告 (Phase 6)

## 一、專案核心檔案盤點與受保護架構

依據系統開發原則與規範，以下核心骨幹檔案均經過嚴格檢驗並完整保留：
1. `Dockerfile`: Docker 輕量化容器封裝檔（預設暴露端口 8999，支援雲端 24/7 背景無人值守運轉）。
2. `render.yaml`: Render.com 自動化雲端部署規範（已配置新加坡節點、USDC 零手續費與輕量化推論環境變數）。
3. `kaggle_notebooks/`: 雲端 GPU/CPU 訓練筆記本生成與執行庫（含 `generate_notebooks.py`、`README.md` 等完整生成邏輯）。
4. `GEMINI.md`: 系統核心規範與量化工程開發規範指南。
5. `docs/research/ADVANCED_QUANT_RESEARCH.md`: 前沿學術論文庫、HMM 狀態辨識、貝葉斯條件機率與可微損失函數文獻庫。

---

## 二、檔案合併與清理紀錄表

| 項目 / 檔案路徑 | 操作類型 | 異動原因與處置說明 |
| :--- | :--- | :--- |
| `web_ui/server.py` 開頭重複 import | 合併優化 | 刪除第 1~28 行重複之模組導入代碼，整合至統一的標準模組載入區塊，消除潛在 namespace 混亂。 |
| `test_write.txt` | 臨時清理 | 診斷期間建立之空暫存檔，已清空標記。 |
| `web_ui/server.py` 之 `/api/taiwan/market` 與 `/api/taiwan/report` | 邏輯合併 | 原先上方儀表板與下方報告分別呼叫導致分數與均線型態打架，重構為共用 `QuantAppContext.get_taiwan_market_analysis()` 與 60 秒記憶體快取。 |
| `taiwan_stock/stock_screener.py` | 函式擴充合併 | 將篩選門檻與個股詳細評估合併至 `evaluate_stock()` 與 `run_screener_detailed()`，並保留舊有 `run_screener()` 向後相容介面。 |
| `taiwan_stock/scheduler.py` | 排程合併重構 | 原先僅支援單一時段 08:00，升級合併為 Asia/Taipei 雙時段排程：16:30 (盤後) 與 08:00 (盤前)。 |
| `web_ui/static/index.html` (Lightweight Charts) | 視圖整合 | 引入 TradingView Lightweight Charts 4.1.1，將即時 K 線、MA、布林帶、買賣標記、持倉成本線與選股池點擊連動完整融入 SPA 操盤室中。 |
| `docs/BATCH[1-5]_*.md` | 冗餘清理 | 刪除 Batch 1 至 Batch 5 過程性規格/提示暫存檔，成果統整歸納入 `docs/BATCH_ACCEPTANCE_MATRIX.md`。 |
| `docs/*.gdoc` | 冗餘清理 | 刪除 Google Drive 同步殘留之 .gdoc 偽檔案（`TRAIN_CLI_THREE_MODELS_SPEC.md.gdoc`、`KAGGLE_ONE_SHOT_NOTEBOOK_SPEC.md.gdoc` 等 13 個暫存檔）。 |
| `crypto_trading/feature_engineer.py` (.bfill 修復) | 洩漏修復 | 徹底移除跨幣種計算中的 `.bfill()`，改為純前向填充 `.ffill()`，杜絕未來收盤價反向洩漏。 |
| `crypto_trading/feature_engineer.py` (target_direction 修復) | 邏輯修復 | 修復尾端未來收益為 NaN 時 `target_direction` 誤轉為 0 的前瞻偏差問題，確保尾端保持為 NaN/None。 |
| `crypto_trading/ml_trainer.py` (KeyError 與 Sharpe 修正) | Bug 修補 | 分類目標欄位動態綁定 `target_direction_3`，徹底消除 `target_binary` KeyError；年化因子依 3 期重疊收益除以 3，消除夏普比率虛胖。 |
| `crypto_trading/bitunix_client.py` (reduceOnly) | 交易安全 | 下單函式與一鍵平倉補齊 `reduce_only=True`，防止平倉單因保證金不足或行情跳空轉為反向反手開倉。 |
| `crypto_trading/testnet_client.py` (異常假成功) | 交易安全 | 修復幣安測試網遠端拋出異常時誤回傳 `success=True` 的問題，改為精準回傳失敗狀態與原因。 |
| `worldquant_brain/alpha_vault.py` (換手率誤除 100) | 標準審核 | 嚴格鎖定 `0.01 <= turnover <= 0.70`，小數換手率 > 0.70 一律判定超標，徹底修補先前 `turnover > 1.0` 盲目除以 100 造成假通過的漏洞。 |
| `crypto_trading/rl_env.py` (成本價重置) | 交易邏輯 | 重置環境時持倉成本價正確初始化為 0.0，且在全平倉動作下精確歸零成本價。 |
| `taiwan_stock/stock_screener.py` (零分母防禦) | 數值穩定 | KD 指標分母改採 `np.maximum(..., 1e-4)`、成交量比率改採 `max(..., 1.0)`，徹底修補 1e-9 浮點微小值導致的分數計算暴衝。 |
| `web_ui/server.py` & `data_persistence/gdrive_sync.py` (bot_state 欄位衝突) | 資料庫修補 | 統一 `bot_state` 資料表欄位結構，補齊 `updated_at` 欄位並設置預設時間戳，避免 Schema 不一致與約束報錯。 |
| `data_persistence/gdrive_sync.py` (sqlite3.backup()) | 原子備份 | 本地 Rolling Backups 優先採用 `sqlite3.backup()` 在線鎖定備份，保證運行中資料庫的一致性。 |
| `crypto_trading/bot_runner.py` | 架構升級 | 下單撮合引擎全生命週期串接 SQLite `orders`、`positions`、`trade_logs`、`account_equity`，並於虧損時自動觸發 `post_mortems`。 |
| `web_ui/server.py` (買賣標記動態化) | 邏輯修復 | 徹底移除寫死假箭頭，改為動態自 SQLite orders 歷史提取成交時間點並注入圖表。 |

---

## 三、未來維護與擴充建議

1. **快取生命週期**：大盤分析每 60 秒刷新一次，若需強制刷新可呼叫 `force_refresh=True` 或透過前端「立即手動生成最新晨報」POST 按鈕。
2. **圖表資源釋放**：切換至其他 Tab（台股、BRAIN、雲端）時，前端圖表輪詢自動中斷（`stopChartPolling`），保護瀏覽器記憶體並完全隔離後端 24/7 常駐背景機器人線程。
