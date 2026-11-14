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

---

## 三、未來維護與擴充建議

1. **快取生命週期**：大盤分析每 60 秒刷新一次，若需強制刷新可呼叫 `force_refresh=True` 或透過前端「立即手動生成最新晨報」POST 按鈕。
2. **圖表資源釋放**：切換至其他 Tab（台股、BRAIN、雲端）時，前端圖表輪詢自動中斷（`stopChartPolling`），保護瀏覽器記憶體並完全隔離後端 24/7 常駐背景機器人線程。
