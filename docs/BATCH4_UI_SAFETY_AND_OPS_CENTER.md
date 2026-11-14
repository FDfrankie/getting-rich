# BATCH 4: 前端交易控制台安全重構、點位標記與維運中心 (UI Safety & Ops Center)

## 一、核心目標與任務清單
1. **多線程 HTTP 服務器升級 (ThreadingHTTPServer)**：
   - 升級 web_ui/server.py，採用 ThreadingHTTPServer 替代單線程伺服器，解決高頻輪詢或長耗時請求造成的介面卡死問題。
2. **TradingView 擬真圖表與買賣點位真實標記**：
   - 徹底移除寫死於前端的假 BUY 箭頭與靜態假資料。
   - 將系統實際成交之 Order 歷史動態注入 K 線圖表，精確呈現開多 (BUY)、開空 (SELL)、平倉 (CLOSE) 之標記與損益曲線。
3. **台股智慧選股多空評分系統標準化 (0-100 Score)**：
   - 台股評分引擎統一輸出 0-100 區間標準化評分，清楚區分「強力做多 (>75)」、「中性盤整 (40-60)」、「空方警示 (<25)」。
   - 整合多因子 LightGBM Ranker (	aiwan_stock_ranker.joblib) 與開盤前/收盤後雙時段排程更新。
4. **健康檢查與維運端點 (/health & Dashboard)**：
   - 建立 /health API，提供交易機器人存活狀態、背景線程計數、幣安連線狀態與最新心跳時間戳。
