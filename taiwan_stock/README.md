# 🇹🇼 台股大盤診斷、機器學習多因子排序與每日 08:00 晨報系統

本模組專注於台灣股市（TWSE / TPEx）之量化多空環境掌握、多因子機器學習預測（LightGBM / XGBoost）與突破型動量標的篩選，並在每日開盤前（08:00 AM）自動完成市場計算與決策報告生成。

---

## 📌 核心功能

1. **大盤環境分析器 (`market_analyzer.py`)**：
   - 監控加權指數（`^TWII`）長中短期均線體系（MA5 / MA20 月線 / MA60 季線）。
   - 評定市場狀態：`強烈看多`、`謹慎偏多`、`區間震盪`、`空頭防禦`。
   - 計算 20 日年化波動率與多空綜合量化評分（0~100 分）。

2. **機器學習多因子選股排序器 (`ml_ranker.py`)**：
   - **因子體系**：提取動量因子（5d, 10d, 20d, 60d）、均線乖離率（MA5, 20, 60）、20 日年化波動率、ATR 佔比、爆量比、KD/RSI 震盪擺動因子。
   - **預測標的**：預測個股未來 5 日超額報酬（Forward 5-Day Return）。
   - **排序機制**：透過 LightGBM/XGBoost 模型進行全池截面打分排序，選出預測回報率最高的領頭潛力標的。
   - 支援特徵重要性統計與模型持久化導出 (`.joblib`)。

3. **技術面智慧選股器 (`stock_screener.py`)**：
   - **均線排列**：過濾出短天期均線穿越並發散之標的。
   - **爆量突破**：成交量大於 5 日均量 1.5 倍以上。
   - **布林通道**：帶量突破布林上軌之起漲動態。
   - **KD 黃金交叉**：9,3,3 參數下低/中檔黃金交叉或高檔強勢鈍化。
   - **動態 ATR 防守點**：自動依真實波動區間（ATR）計算建議停損點與 1:2 風險報酬比之目標價。

4. **自動晨報生成器 (`report_generator.py`)**：
   - 彙整當日大盤結構、技術突破標的以及 **AI / ML 多因子未來 5 日超額報酬排序榜**，產出結構清晰之 Markdown 晨報並存檔於 `reports/`。

5. **定時守護排程器 (`scheduler.py`)**：
   - 使用輕量級 `schedule` 框架，在每日 08:00 自動觸發選股與晨報生成，支援後續串接 LINE Notify 或 Discord Webhook。

---

## 🚀 執行指南

### 1. 立即手動生成一份今日晨報 (包含大盤診斷、技術突破與 ML 排行)
```bash
python taiwan_stock/report_generator.py
```
報告將即時印出於終端機，並存檔至 `taiwan_stock/reports/morning_report_YYYY-MM-DD.md`。

### 2. 本機執行台股 ML 模型訓練與排序
```python
import yfinance as yf
from taiwan_stock.ml_ranker import TaiwanStockMLRanker
from taiwan_stock.config import DEFAULT_WATCHLIST

stocks = {sym: yf.Ticker(sym).history(period="1y") for sym in DEFAULT_WATCHLIST[:6]}
ranker = TaiwanStockMLRanker(model_type="lightgbm")
clean_data, _ = ranker.prepare_dataset(stocks)
metrics = ranker.train_model(clean_data)
print("Rank IC:", metrics["rank_ic"])

rank_df = ranker.predict_rankings(stocks)
print(rank_df.head())
```

### 3. 在 Kaggle 上執行台股多因子排序筆記本
- 檔案路徑：[`kaggle_notebooks/kaggle_taiwan_stock_ml_ranker.ipynb`](../kaggle_notebooks/kaggle_taiwan_stock_ml_ranker.ipynb)
- 具備完整 yfinance 下載、LightGBM 訓練、Walk-Forward 交叉檢驗與未來 5 日選股排序輸出。
