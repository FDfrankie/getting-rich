# 🧠 WorldQuant BRAIN 常用數據庫與欄位指南 (Data Fields Guide)

在 WorldQuant BRAIN 平台中，數據按照類型分為不同集合（Datasets）。了解各數據欄位的經濟學含義是開發高相關性、具備真實定價邏輯 Alpha 的關鍵。

---

## 1. 量價數據 (Price / Volume - Model Data)
最基礎且流動性最強的數據集，所有等級皆可使用：

- `open`: 開盤價 (已復權)
- `close`: 收盤價 (已復權)
- `high`: 最高價
- `low`: 最低價
- `volume`: 成交股數
- `vwap`: 成交量加權平均價 (Volume Weighted Average Price)
- `returns`: 當日收益率 `(close / ts_delay(close, 1)) - 1`
- `cap`: 公司總市值 (Market Capitalization)
- `sharesout`: 流通股本 (Shares Outstanding)

---

## 2. 產業與分組向量 (Grouping Vectors)
用於中性化與產業內部排序：

- `sector`: 一級板塊分類 (約 11 大類)
- `industry`: 二級產業分類 (約 60 類)
- `subindustry`: 三級細分產業分類 (約 150+ 類，最推薦之中性化基準)
- `country`: 國家/地區代碼
- `market`: 市場代碼

---

## 3. 基本面財務數據 (Fundamental Data)
解鎖 Expert 或 Master 後常用，具備強大的長線定價能力：

- `assets`: 總資產
- `liabilities`: 總負債
- `debt`: 總債務
- `equity`: 股東權益 / 淨資產
- `sales`: 營業收入 (Revenue)
- `cogs`: 銷貨成本 (Cost of Goods Sold)
- `operating_income`: 營業利益
- `net_income`: 稅後淨利
- `operating_cashflow`: 營業活動現金流量
- `capex`: 資本支出 (Capital Expenditure)
- `ebitda`: 稅息折舊及攤銷前利潤
- `eps`: 每股盈餘 (Earnings Per Share)

---

## 4. 分析師預期數據 (Analyst Consensus Data)
機構專業定價的先行指標：

- `eps_consensus_mean`: 分析師對未來 EPS 預期均值
- `target_price`: 分析師平均目標價
- `target_price_high`: 分析師最高目標價
- `target_price_low`: 分析師最低目標價
- `analyst_rating_mean`: 分析師綜合評級 (1: Strong Buy ~ 5: Sell)
- `num_analysts`: 涵蓋該標的之分析師數量
