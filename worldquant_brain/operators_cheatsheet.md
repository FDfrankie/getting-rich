# 🧠 WorldQuant BRAIN 核心運算子速查手冊 (Operators Cheatsheet)

WorldQuant BRAIN 使用專屬的向量化表達式語法（Expressions）。掌握常用運算子是構造高 Sharpe、低 Turnover Alpha 的關鍵基石。

---

## 1. 橫截面運算子 (Cross-Sectional Operators)
作用於同一個時間截面上的所有股票：

| 運算子 | 範例 | 數理語意 |
| :--- | :--- | :--- |
| `rank(x)` | `rank(close)` | 將橫截面上數值排序並均勻映射至 [0, 1] 區間 |
| `scale(x, a)` | `scale(alpha, 1)` | 重新縮放數值，使得所有股票的絕對值總和等於 `a` (預設為 1) |
| `group_rank(x, group)` | `group_rank(pe, industry)` | 在同一個產業/板塊內部計算排序 (產業內相對強弱) |
| `group_neutralize(x, group)` | `group_neutralize(alpha, subindustry)` | 減去該產業平均值，消除行業風險敞口 (極度重要！) |
| `quantile(x)` | `quantile(volume)` | 橫截面分位數計算 |

---

## 2. 時間序列運算子 (Time Series Operators)
作用於單一檔股票歷史 K 線的時間維度：

| 運算子 | 範例 | 數理語意 |
| :--- | :--- | :--- |
| `ts_rank(x, d)` | `ts_rank(close, 20)` | 計算 `x` 在過去 `d` 天內的時間序列分位數 (0~1) |
| `ts_mean(x, d)` | `ts_mean(volume, 10)` | 過去 `d` 天的簡單算術平均值 (SMA) |
| `ts_decay_linear(x, d)`| `ts_decay_linear(returns, 5)` | 線性衰減加權移動平均 (越靠近今天權重越大，能有效平滑換手率！) |
| `ts_std_dev(x, d)` | `ts_std_dev(close, 20)` | 過去 `d` 天滾動歷史標準差 (波動率) |
| `ts_correlation(x, y, d)` | `ts_correlation(close, volume, 20)` | 過去 `d` 天 `x` 與 `y` 的滾動皮爾森相關係數 |
| `ts_delta(x, d)` | `ts_delta(close, 5)` | 今日數值減去 `d` 天前數值的變動量 ($x_t - x_{t-d}$) |
| `ts_delay(x, d)` | `ts_delay(close, 1)` | 取 `d` 天前的歷史數值 ($x_{t-d}$) |
| `ts_argmax(x, d)` | `ts_argmax(high, 20)` | 過去 `d` 天內最大值出現在第幾天 |
| `ts_argmin(x, d)` | `ts_argmin(low, 20)` | 過去 `d` 天內最小值出現在第幾天 |

---

## 3. 邏輯與條件運算子 (Logical & Conditional)

| 運算子 | 範例 | 數理語意 |
| :--- | :--- | :--- |
| `signed_power(x, y)` | `signed_power(returns, 2)` | 保留符號的冪次方運算：$\text{sign}(x) \cdot |x|^y$ |
| `x ? y : z` | `close > open ? 1 : -1` | 三元條件運算式 |
| `max(x, y)` / `min(x, y)` | `max(returns, 0)` | 兩變量取較大值/較小值 |

---

## 💡 Alpha 優化實戰技巧 (Top Performance Tips)

1. **降換手率神器：`ts_decay_linear(alpha, d)`**
   - 若回測出現 `Turnover > 70%`，只要在外層包裹 `ts_decay_linear(alpha, 3)` 或 `ts_decay_linear(alpha, 5)`，換手率通常可立即降低 30%~50%，且往往能提升 Sharpe 比率。
2. **行業中性化必備：`group_neutralize(alpha, subindustry)`**
   - 原始 Alpha 容易受到單一強勢族群（如半導體爆發）干擾。加入中性化後，Alpha 考察的是「同業中誰最強」，能有效通過不同年份與子股票池的穩定性檢驗。
3. **動量與反轉結合**：
   - 短週期（1~5天）通常呈現流動性過熱之**反轉 (Mean Reversion)** 特徵（使用負號 `-`）。
   - 中長週期（20~60天）通常呈現機構建倉之**動量 (Momentum)** 特徵。
